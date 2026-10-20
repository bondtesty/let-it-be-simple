#!/usr/bin/env python3
"""
MNIST Image Captioning - Flow Matching v3 Ablation (Mask Type)
==============================================================
基于v3架构，消融不同mask类型：
  - causal: [IMG]+text，text使用causal mask
  - bidirectional: [IMG]+text，双向attention（无mask）
  - no_mask: 不加任何mask
"""

import os
import math
import random
import argparse
import json
from pathlib import Path
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image
import numpy as np
from tqdm import tqdm

NUM2WORD = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
    5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"
}

class MNISTCaptionDataset(Dataset):
    def __init__(self, root="./data", train=True, mode="single", max_len=16, 
                 img_size=64, augment=False):
        self.mode = mode
        self.max_len = max_len
        self.img_size = img_size
        self.augment = augment and train
        self.mnist = datasets.MNIST(root=root, train=train, download=True)
        self.digit_indices = {i: [] for i in range(10)}
        for idx, (_, label) in enumerate(self.mnist):
            self.digit_indices[label].append(idx)
        self.base_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        if self.augment:
            self.aug_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomRotation(degrees=5),
            ])
        else:
            self.aug_transform = None
    
    def __len__(self):
        if self.mode == "single":
            return len(self.mnist)
        elif self.mode == "grid2x2":
            return len(self.mnist) // 4
    
    def _get_single(self, idx):
        img, label = self.mnist[idx]
        img = img.convert("RGB")
        if self.aug_transform:
            img = self.aug_transform(img)
        img_tensor = self.base_transform(img)
        caption = NUM2WORD[label]
        return img_tensor, caption, [label]
    
    def _get_grid2x2(self, idx):
        labels = [random.randint(0, 9) for _ in range(4)]
        imgs = []
        for l in labels:
            i = random.choice(self.digit_indices[l])
            img, _ = self.mnist[i]
            imgs.append(np.array(img))
        top = np.concatenate([imgs[0], imgs[1]], axis=1)
        bottom = np.concatenate([imgs[2], imgs[3]], axis=1)
        grid = np.concatenate([top, bottom], axis=0)
        grid_img = Image.fromarray(grid).convert("RGB")
        if self.aug_transform:
            grid_img = self.aug_transform(grid_img)
        img_tensor = self.base_transform(grid_img)
        caption = " ".join(NUM2WORD[l] for l in labels)
        return img_tensor, caption, labels
    
    def __getitem__(self, idx):
        if self.mode == "single":
            img, caption, labels = self._get_single(idx)
        else:
            img, caption, labels = self._get_grid2x2(idx)
        return {
            "image": img,
            "caption": caption,
            "labels": labels,
            "image_id": idx
        }

class DigitTokenizer:
    def __init__(self, max_len=16):
        self.max_len = max_len
        self.vocab_words = ["<pad>", "<unk>", "<eos>"] + list(NUM2WORD.values())
        self.word2idx = {w: i for i, w in enumerate(self.vocab_words)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self.pad_token_id = 0
        self.eos_token_id = 2
        self.vocab_size = len(self.vocab_words)
    
    def encode(self, text: str, include_eos: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        words = text.split()
        if include_eos:
            words = words + ["<eos>"]
        words = words[:self.max_len]
        ids = [self.word2idx.get(w, 1) for w in words]
        padding_mask = torch.zeros(self.max_len, dtype=torch.bool)
        if len(ids) < self.max_len:
            padding_mask[len(ids):] = True
            ids.extend([0] * (self.max_len - len(ids)))
        return torch.tensor(ids, dtype=torch.long), padding_mask
    
    def decode(self, ids: torch.Tensor, stop_at_eos: bool = True) -> str:
        words = []
        for i in ids:
            idx = int(i)
            if idx == 0:
                break
            if idx == self.eos_token_id and stop_at_eos:
                break
            words.append(self.idx2word.get(idx, "<unk>"))
        return " ".join(words)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out

class EnhancedImageEncoder(nn.Module):
    def __init__(self, in_channels=3, d_model=256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.stage1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            ResidualBlock(128),
            ResidualBlock(128),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            ResidualBlock(256),
            nn.AdaptiveAvgPool2d(1)
        )
        self.proj = nn.Sequential(
            nn.Linear(256, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
    
    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = x.squeeze(-1).squeeze(-1)
        return self.proj(x)

class FlowMatchingTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=8, 
                 max_len=16, dropout=0.1, mask_type="causal"):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.mask_type = mask_type
        
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.time_embed = nn.Sequential(
            nn.Linear(1, d_model), nn.SiLU(),
            nn.Linear(d_model, d_model)
        )
        self.self_cond_proj = nn.Linear(d_model * 2, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4*d_model,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model)
        )
        self._init_weights()
        
        # 不同mask类型
        if mask_type == "causal":
            mask = torch.zeros(1 + max_len, 1 + max_len, dtype=torch.bool)
            for i in range(max_len):
                mask[i + 1, i + 1:] = True
            self.register_buffer('causal_mask', mask)
        else:
            self.causal_mask = None
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, xt, t, image_cond, x_pred=None, padding_mask=None):
        B, L, D = xt.shape
        if x_pred is not None:
            x_input = self.self_cond_proj(torch.cat([xt, x_pred], dim=-1))
        else:
            x_input = xt
        t_emb = self.time_embed(t)
        x_input = x_input + t_emb.unsqueeze(1)
        img_token = image_cond.unsqueeze(1)
        seq = torch.cat([img_token, x_input], dim=1)
        pos = torch.cat([
            torch.zeros(B, 1, D, device=seq.device),
            self.pos_embed[:, :L, :].expand(B, -1, -1)
        ], dim=1)
        seq = seq + pos
        
        if padding_mask is not None:
            extended_mask = torch.cat([
                torch.zeros(B, 1, dtype=torch.bool, device=padding_mask.device),
                padding_mask
            ], dim=1)
        else:
            extended_mask = None
        
        if self.mask_type == "causal" and self.causal_mask is not None:
            causal_float = torch.zeros_like(self.causal_mask, dtype=torch.float)
            causal_float = causal_float.masked_fill(self.causal_mask, float('-inf'))
            out = self.transformer(seq, mask=causal_float, 
                                  src_key_padding_mask=extended_mask)
        else:
            out = self.transformer(seq, src_key_padding_mask=extended_mask)
        
        out = out[:, 1:, :]
        return self.out_proj(out)

class ImageToTextFlowMatching(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=8, 
                 max_len=16, dropout=0.1, img_channels=3, mask_type="causal"):
        super().__init__()
        self.image_encoder = EnhancedImageEncoder(img_channels, d_model)
        self.transformer = FlowMatchingTransformer(
            vocab_size, d_model, nhead, num_layers, max_len, dropout, mask_type
        )
        self.d_model = d_model
        self.max_len = max_len
        self.mask_type = mask_type
    
    def forward(self, images, tokens, t, x_pred=None, padding_mask=None):
        image_cond = self.image_encoder(images)
        x1 = self.transformer.token_embed(tokens)
        x0 = torch.randn_like(x1)
        t_exp = t.unsqueeze(-1)
        xt = (1 - t_exp) * x0 + t_exp * x1
        v_pred = self.transformer(xt, t, image_cond, x_pred, padding_mask)
        v_target = x1 - x0
        return v_pred, v_target, x1, xt
    
    @torch.no_grad()
    def sample(self, images, tokenizer, n_steps=10, temperature=1.0):
        B = images.shape[0]
        device = images.device
        image_cond = self.image_encoder(images)
        xt = torch.randn(B, self.max_len, self.d_model, device=device) * temperature
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((B, 1), i * dt, device=device)
            v = self.transformer(xt, t, image_cond, x_pred=None, padding_mask=None)
            xt = xt + v * dt
        logits = torch.matmul(xt, self.transformer.token_embed.weight.t())
        token_ids = logits.argmax(dim=-1)
        return token_ids

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_step = 0
    
    def step(self):
        self.current_step += 1
        if self.current_step < self.warmup_steps:
            lr = self.base_lr * self.current_step / self.warmup_steps
        else:
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

def train_epoch(model, dataloader, optimizer, scheduler, device, 
                use_self_cond=False, self_cond_prob=0.5, alpha_ce=0.5,
                grad_accum_steps=1, scaler=None):
    model.train()
    total_loss = 0.0
    total_tokens = 0
    total_mse = 0.0
    total_ce = 0.0
    pbar = tqdm(dataloader, desc="Train")
    optimizer.zero_grad()
    for step, batch in enumerate(pbar):
        images = batch["image"].to(device)
        tokens = batch["tokens"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        B = images.shape[0]
        t = torch.rand(B, 1, device=device)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                x_pred = None
                if use_self_cond and random.random() < self_cond_prob:
                    with torch.no_grad():
                        _, _, x1, xt = model(images, tokens, t, x_pred=None, padding_mask=padding_mask)
                        v_pred_1 = model.transformer(xt, t, model.image_encoder(images),
                                                    x_pred=None, padding_mask=padding_mask)
                        x_pred = xt + v_pred_1 * (1 - t.unsqueeze(-1))
                        x_pred = x_pred.detach()
                v_pred, v_target, x1, xt = model(images, tokens, t, x_pred, padding_mask)
                loss_per_token = ((v_pred - v_target) ** 2).mean(dim=-1)
                mse_loss = (loss_per_token * (~padding_mask)).sum() / (~padding_mask).sum()
                t_exp = t.unsqueeze(-1)
                x_pred_clean = xt + v_pred * (1 - t_exp)
                logits = torch.matmul(x_pred_clean, model.transformer.token_embed.weight.t())
                ce_loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    tokens.view(-1),
                    reduction='none'
                )
                ce_loss = (ce_loss * (~padding_mask).view(-1)).sum() / (~padding_mask).sum()
                loss = mse_loss + alpha_ce * ce_loss
                loss = loss / grad_accum_steps
            scaler.scale(loss).backward()
        else:
            x_pred = None
            if use_self_cond and random.random() < self_cond_prob:
                with torch.no_grad():
                    _, _, x1, xt = model(images, tokens, t, x_pred=None, padding_mask=padding_mask)
                    v_pred_1 = model.transformer(xt, t, model.image_encoder(images),
                                                x_pred=None, padding_mask=padding_mask)
                    x_pred = xt + v_pred_1 * (1 - t.unsqueeze(-1))
                    x_pred = x_pred.detach()
            v_pred, v_target, x1, xt = model(images, tokens, t, x_pred, padding_mask)
            loss_per_token = ((v_pred - v_target) ** 2).mean(dim=-1)
            mse_loss = (loss_per_token * (~padding_mask)).sum() / (~padding_mask).sum()
            t_exp = t.unsqueeze(-1)
            x_pred_clean = xt + v_pred * (1 - t_exp)
            logits = torch.matmul(x_pred_clean, model.transformer.token_embed.weight.t())
            ce_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                tokens.view(-1),
                reduction='none'
            )
            ce_loss = (ce_loss * (~padding_mask).view(-1)).sum() / (~padding_mask).sum()
            loss = mse_loss + alpha_ce * ce_loss
            loss = loss / grad_accum_steps
            loss.backward()
        if (step + 1) % grad_accum_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        total_loss += loss.item() * grad_accum_steps * (~padding_mask).sum().item()
        total_mse += mse_loss.item() * (~padding_mask).sum().item()
        total_ce += ce_loss.item() * (~padding_mask).sum().item()
        total_tokens += (~padding_mask).sum().item()
        if step % 50 == 0:
            pbar.set_postfix({
                "loss": f"{loss.item() * grad_accum_steps:.3f}", 
                "mse": f"{mse_loss.item():.3f}", 
                "ce": f"{ce_loss.item():.3f}",
                "lr": f"{scheduler.optimizer.param_groups[0]['lr']:.2e}"
            })
    return total_loss / total_tokens, total_mse / total_tokens, total_ce / total_tokens

@torch.no_grad()
def evaluate(model, dataloader, tokenizer, device, n_steps=10):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0
    total_token_count = 0
    exact_matches = 0
    total_samples = 0
    all_preds = []
    all_gts = []
    for batch in tqdm(dataloader, desc="Eval"):
        images = batch["image"].to(device)
        tokens = batch["tokens"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        captions = batch["caption"]
        B = images.shape[0]
        t = torch.rand(B, 1, device=device)
        v_pred, v_target, _, _ = model(images, tokens, t, x_pred=None, padding_mask=padding_mask)
        loss_per_token = ((v_pred - v_target) ** 2).mean(dim=-1)
        loss = (loss_per_token * (~padding_mask)).sum() / (~padding_mask).sum()
        total_loss += loss.item() * (~padding_mask).sum().item()
        total_tokens += (~padding_mask).sum().item()
        pred_ids = model.sample(images, tokenizer, n_steps=n_steps)
        for i in range(B):
            pred_text = tokenizer.decode(pred_ids[i], stop_at_eos=True)
            gt_text = captions[i]
            all_preds.append(pred_text)
            all_gts.append(gt_text)
            if pred_text == gt_text:
                exact_matches += 1
            total_samples += 1
            pred_words = pred_text.split()
            gt_words = gt_text.split()
            for pw, gw in zip(pred_words, gt_words):
                if pw == gw:
                    correct_tokens += 1
                total_token_count += 1
    metrics = {
        "loss": total_loss / total_tokens,
        "token_acc": correct_tokens / total_token_count if total_token_count > 0 else 0,
        "exact_match": exact_matches / total_samples if total_samples > 0 else 0,
    }
    return metrics, all_preds, all_gts

def collate_fn(batch, tokenizer):
    images = torch.stack([b["image"] for b in batch])
    captions = [b["caption"] for b in batch]
    labels = [b["labels"] for b in batch]
    tokens_list = []
    masks_list = []
    for cap in captions:
        ids, pad_mask = tokenizer.encode(cap, include_eos=True)
        tokens_list.append(ids)
        masks_list.append(pad_mask)
    return {
        "image": images,
        "tokens": torch.stack(tokens_list),
        "padding_mask": torch.stack(masks_list),
        "caption": captions,
        "labels": labels
    }

def run_experiment(mask_type, args):
    print(f"\n{'='*60}")
    print(f"Running experiment: mask_type={mask_type}")
    print(f"{'='*60}")
    
    save_dir = os.path.join(args.save_dir, mask_type)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device(args.device)
    tokenizer = DigitTokenizer(max_len=args.max_len)
    
    train_ds = MNISTCaptionDataset(root=args.data_dir, train=True, mode=args.mode, 
                                   max_len=args.max_len, augment=args.augment)
    val_ds = MNISTCaptionDataset(root=args.data_dir, train=False, mode=args.mode, 
                                 max_len=args.max_len, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                             num_workers=4, pin_memory=True, 
                             collate_fn=lambda b: collate_fn(b, tokenizer))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=4, pin_memory=True,
                           collate_fn=lambda b: collate_fn(b, tokenizer))
    
    model = ImageToTextFlowMatching(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        max_len=args.max_len,
        img_channels=3,
        mask_type=mask_type
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {n_params:.2f}M")
    print(f"Config: mask={mask_type}, alpha_ce={args.alpha_ce}")
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, 
                           betas=(0.9, 0.98))
    total_steps = len(train_loader) * args.epochs // args.grad_accum
    warmup_steps = len(train_loader) * args.warmup_epochs // args.grad_accum
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps, min_lr=1e-6)
    scaler = torch.cuda.amp.GradScaler() if args.fp16 and device.type == 'cuda' else None
    
    best_exact = 0.0
    history = []
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_loss, train_mse, train_ce = train_epoch(
            model, train_loader, optimizer, scheduler, device,
            alpha_ce=args.alpha_ce, grad_accum_steps=args.grad_accum, scaler=scaler
        )
        metrics, preds, gts = evaluate(model, val_loader, tokenizer, device, n_steps=args.n_steps)
        
        print(f"Train: loss={train_loss:.4f}, mse={train_mse:.4f}, ce={train_ce:.4f}")
        print(f"Val:   loss={metrics['loss']:.4f}, token_acc={metrics['token_acc']:.2%}, exact={metrics['exact_match']:.2%}")
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': metrics['loss'],
            'token_acc': metrics['token_acc'],
            'exact_match': metrics['exact_match']
        })
        
        if epoch % 5 == 0 or metrics['exact_match'] > best_exact:
            print("\n[Samples]")
            for i in range(min(5, len(preds))):
                mark = "✓" if preds[i] == gts[i] else "✗"
                print(f"  {mark} GT: {gts[i]:<25} Pred: {preds[i]}")
        
        if metrics['exact_match'] > best_exact:
            best_exact = metrics['exact_match']
            torch.save({
                "model": model.state_dict(),
                "tokenizer": tokenizer,
                "args": args,
                "history": history
            }, os.path.join(save_dir, f"best_{args.mode}.pt"))
            print(f"*** Saved best (exact={metrics['exact_match']:.2%}) ***")
    
    print(f"\n{'='*60}")
    print(f"Experiment complete! Best exact match: {best_exact:.2%}")
    print(f"{'='*60}")
    
    return {
        "mask_type": mask_type,
        "best_exact": best_exact,
        "best_epoch": max([h['epoch'] for h in history if h['exact_match'] == best_exact], default=0),
        "final_token_acc": history[-1]['token_acc'] if history else 0,
        "history": history
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="grid2x2", choices=["single", "grid2x2"])
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--n_steps", type=int, default=10)
    parser.add_argument("--alpha_ce", type=float, default=0.5)
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default="./checkpoints_v3_ablation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    print("="*60)
    print("MNIST Image Captioning - v3 Mask Ablation")
    print("="*60)
    
    results = {}
    for mask_type in ["causal", "bidirectional", "no_mask"]:
        results[mask_type] = run_experiment(mask_type, args)
    
    # Save comparison
    comparison = {
        mask_type: {
            "best_exact": res["best_exact"],
            "best_epoch": res["best_epoch"],
            "final_token_acc": res["final_token_acc"]
        }
        for mask_type, res in results.items()
    }
    
    with open(os.path.join(args.save_dir, "ablation_results.json"), "w") as f:
        json.dump(comparison, f, indent=2)
    
    print("\n" + "="*60)
    print("ABLATION COMPARISON")
    print("="*60)
    print(f"{'Mask Type':<20} {'Best Exact':<12} {'Best Epoch':<12} {'Final Token Acc':<15}")
    print("-"*60)
    for mask_type, res in comparison.items():
        print(f"{mask_type:<20} {res['best_exact']:<12.2%} {res['best_epoch']:<12} {res['final_token_acc']:<15.2%}")
    print("="*60)

if __name__ == "__main__":
    main()
