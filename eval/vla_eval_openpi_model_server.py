#!/usr/bin/env python3
"""OpenPI model server for AllenAI vla-evaluation-harness.

Run with PYTHONPATH including:
  - this repo's src/
  - packages/openpi-client/src/
  - third_party/vla-evaluation-harness/src/
"""

from __future__ import annotations

import dataclasses
import logging
import os
import time
from typing import Any

import numpy as np
from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.specs import IMAGE_RGB
from vla_eval.specs import LANGUAGE
from vla_eval.specs import RAW
from vla_eval.specs import DimSpec
from vla_eval.types import Action
from vla_eval.types import Observation

logger = logging.getLogger(__name__)


class OpenPIModelServer(PredictModelServer):
    """Batched OpenPI policy server for LIBERO/LIBERO-Plus harness eval."""

    def __init__(
        self,
        config_name: str,
        checkpoint: str,
        image_key: str = "observation/image",
        wrist_image_key: str | None = "observation/wrist_image",
        state_key: str | None = "observation/state",
        state_dim: int = 8,
        image_resolution: int | None = 224,
        action_horizon: int | None = None,
        flow_steps: int = 1,
        solver: str = "euler",
        prediction_type: str | None = None,
        pure_noise_input: bool | None = None,
        action_start_mode: str | None = None,
        learned_action_start_init: str | None = None,
        learned_action_start_init_std: float | None = None,
        action_start_training: str | None = None,
        condition_ablation: str = "full",
        *,
        chunk_size: int = 10,
        action_ensemble: str = "newest",
        ema_alpha: float = 0.5,
        max_batch_size: int = 16,
        max_wait_time: float = 0.01,
        pad_to_max_batch: bool = False,
        continuous_inference: bool = False,
        laas: bool = False,
        hz: float = 10.0,
    ) -> None:
        super().__init__(
            chunk_size=chunk_size,
            action_ensemble=action_ensemble,
            ema_alpha=ema_alpha,
            max_batch_size=max_batch_size,
            max_wait_time=max_wait_time,
            continuous_inference=continuous_inference,
            laas=laas,
            hz=hz,
        )
        self.config_name = config_name
        self.checkpoint = checkpoint
        self.image_key = image_key
        self.wrist_image_key = None if wrist_image_key in (None, "None", "none") else wrist_image_key
        self.state_key = None if state_key in (None, "None", "none") else state_key
        self.state_dim = state_dim
        self.image_resolution = image_resolution
        self.action_horizon = action_horizon
        self.flow_steps = flow_steps
        self.solver = solver
        self.prediction_type = prediction_type
        self.pure_noise_input = pure_noise_input
        self.action_start_mode = action_start_mode
        self.learned_action_start_init = learned_action_start_init
        self.learned_action_start_init_std = learned_action_start_init_std
        self.action_start_training = action_start_training
        self.condition_ablation = condition_ablation
        self._policy = None
        self._pad_to_max_batch = pad_to_max_batch
        self._log_batch_stats = os.environ.get("OPENPI_EVAL_LOG_BATCH_STATS", "0") == "1"
        self._batch_log_interval = max(1, int(os.environ.get("OPENPI_EVAL_BATCH_LOG_INTERVAL", "20")))
        self._batch_count = 0

    def _load_model(self) -> None:
        if self._policy is not None:
            return

        from openpi.policies import policy_config
        from openpi.training import config as _config

        logger.info("Loading OpenPI config: %s", self.config_name)
        config = _config.get_config(self.config_name)

        model_kwargs = {}
        if self.action_horizon is not None:
            model_kwargs["action_horizon"] = self.action_horizon
            logger.info("Overriding action_horizon to %s", self.action_horizon)
        if self.prediction_type is not None:
            model_kwargs["prediction_type"] = self.prediction_type
            logger.info("Overriding prediction_type to %s", self.prediction_type)
        if self.pure_noise_input is not None:
            model_kwargs["pure_noise_input"] = self.pure_noise_input
            logger.info("Overriding pure_noise_input to %s", self.pure_noise_input)
        if self.action_start_mode is not None:
            model_kwargs["action_start_mode"] = self.action_start_mode
            logger.info("Overriding action_start_mode to %s", self.action_start_mode)
        if self.learned_action_start_init is not None:
            model_kwargs["learned_action_start_init"] = self.learned_action_start_init
            logger.info("Overriding learned_action_start_init to %s", self.learned_action_start_init)
        if self.learned_action_start_init_std is not None:
            model_kwargs["learned_action_start_init_std"] = self.learned_action_start_init_std
            logger.info("Overriding learned_action_start_init_std to %s", self.learned_action_start_init_std)
        if self.action_start_training is not None:
            model_kwargs["action_start_training"] = self.action_start_training
            logger.info("Overriding action_start_training to %s", self.action_start_training)
        if model_kwargs:
            config = dataclasses.replace(config, model=dataclasses.replace(config.model, **model_kwargs))

        if self.condition_ablation != "full":
            config = dataclasses.replace(
                config,
                data=dataclasses.replace(config.data, condition_ablation=self.condition_ablation),
            )
            logger.info("Using condition_ablation=%s", self.condition_ablation)

        sample_kwargs = {}
        if self.flow_steps > 0:
            sample_kwargs["num_steps"] = self.flow_steps
        if self.solver != "euler":
            sample_kwargs["solver"] = self.solver

        logger.info("Loading OpenPI checkpoint: %s", self.checkpoint)
        self._policy = policy_config.create_trained_policy(
            config,
            self.checkpoint,
            sample_kwargs=sample_kwargs or None,
        )
        logger.info("OpenPI policy loaded")

    def _maybe_resize(self, image: np.ndarray) -> np.ndarray:
        image = np.asarray(image, dtype=np.uint8)
        if self.image_resolution is None or image.shape[:2] == (self.image_resolution, self.image_resolution):
            return image
        from PIL import Image

        pil = Image.fromarray(image)
        pil = pil.resize((self.image_resolution, self.image_resolution), Image.Resampling.BILINEAR)
        return np.asarray(pil)

    def _to_openpi_obs(self, obs: Observation) -> dict[str, Any]:
        openpi_obs: dict[str, Any] = {}
        images = obs.get("images", {})
        image_values = list(images.values()) if isinstance(images, dict) else []

        base_image = np.asarray(image_values[0], dtype=np.uint8) if image_values else np.zeros((256, 256, 3), np.uint8)
        base_image = self._maybe_resize(base_image)
        openpi_obs[self.image_key] = base_image

        if self.wrist_image_key:
            wrist_image = (
                np.asarray(image_values[1], dtype=np.uint8) if len(image_values) > 1 else np.zeros_like(base_image)
            )
            openpi_obs[self.wrist_image_key] = self._maybe_resize(wrist_image)

        openpi_obs["prompt"] = obs.get("task_description", "")

        if self.state_key:
            raw_state = obs.get("states", obs.get("state"))
            if raw_state is None:
                raw_state = np.zeros(self.state_dim, dtype=np.float64)
            openpi_obs[self.state_key] = np.asarray(raw_state, dtype=np.float64)

        return openpi_obs

    def get_observation_params(self) -> dict[str, Any]:
        return {
            "send_wrist_image": self.wrist_image_key is not None,
            "send_state": self.state_key is not None,
        }

    def get_action_spec(self) -> dict[str, DimSpec]:
        return {"actions": RAW}

    def get_observation_spec(self) -> dict[str, DimSpec]:
        spec: dict[str, DimSpec] = {"agentview": IMAGE_RGB, "language": LANGUAGE}
        if self.wrist_image_key is not None:
            spec["wrist"] = IMAGE_RGB
        if self.state_key is not None:
            spec["state"] = RAW
        return spec

    def predict(self, obs: Observation, ctx: SessionContext) -> Action:
        self._load_model()
        assert self._policy is not None
        result = self._policy.infer(self._to_openpi_obs(obs))
        return {"actions": result["actions"]}

    def predict_batch(self, obs_batch: list[Observation], ctx_batch: list[SessionContext]) -> list[Action]:
        self._load_model()
        assert self._policy is not None
        requested_batch_size = len(obs_batch)
        effective_obs_batch = obs_batch
        if self._pad_to_max_batch and 0 < requested_batch_size < self.max_batch_size:
            effective_obs_batch = [*obs_batch, *([obs_batch[-1]] * (self.max_batch_size - requested_batch_size))]
        openpi_batch = [self._to_openpi_obs(obs) for obs in effective_obs_batch]
        start = time.monotonic()
        results = self._policy.infer_batch(openpi_batch)[:requested_batch_size]
        self._batch_count += 1
        if self._log_batch_stats and self._batch_count % self._batch_log_interval == 0:
            timing = results[0].get("policy_timing", {}) if results else {}
            logger.info(
                "predict_batch count=%d batch_size=%d effective_batch_size=%d latency_ms=%.1f policy_total_ms=%.1f model_ms=%.1f input_ms=%.1f stack_ms=%.1f device_get_ms=%.1f output_ms=%.1f",
                self._batch_count,
                requested_batch_size,
                len(openpi_batch),
                (time.monotonic() - start) * 1000,
                float(timing.get("total_ms", 0.0)),
                float(timing.get("infer_ms", 0.0)),
                float(timing.get("input_transform_ms", 0.0)),
                float(timing.get("stack_ms", 0.0)),
                float(timing.get("device_get_ms", 0.0)),
                float(timing.get("output_transform_ms", 0.0)),
            )
        return [{"actions": result["actions"]} for result in results]


if __name__ == "__main__":
    from vla_eval.model_servers.serve import run_server

    run_server(OpenPIModelServer)
