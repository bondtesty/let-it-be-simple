#!/usr/bin/env python3
"""Deployed one-step map retains boundary timing (camera-ready).

At the 33 screened high-sigma_1 anchors (H20 uniform policy), the
gripper-channel peak h* of the one-step map's leading output direction u_1
vs the recorded gripper transition step; the pure-noise control (screened at
its own anchors) peaks systematically early.

Data: pi0-endpoint-geometry result JSONs (read-only; under LBS_RESULTS_ROOT).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_ROOT = Path(os.environ.get("LBS_RESULTS_ROOT", "results"))
ROOT = RESULTS_ROOT / "pi0-endpoint-geometry/results"
REPO = Path(__file__).resolve().parents[1]

C_1STEP = "#e67e22"  # deployed one-step map
C_PN = "#27ae60"     # pure-noise control


def main() -> None:
    al = json.loads((ROOT / "pi0_jflow_j1step_alignment.json").read_text())
    recs = al["records"]

    fig, ax = plt.subplots(1, 1, figsize=(4.0, 3.2), constrained_layout=True)

    pop = json.loads((ROOT / "pi0_pop_switch.json").read_text())
    high = {(h["episode"], h["frame"], h["source"]): h for h in pop["high"]}
    pn = json.loads((ROOT / "pi0_purenoise_popswitch.json").read_text())["points"]
    b_i = np.array([high[(r["episode"], r["frame"], r["source"])]["b_raw"] - r["frame"]
                    for r in recs if r["m0"] is not None])
    h_i = np.array([r["h_star_1step"] for r in recs if r["m0"] is not None])
    b_p = np.array([p["b_raw"] - p["frame"] for p in pn])
    h_p = np.array([p["h_star"] for p in pn])
    ax.plot([-1, 21], [-1, 21], color="#7f8c8d", ls="--", lw=1.1)
    ax.plot(b_p, h_p, "^", color=C_PN, ms=4.2, alpha=0.45,
            label=rf"pure-noise one-step ($n={len(pn)}$)")
    ax.plot(b_i, h_i, "o", color=C_1STEP, ms=5.0, alpha=0.85,
            label=rf"interp one-step ($n={len(b_i)}$)")
    ax.set_xlabel("recorded gripper transition (chunk step)", fontsize=9.5)
    ax.set_ylabel(r"$u_1$ gripper peak $h^*$", fontsize=9.5)
    ax.set_xlim(-0.5, 20.5)
    ax.set_ylim(-0.5, 20.5)
    ax.legend(fontsize=8, loc="upper left", frameon=True, facecolor="white",
              edgecolor="#e0e0e0", framealpha=0.95)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.tick_params(labelsize=8)

    out = REPO / "figures" / "app-onestep-geometry"
    fig.savefig(out.with_suffix(".png"), dpi=300)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[*] saved {out}.png/.pdf")


if __name__ == "__main__":
    sys.exit(main())
