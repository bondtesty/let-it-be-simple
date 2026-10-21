"""PERSISTENT generated
timing + right-censoring, applied offline to the saved grip traces.

Decoder states (consistent with the canonical raw-event detector, PERSIST=3):
  persistent     first crossing whose new sign holds >= PERSIST coordinates
  transient      crossing verifiably fails persistence
  censored       crossing begins in the final PERSIST-1 positions
                 (persistence unverifiable because the chunk ends)
  none           no crossing of the pre-chunk side

D1 is reported primarily on the clearly-observable persistent subset;
right-censored cases stay a separate descriptive category. For D2/D3, a
perturbation that changes only censored-vs-observable state (same crossing
index) is NOT a timing switch and is recorded separately.

Run (local): uv run --with numpy experiments/pi0-endpoint-geometry/pi0_pop_recode3.py
"""

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"
PERSIST = 3
THR = 0.049703  # derived programmatically in pi0_pop_recode2.py


def decode(g, g_prev, thr=THR, persist=PERSIST):
    """Scan for the FIRST run of >= persist consecutive coordinates on the
    non-pre-chunk side (generated-commit analogue of the canonical
    persistent raw-event detector).

    Edge cases handled explicitly (v2):
    - an early transient blip does NOT terminate the scan: a later
      persistent crossing is still found; "transient" means transient-ONLY;
    - a run that reaches the chunk end with length < persist is
      right-CENSORED (unverifiable), while an observed return to the
      pre-chunk side is a verifiable transient blip (scan continues);
    - exact-threshold values never produce sign()==0: sides are the
      deterministic binary map (x >= thr) -> +1 else -1, for g and g_prev.
    """
    sgn = np.where(np.asarray(g, dtype=float) >= thr, 1.0, -1.0)
    s_prev = 1.0 if g_prev >= thr else -1.0
    n = len(g)
    first_cross = None
    i = 0
    while i < n:
        if sgn[i] != s_prev:
            if first_cross is None:
                first_cross = int(i)
            j = i
            while j < n and sgn[j] != s_prev:
                j += 1
            run_len = j - i  # run covers [i, j)
            if run_len >= persist:
                return {"state": "persistent", "m": int(i),
                        "first_crossing": first_cross}
            if j == n:
                # reaches the chunk end shorter than persist: unverifiable
                return {"state": "censored", "m": int(i),
                        "first_crossing": first_cross}
            # verifiable transient blip: resume scanning after the return
            i = j
        else:
            i += 1
    if first_cross is not None:
        return {"state": "transient", "m": first_cross,
                "first_crossing": first_cross}
    return {"state": "none", "m": None, "first_crossing": None}


def main():
    d = json.loads((RESULTS_DIR / "pi0_pop_switch.json").read_text())
    thr = d["d1"].get("threshold_derived", THR)

    def classify(pt):
        out = {}
        for e_str, tr in pt["grip_traces"].items():
            out[e_str] = decode(np.asarray(tr), pt["g_prev"], thr)
        return out

    def switch_flag(dec):
        """Timing switch = m differs between +eps/-eps at some eps, or a
        crossing appears/disappears. State-only changes (same m, different
        state) are recorded separately and do NOT count."""
        timing, state_only, removed = 0, 0, 0
        for e in d["eps_grid"]:
            if e <= 0:
                continue
            a, b = dec[str(e)], dec[str(-e)]
            ma, mb = a["m"], b["m"]
            sa, sb = a["state"], b["state"]
            if ma is not None and mb is not None:
                if ma != mb:
                    timing = 1
                elif sa != sb:
                    state_only = 1
            elif (ma is None) != (mb is None):
                removed = 1
        return int(timing or removed), int(state_only)

    for group in ("high", "controls"):
        for p in d[group]:
            dec = classify(p)
            p["decode"] = dec
            p["m0"], p["state0"] = dec["0.0"]["m"], dec["0.0"]["state"]
            S, state_only = switch_flag(dec)
            p["S"] = S
            p["S_state_only_change"] = state_only

    hi = d["high"]
    states = [p["state0"] for p in hi]
    pers = [p for p in hi if p["state0"] == "persistent"]
    exact = sum(1 for p in pers if p["h_star"] == p["m0"])
    within1 = sum(1 for p in pers if abs(p["h_star"] - p["m0"]) <= 1)
    d["d1"] = {
        "n": len(hi),
        "state_counts": {s: int(states.count(s)) for s in
                         ("persistent", "transient", "censored", "none")},
        "P1_within1_persistent": within1 / len(pers),
        "P0_persistent": exact / len(pers),
        "n_persistent": len(pers),
        "within1_persistent": within1,
        "primary": "P(|h*-m|<=1 | persistent observable)",
        "decoder": f"persistent PERSIST={PERSIST}, censoring-aware, thr={thr:.6f} (derived)",
        "threshold_derived": thr,
        "nonpersistent_cases": [{"episode": p["episode"], "frame": p["frame"],
                                 "source": p["source"], "state": p["state0"],
                                 "m": p["m0"], "h_star": p["h_star"]}
                                for p in hi if p["state0"] != "persistent"],
    }

    def rates(points):
        per_win = {}
        for p in points:
            per_win.setdefault((p["episode"], p["b_raw_canonical"]), []).append(p["S"])
        n_state_only = sum(p.get("S_state_only_change", 0) for p in points)
        return {"pooled": float(np.mean([p["S"] for p in points])),
                "per_window": {f"ep{e}_b{b}": float(np.mean(v))
                               for (e, b), v in per_win.items()},
                "n": len(points), "n_state_only_excluded": n_state_only}

    d["d2"] = rates(hi)
    d["d3"] = rates(d["controls"])

    print("D1 states:", d["d1"]["state_counts"])
    print(f"D1 persistent: {within1}/{len(pers)} within +-1 "
          f"({d['d1']['P1_within1_persistent']:.3f}); exact {exact}/{len(pers)}")
    print("non-persistent:", d["d1"]["nonpersistent_cases"])
    print(f"D2 P_switch^high={d['d2']['pooled']:.3f} "
          f"(state-only excluded: {d['d2']['n_state_only_excluded']}) "
          f"per-win {d['d2']['per_window']}")
    print(f"D3 P_switch^low ={d['d3']['pooled']:.3f} "
          f"(state-only excluded: {d['d3']['n_state_only_excluded']}) "
          f"per-win {d['d3']['per_window']}")

    with (RESULTS_DIR / "pi0_pop_switch.json").open("w") as fh:
        json.dump(d, fh, indent=1)
    print("wrote", RESULTS_DIR / "pi0_pop_switch.json")


if __name__ == "__main__":
    main()
