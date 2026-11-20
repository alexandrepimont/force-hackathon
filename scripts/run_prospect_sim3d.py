"""Extrapolate the 1D pseudo-well simulation to 3D over a decimated grid, with a blind test at the held-out wells.

Examples:
    python scripts/run_prospect_sim3d.py                         # blind test + grid (configs/prospect_sim3d.yaml)
    python scripts/run_prospect_sim3d.py --blind-only            # only the held-out well check (fast)
    python scripts/run_prospect_sim3d.py --il-step 32 --xl-step 48 --n 3000 --name coarse

Results: outputs/simulation3d/<name>.npz/.json and <name>_blind.csv / _blind.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from avo import prospect_sim3d as p3  # noqa: E402

OUT = ROOT / "outputs" / "simulation3d"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="grid3d")
    ap.add_argument("--il-step", type=int)
    ap.add_argument("--xl-step", type=int)
    ap.add_argument("--n", type=int, help="pseudo-wells per depth band")
    ap.add_argument("--blind-only", action="store_true")
    ap.add_argument("--no-blind", action="store_true")
    a = ap.parse_args()

    c3 = p3.load_config(ROOT)
    if a.il_step:
        c3["grid"]["il_step"] = a.il_step
    if a.xl_step:
        c3["grid"]["xl_step"] = a.xl_step
    if a.n:
        c3["library"]["n"] = a.n
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ctx = p3.Context(ROOT, c3)
    print(f"prior ready ({time.time() - t0:.0f} s); surface from "
          + ", ".join(f"{p['well']} {p['twt_ms']:.0f} ms" for p in ctx.surface["points"]))

    if not a.no_blind:
        tables, infos = [], []
        for well in ctx.held_out:
            tb, info = p3.blind_check(ctx, well)
            tables.append(tb)
            infos.append(info)
            print(json.dumps(info))
        blind = pd.concat(tables, ignore_index=True)
        blind.to_csv(OUT / f"{a.name}_blind.csv", index=False)
        (OUT / f"{a.name}_blind.json").write_text(json.dumps(infos, indent=2), encoding="utf-8")
        print(blind.to_string(index=False))
    if a.blind_only:
        return

    last = [0.0]

    def progress(f, msg):
        if f - last[0] >= 0.05 or f >= 1:
            last[0] = f
            print(f"  {f:5.0%} {msg} ({time.time() - t0:.0f} s)", flush=True)

    meta = p3.run_grid(ROOT, a.name, progress=progress, ctx=ctx)
    print(f"{a.name}: {meta['n_traces']} traces, bands {meta['depth_bands_m']}, median beta {meta['beta_median']}, "
          f"low support {meta['low_support_fraction']:.0%}, median fit corr {meta['fit_corr_median']}, "
          f"{meta['runtime_s']} s")


if __name__ == "__main__":
    main()
