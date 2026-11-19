"""Run the 1D prospect (pseudo-well) simulation at a location, or at a well for a blind test.

Examples:
    python scripts/run_prospect_sim.py --well Poseidon-2                  # blind test at a held-out well
    python scripts/run_prospect_sim.py --il 3000 --xl 2000 --t0 3150 --t1 3350 --name prospect_A

Results: outputs/simulation/<name>.npz/.json (+ <name>_blind.csv for wells). Every run is saved on completion.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from avo import prospect_sim as ps  # noqa: E402
from avo.welltie import well_traces  # noqa: E402

CACHE = ROOT / "data" / "poseidon" / "seismic" / "cache"
OUT = ROOT / "outputs" / "simulation"


def grid_xy(il: int, xl: int) -> tuple[float, float]:
    g = np.load(CACHE / "poseidon_near.geometry.npz")
    i = int(np.searchsorted(g["ilines"], il))
    j = int(np.searchsorted(g["xlines"], xl))
    return float(g["cdp_x"][i, j]), float(g["cdp_y"][i, j])


def blind_comparison(res_npz: Path, well: str, t0: float, t1: float, facies: list[str]) -> pd.DataFrame:
    """Compare posterior P10/P50/P90 with the actual (held-out) logs of the well, blocked to 1 ms."""
    d = np.load(res_npz)
    logs = pd.read_parquet(ROOT / "data" / "interim" / "poseidon_logs" / f"{well}.parquet")
    logs = logs[(logs["twt"] >= t0) & (logs["twt"] <= t1)]
    t = d["t_model_ms"]
    rows = []
    ms = np.round(logs["twt"])
    lith = logs.groupby(ms)["lithotype"].agg(lambda s: s.mode().iat[0])
    sand = lith.isin(["sand_high_phi", "sand_low_phi"])
    tol = {"phi": 0.003, "vsh": 0.005, "sw": 0.005, "vp": 25.0, "vs": 20.0, "rho": 0.005}   # ~ one histogram bin
    for prop in ["phi", "vsh", "sw", "vp", "vs", "rho"]:
        actual = logs.groupby(ms)[prop].mean()
        idx = np.searchsorted(t, actual.index.to_numpy()).clip(0, len(t) - 1)
        for name in ("prior", "post"):
            q = d[f"{name}_{prop}_q"][:, idx]
            a = actual.to_numpy()
            ok = np.isfinite(a)
            if prop == "sw":   # Archie Sw is only meaningful in sands; the simulation sets non-reservoir Sw = 1
                ok &= sand.reindex(actual.index).fillna(False).to_numpy()
            if not ok.any():
                continue
            pc = (np.corrcoef(a[ok], q[1][ok])[0, 1]
                  if ok.sum() > 3 and np.std(a[ok]) > 0 and np.std(q[1][ok]) > 0 else np.nan)
            rows.append({"property": prop + (" (sands)" if prop == "sw" else ""), "ensemble": name,
                         "n_ms": int(ok.sum()),
                         "coverage_p10_p90": round(float(((a >= q[0] - tol[prop]) & (a <= q[2] + tol[prop]))[ok].mean()), 3),
                         "mae_p50": round(float(np.abs(a - q[1])[ok].mean()), 4),
                         "corr_p50": round(float(pc), 3)})
    idx = np.searchsorted(t, lith.index.to_numpy()).clip(0, len(t) - 1)
    known = lith.isin(facies).to_numpy()
    hc = logs.groupby(ms)["hc_indicator"].agg(lambda s: s.isin(["gas_indicated", "possible"]).any())
    hc = hc.reindex(lith.index).fillna(False)
    gas_ind = (hc & sand).to_numpy()
    wet_sand = (sand & ~hc).to_numpy()
    si = [facies.index("sand_high_phi"), facies.index("sand_low_phi")]
    for name in ("prior", "post"):
        prob = d[f"{name}_facies_prob"][:, idx]
        pred = np.array(facies)[prob.argmax(axis=0)]
        true_i = np.array([facies.index(x) if x in facies else 0 for x in lith])
        p_true = prob[true_i, np.arange(len(true_i))]
        rows.append({"property": "facies", "ensemble": name, "n_ms": int(known.sum()),
                     "accuracy_map": round(float((pred == lith.to_numpy())[known].mean()), 3),
                     "mean_p_true_facies": round(float(p_true[known].mean()), 3)})
        pg = d[f"{name}_p_gas"][idx] / np.maximum(d[f"{name}_facies_prob"][si][:, idx].sum(axis=0), 1e-6)
        rows.append({"property": "P(gas | sand) at log gas-indicated sands", "ensemble": name,
                     "n_ms": int(gas_ind.sum()),
                     "mean_p": round(float(pg[gas_ind].mean()), 3) if gas_ind.any() else np.nan,
                     "mean_p_other_sands": round(float(pg[wet_sand].mean()), 3) if wet_sand.any() else np.nan})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--well")
    ap.add_argument("--il", type=int)
    ap.add_argument("--xl", type=int)
    ap.add_argument("--t0", type=float, help="window top TWT (ms)")
    ap.add_argument("--t1", type=float, help="window base TWT (ms)")
    ap.add_argument("--n", type=int)
    ap.add_argument("--name")
    a = ap.parse_args()

    cfg, calib, petro = ps.load_configs(ROOT)
    held_out = petro["held_out_wells"]
    tie_cfg = yaml.safe_load((ROOT / "configs" / "well_tie.yaml").read_text(encoding="utf-8"))
    if a.well:
        tie = json.loads((ROOT / "outputs" / "well_ties" / f"{a.well}.json").read_text(encoding="utf-8"))["result"]
        il, xl = tie["inline"], tie["crossline"]
        t0, t1 = (a.t0, a.t1) if a.t0 else tie["window_ms"]
    else:
        il, xl, t0, t1 = a.il, a.xl, a.t0, a.t1
        if None in (il, xl, t0, t1):
            ap.error("give --well or all of --il --xl --t0 --t1")
    name = a.name or (f"well_{a.well}" if a.well else f"il{il}_xl{xl}_{int(t0)}_{int(t1)}")
    x, y = grid_xy(il, xl)

    t_start = time.time()
    prior = ps.build_prior(ROOT / "data" / "interim" / "poseidon_logs", held_out, cfg)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "prior.json").write_text(json.dumps(prior.to_dict(), indent=2), encoding="utf-8")
    td_t, td_z, td_well, td_km = ps.location_td(ROOT, x, y, held_out)
    t_seis, obs = well_traces(CACHE, list(calib["stacks"]), il, xl, tie_cfg["trace_bundle"])
    res = ps.run_simulation(prior=prior, cfg=cfg, calib=calib, t_seis_s=t_seis, obs=obs, t_top_s=t0 / 1000,
                            t_base_s=t1 / 1000, td=(td_t, td_z), n=a.n)
    meta = {"name": name, "inline": il, "crossline": xl, "x": x, "y": y, "window_twt_ms": [t0, t1],
            "well": a.well, "well_is_held_out": bool(a.well in held_out) if a.well else None,
            "training_wells": prior.wells, "held_out_wells_excluded": held_out,
            "time_depth_from": td_well, "time_depth_distance_km": round(td_km, 2),
            "trace_selection": f"mean of {tie_cfg['trace_bundle']}x{tie_cfg['trace_bundle']} traces",
            "runtime_s": round(time.time() - t_start, 1), "config": cfg}
    ps.save_result(OUT / name, res, meta)
    m = json.loads((OUT / f"{name}.json").read_text(encoding="utf-8"))
    print(f"{name}: local ESS min {m['ess_local_min']} / median {m['ess_local_median']} of {res['n']} "
          f"(global {m['ess_global']}), P(gas column) prior {m['p_gas_column_prior']} -> posterior "
          f"{m['p_gas_column_posterior_global']}, {meta['runtime_s']} s")
    if m["ess_local_min"] < cfg["likelihood"]["min_ess_warning"]:
        print(f"WARNING: low local effective sample size ({m['ess_local_min']}); increase --n")
    if a.well:
        cmp = blind_comparison(OUT / f"{name}.npz", a.well, t0, t1, prior.facies)
        cmp.to_csv(OUT / f"{name}_blind.csv", index=False)
        print(cmp.to_string(index=False))


if __name__ == "__main__":
    main()
