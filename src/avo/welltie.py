"""Automated Poseidon well tie and seismic calibration (agent task 03).

For every well with seismic coverage:
1. time-depth: the well's own checkshot-calibrated curve, otherwise the nearest *eligible* well's curve
   (distance and Plover-depth compatibility checked, rejected donors recorded);
2. traces: n x n bundle around the nearest trace, near/mid/far (documented 6-18, 18-30, 30-42 deg);
3. automatic tie: bounded bulk shift + linear stretch maximising the mean near/mid/far correlation;
4. wavelets (Ricker freq + residual phase, documented reverse polarity) and one global scale and noise level
   per stack from TRAINING wells only (held-out wells excluded);
5. everything before/after is stored so the tie can be QC'd (outputs/well_ties/{well}.npz/.png/.json).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from avo import forward as fw
from avo.data import StackDef

FT = 0.3048


# ----------------------------------------------------------------------------------------- time-depth
def load_td_sources(wells_dir: Path, cfg: dict) -> dict[str, dict]:
    import lasio

    out = {}
    for name, src in cfg["td_sources"].items():
        path = wells_dir / src["file"]
        if not path.is_file():
            continue
        if src["kind"] == "vsp_adjusted_sonic_owt":
            las = lasio.read(str(path))
            z, owt = las.index.astype(float), las.data[:, src["column_index"]].astype(float)
            ok = np.isfinite(owt)
            z, twt = z[ok], 2 * owt[ok]
        elif src["kind"] == "vsp_calibrated_sonic":
            las = lasio.read(str(path))
            z, dt = las.index.astype(float), las.data[:, src["column_index"]].astype(float)
            ok = np.isfinite(dt)
            dt = np.interp(z, z[ok], dt[ok])
            dz = np.diff(z, prepend=z[0])
            twt = 2 * np.cumsum(dt * 1e-6 / FT * dz)
        else:
            table = pd.read_csv(path, sep="\t")
            z, twt = table.iloc[:, 0].to_numpy(float), 2 * table.iloc[:, 1].to_numpy(float)
        order = np.argsort(z)
        z, twt = z[order], np.maximum.accumulate(twt[order])
        out[name] = {"tvdss": z, "twt_s": twt, "meta": {**src, "file": str(path).replace("\\", "/"),
                                                         "depth_range_m": [float(z[0]), float(z[-1])]}}
    return out


def select_td(well: str, xy: dict, plover: dict, sources: dict, cfg: dict) -> dict:
    if well in sources:
        return {"donor": well, "transferred": False, "distance_km": 0.0, "rejected": []}
    t = cfg["transfer"]
    cands = []
    for name in sources:
        if name not in xy:
            continue
        d = float(np.hypot(xy[name][0] - xy[well][0], xy[name][1] - xy[well][1]) / 1000)
        cands.append((d, name))
    rejected, chosen = [], None
    for d, name in sorted(cands):
        reasons = []
        if d > t["max_distance_km"]:
            reasons.append(f"distance {d:.1f} km > {t['max_distance_km']} km")
        if well in plover and name in plover and abs(plover[well] - plover[name]) > t["max_plover_depth_diff_m"]:
            reasons.append(f"Plover top differs {abs(plover[well] - plover[name]):.0f} m")
        if chosen is None and not reasons:
            chosen = (d, name)
        elif chosen is not None:
            rejected.append({"donor": name, "distance_km": round(d, 1), "reason": "further than selected donor"})
        else:
            rejected.append({"donor": name, "distance_km": round(d, 1), "reason": "; ".join(reasons)})
    if chosen is None:
        return {"donor": None, "transferred": True, "distance_km": None, "rejected": rejected}
    checks = {"plover_top_target_m": plover.get(well), "plover_top_donor_m": plover.get(chosen[1])}
    return {"donor": chosen[1], "transferred": True, "distance_km": round(chosen[0], 1), "rejected": rejected,
            "compatibility": checks}


# ----------------------------------------------------------------------------------------- logs / seismic
def conditioned_logs(frame: pd.DataFrame, max_gap_m: float = 5.0) -> tuple[pd.DataFrame, dict]:
    """Longest continuous Vp/Vs/rho interval; flagged samples and short gaps are interpolated (documented)."""
    f = frame.sort_values("tvdss").reset_index(drop=True)
    step = float(np.median(np.diff(f["tvdss"])))
    limit = max(1, int(max_gap_m / step))
    cols = {}
    for c in ("vp", "vs", "rho"):
        s = f[c].where(~f["flag_bad"])
        cols[c] = s.interpolate(limit=limit, limit_area="inside")
    valid = cols["vp"].notna() & cols["vs"].notna() & cols["rho"].notna()
    runs = (valid != valid.shift()).cumsum()
    lengths = valid.groupby(runs).sum()
    best = lengths.idxmax()
    m = (runs == best) & valid
    out = pd.DataFrame({"tvdss": f.loc[m, "tvdss"], "md": f.loc[m, "md"], **{c: cols[c][m] for c in cols},
                        "gr": f.loc[m, "gr"], "vs_source": f.loc[m, "vs_source"]})
    info = {"interval_tvdss_m": [float(out["tvdss"].iloc[0]), float(out["tvdss"].iloc[-1])],
            "interpolated_samples": int((m & (f["flag_bad"] | f[["vp", "vs", "rho"]].isna().any(axis=1))).sum()),
            "vs_estimated_fraction": round(float((out["vs_source"] == "estimated").mean()), 2)}
    return out.reset_index(drop=True), info


def well_traces(cache_dir: Path, stacks: list[str], il: int, xl: int, bundle: int) -> tuple[np.ndarray, np.ndarray]:
    geometry = np.load(cache_dir / f"poseidon_{stacks[0]}.geometry.npz")
    i = int(np.searchsorted(geometry["ilines"], il))
    j = int(np.searchsorted(geometry["xlines"], xl))
    h = bundle // 2
    traces = []
    for s in stacks:
        g = np.load(cache_dir / f"poseidon_{s}.geometry.npz")
        cube = np.load(cache_dir / f"poseidon_{s}.f16.npy", mmap_mode="r")
        block = np.asarray(cube[max(i - h, 0):i + h + 1, max(j - h, 0):j + h + 1], dtype=np.float32)
        traces.append(block.reshape(-1, block.shape[-1]).mean(axis=0) * float(g["scale"]))
    return geometry["samples"] / 1000.0, np.array(traces)


# ----------------------------------------------------------------------------------------- synthetics
def _corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def synthetics(logs, twt_s, stack_defs, t_out, cfg, wavelets, polarity, shift_s=0.0, stretch=0.0, tc=None):
    tc = float(np.mean(twt_s)) if tc is None else tc
    twt2 = tc + (twt_s - tc) * (1 + stretch) + shift_s
    refl, r = fw.model_reflectivity(twt2, logs["vp"].to_numpy(), logs["vs"].to_numpy(), logs["rho"].to_numpy(),
                                    stack_defs, t_out, cfg["dt_model_s"])
    syn = polarity * np.array([fw.convolve_decimate(refl[k], wavelets[k], r) for k in range(len(stack_defs))])
    margin = cfg["edge_margin_s"]
    window = (t_out >= twt2[0] + margin) & (t_out <= twt2[-1] - margin)
    return syn, window


def auto_tie(logs, twt_s, stack_defs, t_out, obs, cfg, wavelets, polarity, max_shift_ms):
    a = cfg["auto_tie"]
    tc = float(np.mean(twt_s))
    best = None
    stretches = np.arange(-a["max_stretch_pct"], a["max_stretch_pct"] + 1e-9, a["stretch_step_pct"]) / 100
    for stretch in stretches:
        for s_ms in range(-max_shift_ms, max_shift_ms + 1, a["shift_step_ms"]):
            syn, win = synthetics(logs, twt_s, stack_defs, t_out, cfg, wavelets, polarity, s_ms / 1000, stretch, tc)
            if win.sum() < cfg["min_window_samples"]:
                continue
            c = np.mean([_corr(syn[k][win], obs[k][win]) for k in range(len(stack_defs))])
            score = c - a["shift_penalty"] * abs(s_ms) / max_shift_ms - a["shift_penalty"] * abs(stretch) / (
                a["max_stretch_pct"] / 100 + 1e-9)
            if best is None or score > best[0]:
                best = (score, s_ms / 1000, float(stretch), c)
    return None if best is None else {"shift_s": best[1], "stretch": best[2], "corr": best[3], "tc": tc}


def estimate_wavelets(ties: dict, stack_defs, t_out, cfg, polarity):
    freqs = np.arange(*cfg["wavelet_freqs_hz"])
    p = cfg["wavelet_phases_deg"]
    phases = np.arange(p[0], p[1] + 1e-9, p[2])
    out = {}
    for k, st in enumerate(stack_defs):
        best = None
        for f in freqs:
            for ph in phases:
                wl = fw.ricker(f, cfg["dt_model_s"], cfg["wavelet_length_s"], ph)
                scores = []
                for d in ties.values():
                    wls = [wl] * len(stack_defs)
                    syn, win = synthetics(d["logs"], d["twt_init"], stack_defs, t_out, cfg, wls, polarity,
                                          d["tie"]["shift_s"], d["tie"]["stretch"], d["tie"]["tc"])
                    scores.append(_corr(syn[k][win], d["obs"][k][win]))
                score = float(np.mean(scores))
                if best is None or score > best[0]:
                    best = (score, float(f), float(ph))
        out[st.name] = {"freq_hz": best[1], "phase_deg": best[2], "mean_corr_training": round(best[0], 3)}
    return out


# ----------------------------------------------------------------------------------------- run
def run(root: Path, cfg: dict, seismic_cfg: dict, petro_cfg: dict) -> pd.DataFrame:
    wells_dir = root / "data" / "poseidon" / "wells"
    cache_dir = root / "data" / "poseidon" / "seismic" / "cache"
    out_td = root / "data" / "interim" / "poseidon_td"
    out_ties = root / "outputs" / "well_ties"
    out_wl = root / "data" / "processed" / "wavelets"
    for d in (out_td, out_ties, out_wl):
        d.mkdir(parents=True, exist_ok=True)

    stack_defs = [StackDef(n, v["angle_min"], v["angle_max"]) for n, v in seismic_cfg["stacks"].items() if n != "full"]
    polarity = -1.0  # documented: trough = increase in impedance
    held_out = set(petro_cfg["held_out_wells"])

    wells = pd.read_csv(root / "data" / "poseidon" / "dashboard" / "wells.csv")
    xy = {r.WELL_ID: (r.X_M, r.Y_M) for r in wells.itertuples()}
    for name, src in cfg["td_sources"].items():
        if "x" in src:
            xy[name] = (src["x"], src["y"])
    tops = pd.read_csv(root / "data" / "poseidon" / "dashboard" / "well_tops.csv")
    plover = tops[tops["SURFACE"].str.contains("Plover", na=False)].groupby("WELL_ID")["TVDSS_M"].min().to_dict()
    sources = load_td_sources(wells_dir, cfg)

    import sys
    sys.path.insert(0, str(root))
    from src.dashboard_data import read_seismic_geometry, wells_on_grid

    geometry = read_seismic_geometry(root / "data" / "poseidon" / "seismic" / "poseidon_near.segy")
    grid = wells_on_grid(geometry, wells).set_index("WELL_ID")

    t_out = geometry["samples"] / 1000.0
    ref_wl = fw.ricker(cfg["reference_wavelet_hz"], cfg["dt_model_s"], cfg["wavelet_length_s"])
    ties, records, td_meta = {}, [], {}
    for well in wells["WELL_ID"]:
        qc = pd.read_parquet(root / "data" / "interim" / "poseidon_logs" / f"{well}.parquet")
        sel = select_td(well, xy, plover, sources, cfg)
        td_meta[well] = sel
        if sel["donor"] is None:
            records.append({"well": well, "status": "stopped: no eligible time-depth source"})
            continue
        src = sources[sel["donor"]]
        twt_init_all = np.interp(qc["tvdss"], src["tvdss"], src["twt_s"], left=np.nan, right=np.nan)
        pd.DataFrame({"md": qc["md"], "tvdss": qc["tvdss"], "twt_initial_ms": twt_init_all * 1000}).to_csv(
            out_td / f"{well}.csv", index=False)
        if not np.isfinite(grid.loc[well, "INLINE"]):
            records.append({"well": well, "status": "stopped: well outside the seismic sub-volume",
                            "td_source": sel["donor"]})
            continue
        logs, cond = conditioned_logs(qc)
        twt_init = np.interp(logs["tvdss"], src["tvdss"], src["twt_s"])
        _, obs = well_traces(cache_dir, [s.name for s in stack_defs], int(grid.loc[well, "INLINE"]),
                             int(grid.loc[well, "CROSSLINE"]), cfg["trace_bundle"])
        max_shift = cfg["auto_tie"]["max_shift_ms_transferred" if sel["transferred"] else "max_shift_ms_own_td"]
        ties[well] = {"logs": logs, "twt_init": twt_init, "obs": obs, "max_shift": max_shift, "cond": cond,
                      "qc": qc, "src": src, "il": int(grid.loc[well, "INLINE"]), "xl": int(grid.loc[well, "CROSSLINE"])}

    wavelets = [ref_wl] * len(stack_defs)
    wavelet_params = None
    for _ in range(cfg["auto_tie"]["passes"]):
        for well in list(ties):
            d = ties[well]
            d["tie"] = auto_tie(d["logs"], d["twt_init"], stack_defs, t_out, d["obs"], cfg, wavelets, polarity,
                                d["max_shift"])
            if d["tie"] is None:
                records.append({"well": well, "status": "stopped: log/seismic overlap too short for a tie",
                                "td_source": td_meta[well]["donor"]})
                del ties[well]
        training = {w: d for w, d in ties.items() if w not in held_out}
        wavelet_params = estimate_wavelets(training, stack_defs, t_out, cfg, polarity)
        wavelets = [fw.ricker(wavelet_params[s.name]["freq_hz"], cfg["dt_model_s"], cfg["wavelet_length_s"],
                              wavelet_params[s.name]["phase_deg"]) for s in stack_defs]

    # final synthetics, global scale and noise per stack from training wells only
    for d in ties.values():
        d["syn_before"], d["win_before"] = synthetics(d["logs"], d["twt_init"], stack_defs, t_out, cfg, wavelets,
                                                      polarity)
        d["syn_after"], d["win_after"] = synthetics(d["logs"], d["twt_init"], stack_defs, t_out, cfg, wavelets,
                                                    polarity, d["tie"]["shift_s"], d["tie"]["stretch"], d["tie"]["tc"])
    scales, noise = {}, {}
    for k, st in enumerate(stack_defs):
        obs = np.concatenate([d["obs"][k][d["win_after"]] for w, d in ties.items() if w not in held_out])
        syn = np.concatenate([d["syn_after"][k][d["win_after"]] for w, d in ties.items() if w not in held_out])
        scales[st.name] = float(np.sqrt(obs @ obs / (syn @ syn)))
        noise[st.name] = float((obs / scales[st.name] - syn).std())

    polarity_check = {}
    for sign in (-1.0, 1.0):
        polarity_check[f"{'reverse' if sign < 0 else 'normal'}"] = round(float(np.mean([
            _corr(sign * polarity * d["syn_after"][k][d["win_after"]], d["obs"][k][d["win_after"]])
            for w, d in ties.items() if w not in held_out for k in range(len(stack_defs))])), 3)

    for well, d in ties.items():
        sel, tie = td_meta[well], d["tie"]
        row = {"well": well, "status": "tied", "held_out": well in held_out, "td_source": sel["donor"],
               "td_transferred": sel["transferred"], "td_distance_km": sel["distance_km"],
               "inline": d["il"], "crossline": d["xl"], "shift_ms": round(tie["shift_s"] * 1000, 1),
               "stretch_pct": round(tie["stretch"] * 100, 2), "max_shift_ms": d["max_shift"],
               "window_ms": [round(float(t_out[d["win_after"]][0] * 1000)), round(float(t_out[d["win_after"]][-1] * 1000))],
               "vs_estimated_fraction": d["cond"]["vs_estimated_fraction"]}
        for k, st in enumerate(stack_defs):
            row[f"corr_before_{st.name}"] = round(_corr(d["syn_before"][k][d["win_before"]], d["obs"][k][d["win_before"]]), 3)
            row[f"corr_after_{st.name}"] = round(_corr(d["syn_after"][k][d["win_after"]], d["obs"][k][d["win_after"]]), 3)
        row["corr_before_mean"] = round(np.mean([row[f"corr_before_{s.name}"] for s in stack_defs]), 3)
        row["corr_after_mean"] = round(np.mean([row[f"corr_after_{s.name}"] for s in stack_defs]), 3)
        window_ms = float((t_out[d["win_after"]][-1] - t_out[d["win_after"]][0]) * 1000)
        flags = []
        if abs(row["shift_ms"]) >= 0.8 * d["max_shift"]:
            flags.append("shift near limit (possible cycle skip)")
        if window_ms < 200:
            flags.append(f"short tie window ({window_ms:.0f} ms)")
        if row["corr_after_mean"] < cfg["auto_tie"]["g2_min_corr"]:
            flags.append("low correlation")
        if min(row[f"corr_after_{s.name}"] for s in stack_defs) < 0.3:
            flags.append("one stack ties poorly")
        row["qc_flags"] = "; ".join(flags) if flags else "ok"
        row["needs_review"] = bool(flags)
        records.append(row)

        # final time-depth for the whole well (shift/stretch valid within the tie window)
        qc = d["qc"]
        twt0 = np.interp(qc["tvdss"], d["src"]["tvdss"], d["src"]["twt_s"], left=np.nan, right=np.nan)
        twt1 = tie["tc"] + (twt0 - tie["tc"]) * (1 + tie["stretch"]) + tie["shift_s"]
        pd.DataFrame({"md": qc["md"], "tvdss": qc["tvdss"], "twt_initial_ms": twt0 * 1000,
                      "twt_final_ms": twt1 * 1000}).to_csv(out_td / f"{well}.csv", index=False)
        qc["twt"] = twt1 * 1000
        qc.to_parquet(root / "data" / "interim" / "poseidon_logs" / f"{well}.parquet", index=False)
        logs = d["logs"]
        twt_logs = tie["tc"] + (d["twt_init"] - tie["tc"]) * (1 + tie["stretch"]) + tie["shift_s"]
        well_tops = tops[tops["WELL_ID"] == well]
        tops_twt = tie["tc"] + (np.interp(well_tops["TVDSS_M"], d["src"]["tvdss"], d["src"]["twt_s"]) - tie["tc"]) * (
            1 + tie["stretch"]) + tie["shift_s"]
        np.savez(out_ties / f"{well}.npz", t_ms=t_out * 1000, obs=d["obs"], syn_before=d["syn_before"],
                 syn_after=d["syn_after"], win_before=d["win_before"], win_after=d["win_after"],
                 log_twt_ms=twt_logs * 1000, log_twt_initial_ms=d["twt_init"] * 1000,
                 ai=(logs["vp"] * logs["rho"]).to_numpy(), vpvs=(logs["vp"] / logs["vs"]).to_numpy(),
                 gr=logs["gr"].to_numpy(), tops_names=well_tops["SURFACE"].to_numpy(dtype=str),
                 tops_twt_ms=np.asarray(tops_twt) * 1000, scales=np.array([scales[s.name] for s in stack_defs]),
                 stacks=np.array([s.name for s in stack_defs]))
        meta = {"well": well, "held_out": well in held_out, "time_depth": {**sel, "source_meta": d["src"]["meta"]},
                "trace_selection": f"mean of {cfg['trace_bundle']}x{cfg['trace_bundle']} traces around il {d['il']} / xl {d['xl']}",
                "log_conditioning": d["cond"], "auto_tie": {**tie, "max_shift_ms": d["max_shift"]},
                "polarity": "reverse (documented)", "wavelets": wavelet_params, "result": row}
        (out_ties / f"{well}.json").write_text(json.dumps(meta, indent=2, default=float), encoding="utf-8")

    summary = pd.DataFrame(records)
    summary.to_csv(out_ties / "summary.csv", index=False)
    for st, w in zip(stack_defs, wavelets):
        np.save(out_wl / f"{st.name}.npy", w)
        (out_wl / f"{st.name}.json").write_text(json.dumps({
            **wavelet_params[st.name], "type": "Ricker", "dt_s": cfg["dt_model_s"], "length_s": cfg["wavelet_length_s"],
            "polarity": "reverse (trough = increase in impedance)", "angles_deg": [st.angle_min, st.angle_max],
            "training_wells": sorted(w_ for w_ in ties if w_ not in held_out)}, indent=2), encoding="utf-8")
    calibration = {
        "generated_by": "scripts/run_well_tie.py", "polarity": "reverse (documented in processing report)",
        "polarity_check_mean_corr": polarity_check,
        "training_wells": sorted(w for w in ties if w not in held_out), "held_out_wells": sorted(held_out),
        "stacks": {st.name: {"angle_min": st.angle_min, "angle_max": st.angle_max, **wavelet_params[st.name],
                             "scale": scales[st.name], "noise_std_synthetic_units": noise[st.name]}
                   for st in stack_defs},
        "unavailable": {"common_amplitude_scale": "time variant scaling parameters undocumented",
                        "deviation_surveys": "TVDSS is a vertical-well assumption"},
    }
    return summary, calibration, ties, stack_defs, td_meta
