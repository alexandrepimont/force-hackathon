"""Poseidon well QC and petrophysics (agent task 02).

Reads the primary log run of each well (Occam spliced composite LAS), keeps the raw curves and their
provenance, adds normalised measurements, QC flags and documented petrophysical scenarios, and writes
data/interim/poseidon_logs/{well}.parquet, outputs/well_qc/{well}.png and outputs/well_qc/summary.csv.

Rules followed (see agents/02_well_qc_petrophysics.md):
- raw values are never replaced; normalised columns are added next to raw_<MNEMONIC> columns;
- TVDSS is a documented vertical-well assumption (no deviation surveys available) with an error estimate;
- bad-hole samples are flagged with the rule that triggered them, never deleted;
- lithotype is a petrophysical scenario with cut-offs from configs/petrophysics.yaml, unknown stays unknown;
- fluid is not labelled from logs: only a hydrocarbon *indicator* with its evidence is stored.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from avo import rock_physics as rp

US_FT_TO_M_S = 304_800.0

# normalised column -> (accepted source mnemonics in priority order, expected unit family, conversion)
CURVE_MAP = {
    "gr": (("ECGR", "GR", "GRD", "GRARC"), "gAPI", "none"),
    "rt": (("RD", "ATRT", "P34H"), "ohm.m", "none"),
    "rho": (("RHOB", "RHOZ", "HROM"), "g/cm3", "none"),
    "nphi": (("TNPH", "NPHI", "TNP", "HTNP"), "pu", "percent -> v/v (/100)"),
    "dtc": (("DTCO", "DTC", "BATC"), "us/ft", "none"),
    "dts": (("DTSM", "DTS"), "us/ft", "none"),
    "cali": (("HDAR", "CAL1", "DCAV"), "in", "none"),
}

LITHOTYPES = ["shale", "tight", "sand_low_phi", "sand_high_phi", "volcanic", "unknown"]
LITHOTYPE_COLORS = {"shale": "#8c8c8c", "tight": "#9b6fc3", "sand_low_phi": "#f2c14e",
                    "sand_high_phi": "#e8833a", "volcanic": "#3d3d3d", "unknown": "#ffffff"}


@dataclass
class WellHeader:
    well: str
    x: float
    y: float
    datum_elev: float
    td_md: float
    td_tvdss: float


def read_headers(path: Path) -> dict[str, WellHeader]:
    names = {"Boreas_1": "Boreas-1", "Kronons_1": "Kronos-1", "Pharos_1": "Pharos-1", "Poseidon_1": "Poseidon-1",
             "Poseidon_2": "Poseidon-2", "Poseidon_North": "Poseidon-North-1", "Proteus_1": "Proteus-1",
             "Torosa_1": "Torosa-1"}
    h = pd.read_csv(path)
    out = {}
    for _, r in h.iterrows():
        well = names.get(r["Name"])
        if well:
            out[well] = WellHeader(well, r["X"], r["Y"], r["DFE"], r["TD_(MD)"], r["TD_(TVDSS)"])
    return out


def read_primary_run(path: Path, well: str) -> tuple[pd.DataFrame, dict]:
    """Read one LAS run; return raw curves (prefixed raw_) and a provenance record per normalised curve."""
    import lasio

    las = lasio.read(str(path))
    units = {c.mnemonic.upper(): (c.unit or "").strip() for c in las.curves}
    raw = las.df()
    raw.columns = [str(c).upper() for c in raw.columns]
    md = raw.index.to_numpy(dtype=float)
    frame = pd.DataFrame({"well": well, "md": md})
    for col in raw.columns:
        frame[f"raw_{col}"] = raw[col].to_numpy(dtype=float)
    provenance = {"file": str(path).replace("\\", "/"), "null": float(las.well.NULL.value),
                  "depth": {"mnemonic": las.curves[0].mnemonic, "unit": las.curves[0].unit,
                            "description": las.curves[0].descr}, "curves": {}}
    for name, (mnemonics, unit_family, conversion) in CURVE_MAP.items():
        source = next((m for m in mnemonics if m in raw.columns), None)
        if source is None:
            provenance["curves"][name] = {"source": None, "status": "unavailable"}
            frame[name] = np.nan
            continue
        values = raw[source].to_numpy(dtype=float)
        unit = units.get(source, "")
        if name == "nphi":
            if unit.lower() in {"pu", "%"}:
                values = values / 100.0
            elif unit.lower() not in {"v/v", "dec", "frac"}:
                conversion = f"unit '{unit}' unknown; assumed percent"
                values = values / 100.0
        frame[name] = values
        provenance["curves"][name] = {"source": source, "unit": unit, "expected": unit_family,
                                      "conversion": conversion, "status": "measured"}
    return frame, provenance


def add_depths(frame: pd.DataFrame, header: WellHeader) -> dict:
    frame["tvdss"] = frame["md"] - header.datum_elev
    error = float(abs((header.td_md - header.datum_elev) - header.td_tvdss))
    frame["tvdss_method"] = "vertical_assumption"
    return {"method": "MD - DFE (vertical-well assumption; no deviation survey available)",
            "datum_elev_m": header.datum_elev, "max_error_at_td_m": round(error, 1)}


def add_flags(frame: pd.DataFrame, cfg: dict, zone: np.ndarray) -> None:
    b = cfg["bad_hole"]
    reasons = pd.Series("", index=frame.index, dtype="object")

    def flag(mask, text):
        nonlocal reasons
        mask = pd.Series(mask, index=frame.index).fillna(False).to_numpy(bool)
        reasons[mask] = reasons[mask] + (text + ";")

    cali = frame["cali"]
    if cali.notna().any():
        step = float(np.nanmedian(np.diff(frame["md"]))) or 0.5
        window = max(3, int(b["caliper_window_m"] / step))
        running = cali.rolling(window, center=True, min_periods=window // 4).median()
        bits = np.asarray(b["bit_sizes_in"], dtype=float)
        nominal = running.map(lambda v: bits[np.argmin(np.abs(bits - v))] if np.isfinite(v) else np.nan)
        frame["bit_size_inferred"] = nominal
        flag(cali > nominal + b["caliper_excess_in"], "washout(caliper>bit+%.1fin)" % b["caliper_excess_in"])
    flag(~frame["rho"].between(*b["rho_valid"]) & frame["rho"].notna(), "rho_out_of_range")
    vp = US_FT_TO_M_S / frame["dtc"]
    flag(~vp.between(*b["vp_valid"]) & vp.notna(), "vp_out_of_range")
    vpvs = frame["dts"] / frame["dtc"]
    flag(~vpvs.between(*b["vpvs_valid"]) & vpvs.notna(), "vpvs_out_of_range")
    frame["flag_bad"] = reasons != ""
    frame["flag_reason"] = reasons.str.rstrip(";")


def add_petrophysics(frame: pd.DataFrame, cfg: dict, zone: np.ndarray) -> dict:
    params = {}
    gr = frame["gr"]
    if gr[zone].notna().sum() > 50:
        clean = float(np.nanpercentile(gr[zone], cfg["vsh"]["gr_clean_percentile"]))
        shale = float(np.nanpercentile(gr[zone], cfg["vsh"]["gr_shale_percentile"]))
        frame["vsh"] = ((gr - clean) / max(shale - clean, 1e-6)).clip(0, 1)
        params["gr_clean"], params["gr_shale"] = round(clean, 1), round(shale, 1)
    else:
        frame["vsh"] = np.nan
    p = cfg["porosity"]
    phid = ((p["rho_matrix"] - frame["rho"]) / (p["rho_matrix"] - p["rho_fluid"])).clip(0, p["max"])
    phin = frame["nphi"].clip(0, p["max"])
    gas_like = phin < phid
    phi_nd = np.where(gas_like, np.sqrt((phid**2 + phin**2) / 2), (phid + phin) / 2)
    frame["phid"] = phid
    frame["phi"] = np.where(phin.notna(), phi_nd, phid)
    frame["phi_method"] = np.where(phin.notna() & phid.notna(), "neutron-density",
                                   np.where(phid.notna(), "density-only", "unavailable"))
    a = cfg["archie"]
    phi_safe = frame["phi"].where(frame["phi"] > 0.01)
    frame["sw"] = ((a["a"] * a["rw"]) / (phi_safe ** a["m"] * frame["rt"])) ** (1 / a["n"])
    frame["sw"] = frame["sw"].clip(0, 1)
    frame["vp"] = US_FT_TO_M_S / frame["dtc"]
    frame["vs_measured"] = US_FT_TO_M_S / frame["dts"]
    return params


def calibrate_vs(frames: dict[str, pd.DataFrame], held_out: list[str], zone_min: float) -> dict:
    """Bias-correct Greenberg-Castagna on measured Vs in training wells only."""
    ratios = []
    for well, f in frames.items():
        if well in held_out:
            continue
        m = (f["tvdss"] >= zone_min) & f["vs_measured"].notna() & f["vp"].notna() & f["vsh"].notna() & ~f["flag_bad"]
        if m.sum() > 20:
            gc = rp.vs_greenberg_castagna(f.loc[m, "vp"].to_numpy(), f.loc[m, "vsh"].to_numpy())
            ratios.append(f.loc[m, "vs_measured"].to_numpy() / gc)
    r = np.concatenate(ratios)
    return {"method": "Greenberg-Castagna (1992) x bias factor", "bias_factor": float(np.median(r)),
            "relative_std": float(np.std(r) / np.median(r)), "n_samples": int(r.size),
            "training_wells": [w for w in frames if w not in held_out]}


def add_vs(frame: pd.DataFrame, vs_cal: dict) -> None:
    est = rp.vs_greenberg_castagna(frame["vp"].to_numpy(), frame["vsh"].fillna(0.5).to_numpy()) * vs_cal["bias_factor"]
    frame["vs"] = frame["vs_measured"]
    use_est = frame["vs"].isna() & frame["vp"].notna()
    frame.loc[use_est, "vs"] = est[use_est.to_numpy()]
    frame["vs_source"] = np.where(frame["vs_measured"].notna(), "measured",
                                  np.where(use_est, "estimated", "unavailable"))
    frame["vs_uncertainty_rel"] = np.where(use_est, vs_cal["relative_std"], 0.0)


def add_labels(frame: pd.DataFrame, tops: pd.DataFrame, cfg: dict) -> None:
    lab = cfg["labels"]
    frame["formation"] = "no tops available"
    if len(tops):
        ordered = tops.sort_values("MD_M")
        idx = np.searchsorted(ordered["MD_M"].to_numpy(), frame["md"].to_numpy(), "right") - 1
        names = ordered["SURFACE"].to_numpy()
        frame["formation"] = np.where(idx >= 0, names[np.clip(idx, 0, None)], "above first top")
    vsh, phi = frame["vsh"], frame["phi"]
    known = vsh.notna() & pd.Series(phi).notna()
    separation = frame["nphi"] - frame["phid"]
    clay_nd = (separation >= lab["nd_separation_shale"]).fillna(False)
    lith = np.full(len(frame), "unknown", dtype=object)
    shale = known & ((vsh >= lab["vsh_shale"]) | clay_nd)
    lith[shale] = "shale"
    sandy = known & ~shale
    lith[sandy & (phi < lab["phi_tight"])] = "tight"
    lith[sandy & (phi >= lab["phi_tight"]) & (phi < lab["phi_good"])] = "sand_low_phi"
    lith[sandy & (phi >= lab["phi_good"])] = "sand_high_phi"
    top_v, top_r = lab["volcanic_interval"]
    vt = tops.loc[tops["SURFACE"] == top_v, "MD_M"]
    vr = tops.loc[tops["SURFACE"] == top_r, "MD_M"]
    if len(vt) and len(vr):
        lith[(frame["md"] >= vt.iloc[0]) & (frame["md"] < vr.iloc[0]) & known] = "volcanic"
    frame["lithotype"] = lith
    frame["litho"] = pd.Series(lith).replace({"sand_low_phi": "sand", "sand_high_phi": "sand"}).to_numpy()
    frame["label_method"] = ("petrophysical scenario: shale if Vsh(GR) or neutron-density separation above cut-off, "
                             "then porosity classes (configs/petrophysics.yaml)")
    frame["label_confidence"] = np.where(lith == "unknown", "none",
                                         np.where(frame["flag_bad"], "low", "medium"))
    h = cfg["hc_indicator"]
    sand = np.isin(lith, ["sand_low_phi", "sand_high_phi"])
    low_sw = frame["sw"] < h["sw_max"]
    crossover = (frame["phid"] - frame["nphi"].clip(0, None)) > h["crossover_min"]
    frame["hc_indicator"] = np.where(sand & low_sw & crossover, "gas_indicated",
                                     np.where(sand & (low_sw | crossover), "possible",
                                              np.where(sand, "none", "not_applicable")))
    frame["fluid"] = "unknown"


def reservoir_interval(tops: pd.DataFrame, cfg: dict, td_md: float):
    r = cfg["reservoir"]
    top = next((tops.loc[tops["SURFACE"] == s, "MD_M"].min() for s in r["top_surfaces"]
                if (tops["SURFACE"] == s).any()), np.nan)
    base = next((tops.loc[tops["SURFACE"] == s, "MD_M"].min() for s in r["base_surfaces"]
                 if (tops["SURFACE"] == s).any()), td_md)
    return top, base


def summarise(frame: pd.DataFrame, well: str, tops: pd.DataFrame, cfg: dict, depth_info: dict,
              params: dict, td_md: float) -> dict:
    zone = frame["tvdss"] >= cfg["zone_of_interest_tvdss"]
    row = {"well": well, "tvdss_max_error_m": depth_info["max_error_at_td_m"], "n_tops": len(tops)}
    for c in CURVE_MAP:
        row[f"coverage_{c}"] = round(float(frame.loc[zone, c].notna().mean()), 2)
    row["flagged_fraction_zone"] = round(float(frame.loc[zone, "flag_bad"].mean()), 3)
    row["gr_clean"], row["gr_shale"] = params.get("gr_clean"), params.get("gr_shale")
    top, base = reservoir_interval(tops, cfg, td_md)
    row["reservoir_top_md"], row["reservoir_base_md"] = top, base
    if np.isfinite(top):
        res = (frame["md"] >= top) & (frame["md"] < base)
        step = float(np.median(np.diff(frame["md"])))
        sand = res & frame["lithotype"].isin(["sand_low_phi", "sand_high_phi"])
        row["gross_m"] = round(float(res.sum() * step), 1)
        row["net_sand_m"] = round(float(sand.sum() * step), 1)
        row["net_to_gross"] = round(row["net_sand_m"] / row["gross_m"], 2) if row["gross_m"] else np.nan
        row["mean_phi_net"] = round(float(frame.loc[sand, "phi"].mean()), 3) if sand.any() else np.nan
        row["mean_sw_net"] = round(float(frame.loc[sand, "sw"].mean()), 2) if sand.any() else np.nan
        row["gas_indicated_m"] = round(float((sand & (frame["hc_indicator"] == "gas_indicated")).sum() * step), 1)
        row["vs_measured_fraction_res"] = round(float((frame.loc[res, "vs_source"] == "measured").mean()), 2)
        row["flagged_fraction_res"] = round(float(frame.loc[res, "flag_bad"].mean()), 3)
    else:
        row["note"] = "no reservoir top available: reservoir statistics not computed"
    return row


def qc_plot(frame: pd.DataFrame, tops: pd.DataFrame, well: str, cfg: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    z0 = cfg["zone_of_interest_tvdss"]
    d = frame[frame["tvdss"] >= z0]
    if d.empty:
        return
    fig, axes = plt.subplots(1, 8, figsize=(17, 10), sharey=True,
                             gridspec_kw={"width_ratios": [1.2, 1.2, 1.2, 1, 1, 1, 0.45, 0.35]})
    z = d["tvdss"]
    ax = axes[0]
    ax.plot(d["gr"], z, color="#2e7d32", lw=0.7)
    ax.set_xlim(0, 200)
    ax.set_title("GR (API)")
    ax2 = ax.twiny()
    ax2.plot(d["vsh"], z, color="#795548", lw=0.7)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Vsh", color="#795548")
    ax = axes[1]
    ax.plot(d["rho"], z, color="#c62828", lw=0.7)
    ax.set_xlim(1.95, 2.95)
    ax.set_title("RHOB (g/cc) / NPHI")
    ax2 = ax.twiny()
    ax2.plot(d["nphi"], z, color="#1565c0", lw=0.7)
    ax2.set_xlim(0.45, -0.15)
    ax = axes[2]
    ax.plot(d["vp"], z, color="#1565c0", lw=0.7, label="Vp")
    meas = d["vs_source"] == "measured"
    ax.plot(d["vs"].where(meas), z, color="#6a1b9a", lw=0.7, label="Vs measured")
    ax.plot(d["vs"].where(d["vs_source"] == "estimated"), z, color="#6a1b9a", lw=0.7, ls=":", label="Vs estimated")
    ax.set_xlim(1000, 6500)
    ax.set_title("Vp, Vs (m/s)")
    ax.legend(fontsize=7, loc="lower left")
    axes[3].plot(d["phi"], z, color="#ef6c00", lw=0.7)
    axes[3].set_xlim(0.35, 0)
    axes[3].set_title("Porosity (v/v)")
    axes[4].plot(d["rt"], z, color="#333", lw=0.7)
    axes[4].set_xscale("log")
    axes[4].set_xlim(0.2, 200)
    axes[4].set_title("RT (ohm.m)")
    axes[5].plot(d["sw"], z, color="#00838f", lw=0.7)
    axes[5].set_xlim(1, 0)
    axes[5].set_title("Sw (Archie)")
    codes = d["lithotype"].map({k: i for i, k in enumerate(LITHOTYPES)}).to_numpy()
    axes[6].imshow(codes[:, None], aspect="auto", cmap=ListedColormap([LITHOTYPE_COLORS[k] for k in LITHOTYPES]),
                   vmin=-0.5, vmax=len(LITHOTYPES) - 0.5, extent=[0, 1, z.iloc[-1], z.iloc[0]], interpolation="nearest")
    gas = d["hc_indicator"] == "gas_indicated"
    axes[6].scatter(np.full(gas.sum(), 0.5), z[gas], s=1, color="#e0483e")
    axes[6].set_title("Litho-\ntype", fontsize=9)
    axes[6].set_xticks([])
    axes[7].imshow(d["flag_bad"].to_numpy()[:, None].astype(float), aspect="auto", cmap=ListedColormap(["white", "#e0483e"]),
                   vmin=0, vmax=1, extent=[0, 1, z.iloc[-1], z.iloc[0]], interpolation="nearest")
    axes[7].set_title("Bad\nhole", fontsize=9)
    axes[7].set_xticks([])
    in_view = tops[tops["MD_M"] - frame["md"].iloc[0] + frame["tvdss"].iloc[0] >= z0]
    for _, t in in_view.iterrows():
        tz = t["MD_M"] - (frame["md"] - frame["tvdss"]).iloc[0]
        for a in axes:
            a.axhline(tz, color="#555", lw=0.6, ls="--")
        axes[-1].text(1.1, tz, t["SURFACE"], fontsize=7, va="center", transform=axes[-1].get_yaxis_transform())
    axes[0].set_ylim(z.iloc[-1], z.iloc[0])
    axes[0].set_ylabel("TVDSS (m, vertical-well assumption)")
    for a in axes[:6]:
        a.grid(alpha=0.3)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=LITHOTYPE_COLORS[k], ec="#999") for k in LITHOTYPES]
    fig.legend(handles, LITHOTYPES + [], loc="lower center", ncol=len(LITHOTYPES), fontsize=8)
    fig.suptitle(f"{well}: well QC (red dots in lithotype = gas indicated; labels are a petrophysical scenario)",
                 weight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def run(project_root: Path, cfg: dict) -> pd.DataFrame:
    wells_dir = project_root / "data" / "poseidon" / "wells"
    out_logs = project_root / "data" / "interim" / "poseidon_logs"
    out_qc = project_root / "outputs" / "well_qc"
    out_logs.mkdir(parents=True, exist_ok=True)
    out_qc.mkdir(parents=True, exist_ok=True)
    headers = read_headers(wells_dir / "tops" / "t20" / "Well_Headers.csv")
    tops_all = pd.read_csv(wells_dir / "tops" / "well_tops.csv")
    held_out = cfg["held_out_wells"]

    frames, meta = {}, {}
    for las_path in sorted(wells_dir.glob("logs/*/*.LAS")):
        well = las_path.parent.name
        frame, prov = read_primary_run(las_path, well)
        depth_info = add_depths(frame, headers[well])
        zone = (frame["tvdss"] >= cfg["zone_of_interest_tvdss"]).to_numpy()
        add_flags(frame, cfg, zone)
        params = add_petrophysics(frame, cfg, zone)
        frames[well] = frame
        meta[well] = {"provenance": prov, "depth": depth_info, "petrophysics_params": params}

    vs_cal = calibrate_vs(frames, held_out, cfg["zone_of_interest_tvdss"])
    rows = []
    for well, frame in frames.items():
        add_vs(frame, vs_cal)
        tops = tops_all[tops_all["WELL"] == well]
        add_labels(frame, tops, cfg)
        frame["twt"] = np.nan  # filled by the well tie (task 03)
        frame["held_out"] = well in held_out
        frame.to_parquet(out_logs / f"{well}.parquet", index=False)
        meta[well]["vs_estimate"] = vs_cal
        meta[well]["held_out"] = well in held_out
        (out_logs / f"{well}.meta.json").write_text(json.dumps(meta[well], indent=2, default=str), encoding="utf-8")
        qc_plot(frame, tops, well, cfg, out_qc / f"{well}.png")
        rows.append(summarise(frame, well, tops, cfg, meta[well]["depth"], meta[well]["petrophysics_params"],
                              headers[well].td_md))
    summary = pd.DataFrame(rows)
    summary["held_out"] = summary["well"].isin(held_out)
    summary.to_csv(out_qc / "summary.csv", index=False)
    (out_qc / "vs_calibration.json").write_text(json.dumps(vs_cal, indent=2), encoding="utf-8")
    return summary
