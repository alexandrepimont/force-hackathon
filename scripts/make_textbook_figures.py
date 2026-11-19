"""Generate the figures for docs/textbook/AVO_workflow_textbook.md.

Uses the real Poseidon data where available (data/poseidon/dashboard, data/poseidon/wells,
data/poseidon/seismic) and the pipeline's own physics (avo.rock_physics, avo.forward, avo.priors)
for the concept figures. Figures that need missing data are skipped with a message.

    python scripts/make_textbook_figures.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from avo import forward as fw  # noqa: E402
from avo import priors  # noqa: E402
from avo import rock_physics as rp  # noqa: E402
from avo.data import StackDef  # noqa: E402

OUT = ROOT / "docs" / "textbook" / "figures"
DASH = ROOT / "data" / "poseidon" / "dashboard"
WELLS = ROOT / "data" / "poseidon" / "wells"
SEIS = ROOT / "data" / "poseidon" / "seismic"
CFG = yaml.safe_load((ROOT / "configs" / "pipeline.yaml").read_text())
MINERALS = CFG["rock_physics"]["minerals"]
PHIC, EXPONENT = CFG["rock_physics"]["frame"]["phic"], CFG["rock_physics"]["frame"]["exponent"]

# Illustrative angle ranges -- confirm against the Poseidon processing report.
STACKS = [StackDef("near", 5, 15), StackDef("mid", 15, 25), StackDef("far", 25, 35)]
STACK_COLORS = {"near": "#4c9be8", "mid": "#43b581", "far": "#e8833a"}

# Illustrative Plover (~4.8 km TVDSS) conditions; gas values are approximate, not lab-measured.
K_BRINE, RHO_BRINE = rp.brine_properties(55.0, 150.0, 60000)
GAS = {"k_brine": K_BRINE, "rho_brine": RHO_BRINE, "k_oil": 0.15, "rho_oil": 0.35}
OIL = rp.fluid_set(CFG["rock_physics"]["reservoir"])

CLASS_NAMES = ["shale", "tight", "brine sand", "gas sand"]
CLASS_COLORS = ["#8c8c8c", "#9b6fc3", "#3b8bd9", "#e0483e"]

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "font.size": 10, "axes.titlesize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


def rock(phi, vsh, sw, calcite=0.0, fluids=GAS):
    return rp.elastic_from_rock(np.atleast_1d(phi), np.atleast_1d(vsh), np.atleast_1d(calcite),
                                np.atleast_1d(sw), fluids, MINERALS, PHIC, EXPONENT)


# ---------------------------------------------------------------- 1. workflow
def fig_workflow():
    steps = [
        ("1  Data", "wells, tops,\nnear/mid/far seismic", "#dbe9f7"),
        ("2  QC & petrophysics", "clean logs, Vsh, porosity,\nlithology & fluid labels", "#dbe9f7"),
        ("3  Well tie", "depth -> time,\nwavelet & amplitude scale", "#dbe9f7"),
        ("4  Rock physics", "Gassmann fluid substitution,\nis fluid visible? (gate G3)", "#dcf0e2"),
        ("5  Scenarios", "Monte Carlo pseudo-wells:\nporosity, thickness, fluid", "#dcf0e2"),
        ("6  Forward modelling", "synthetic near/mid/far\nfor every scenario", "#dcf0e2"),
        ("7  Train classifiers", "A: intercept-gradient\nB, C: scenario-trained", "#fbe6d3"),
        ("8  Blind-well test", "score on wells never\nused in training (gate G4)", "#fbe6d3"),
        ("9  Apply to 3D", "classify every trace in\nthe reservoir window", "#e9def5"),
        ("10  Uncertainty", "probabilities, entropy,\nwhat-if, dashboard", "#e9def5"),
    ]
    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 2)
    ax.axis("off")
    centers = []
    for i, (title, body, color) in enumerate(steps):
        row = i // 5
        col = i % 5 if row == 0 else 4 - i % 5
        x, y = col + 0.5, 1.5 - row
        centers.append((x, y))
        ax.add_patch(FancyBboxPatch((x - 0.44, y - 0.33), 0.88, 0.66, boxstyle="round,pad=0.02",
                                    fc=color, ec="#5a6470", lw=1))
        ax.text(x, y + 0.17, title, ha="center", va="center", weight="bold", fontsize=10)
        ax.text(x, y - 0.08, body, ha="center", va="center", fontsize=8.5)
    for (x0, y0), (x1, y1) in zip(centers[:-1], centers[1:]):
        if y0 == y1:
            start, end = (x0 + 0.45 * np.sign(x1 - x0), y0), (x1 - 0.45 * np.sign(x1 - x0), y1)
        else:
            start, end = (x0, y0 - 0.34), (x1, y1 + 0.34)
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, color="#5a6470"))
    legend = [("Data preparation", "#dbe9f7"), ("Physics & simulation", "#dcf0e2"),
              ("Learning & testing", "#fbe6d3"), ("Results", "#e9def5")]
    for k, (label, color) in enumerate(legend):
        ax.add_patch(Rectangle((0.3 + 1.2 * k, 0.05), 0.12, 0.08, fc=color, ec="#5a6470"))
        ax.text(0.47 + 1.2 * k, 0.09, label, va="center", fontsize=9)
    ax.set_title("The workflow at a glance: from well logs and seismic to 3D lithology/fluid probabilities",
                 fontsize=12, weight="bold")
    save(fig, "fig01_workflow.png")


# ---------------------------------------------------------------- 2. field map
def seismic_outline():
    path = SEIS / "poseidon_near.segy"
    if not path.is_file():
        return None
    import segyio
    with segyio.open(str(path), iline=189, xline=193, ignore_geometry=False) as f:
        n_xl = len(f.xlines)
        n = f.tracecount
        corners = [0, n_xl - 1, n - 1, n - n_xl]
        pts = []
        for i in corners:
            h = f.header[i]
            s = h[segyio.TraceField.SourceGroupScalar]
            factor = 1 / abs(s) if s < 0 else (s or 1)
            pts.append((h[segyio.TraceField.CDP_X] * factor, h[segyio.TraceField.CDP_Y] * factor))
    return np.array(pts)


def plover_top_by_well():
    tops = pd.read_csv(DASH / "well_tops.csv")
    plover = tops[tops["SURFACE"].str.contains("Plover", na=False)]
    reservoir = plover[plover["SURFACE"].str.contains("Reservoir", na=False)].groupby("WELL_ID")["TVDSS_M"].min()
    first = plover.groupby("WELL_ID")["TVDSS_M"].min()
    return reservoir.combine_first(first)


def fig_field_map():
    wells = pd.read_csv(DASH / "wells.csv")
    plover = plover_top_by_well()
    outline = seismic_outline()
    fig, ax = plt.subplots(figsize=(8.5, 7))
    if outline is not None:
        ax.add_patch(Polygon(outline, closed=True, fc="#f3e9c8", ec="#b58b16", lw=1.5,
                             label="Downloaded seismic sub-volume"))
    offsets = {"Boreas-1": (8, 8), "Poseidon-2": (-10, -24), "Proteus-1": (8, -14)}
    for _, w in wells.iterrows():
        depth = plover.get(w["WELL_ID"])
        ax.plot(w["X_M"], w["Y_M"], "o", ms=9, color="#1f4e79", mec="white", zorder=3)
        label = w["WELL_ID"] + (f"\nPlover top {depth:,.0f} m" if pd.notna(depth) else "\nno Plover pick")
        ax.annotate(label, (w["X_M"], w["Y_M"]), xytext=offsets.get(w["WELL_ID"], (6, 6)),
                    textcoords="offset points", fontsize=8.5,
                    bbox=dict(fc="white", ec="none", alpha=0.7, pad=1))
    ax.set_aspect("equal")
    ax.set_xlabel("Easting (m), GDA94 / MGA zone 51")
    ax.set_ylabel("Northing (m)")
    ax.ticklabel_format(style="plain")
    ax.set_title("Poseidon wells and the seismic area we work in\n(depths are TVDSS, metres below sea level)")
    if outline is not None:
        ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    save(fig, "fig02_field_map.png")


# ---------------------------------------------------------------- 3. log panel
def fig_log_panel(well="Kronos-1", top=4300, base=5330):
    logs = pd.read_parquet(DASH / "well_logs.parquet")
    tops = pd.read_csv(DASH / "well_tops.csv")
    d = logs[(logs["WELL_ID"] == well) & logs["TVDSS_M"].between(top, base)]
    t = tops[(tops["WELL_ID"] == well) & tops["TVDSS_M"].between(top, base)]
    tracks = [("GR_API", "Gamma ray (API)", "#2e7d32", (0, 150)),
              ("RHOB_G_CC", "Density (g/cc)", "#c62828", (2.0, 2.9)),
              ("VP_M_S", "Vp (m/s)", "#1565c0", (2500, 6500)),
              ("VS_M_S", "Vs (m/s)", "#6a1b9a", (1200, 3800)),
              ("AI", "Acoustic impedance", "#ef6c00", (6000, 17000)),
              ("VPVS", "Vp/Vs", "#00838f", (1.4, 2.4))]
    fig, axes = plt.subplots(1, len(tracks), figsize=(13, 8), sharey=True)
    res_top = t[t["SURFACE"].str.contains("Top Reservoir")]["TVDSS_M"]
    res_base = t[t["SURFACE"].str.contains("Nome")]["TVDSS_M"]
    for ax, (col, label, color, lim) in zip(axes, tracks):
        ax.plot(d[col], d["TVDSS_M"], color=color, lw=0.8)
        ax.set_xlim(*lim)
        ax.set_title(label, fontsize=9.5)
        ax.grid(alpha=0.3)
        if len(res_top) and len(res_base):
            ax.axhspan(res_top.iloc[0], res_base.iloc[0], color="#f5d76e", alpha=0.25)
        for _, row in t.iterrows():
            ax.axhline(row["TVDSS_M"], color="#555", lw=0.7, ls="--")
    for _, row in t.iterrows():
        axes[-1].text(axes[-1].get_xlim()[1], row["TVDSS_M"], " " + row["SURFACE"], fontsize=7.5,
                      va="center", clip_on=False)
    axes[0].set_ylim(base, top)
    axes[0].set_ylabel("Depth TVDSS (m)")
    fig.suptitle(f"{well}: logs through the Plover reservoir (yellow = Top Reservoir to Nome)", weight="bold")
    save(fig, "fig03_log_panel.png")


# ---------------------------------------------------------------- 4. crossplot + template
def fig_crossplot():
    logs = pd.read_parquet(DASH / "well_logs.parquet")
    d = logs[logs["FORMATION"].str.contains("Plover", na=False)].dropna(subset=["AI", "VPVS", "GR_API"])
    fig, ax = plt.subplots(figsize=(9, 6.5))
    sc = ax.scatter(d["AI"], d["VPVS"], c=d["GR_API"], cmap="viridis_r", s=4, vmin=10, vmax=120, alpha=0.6)
    fig.colorbar(sc, ax=ax, label="Gamma ray (API): low = clean sand, high = shale")
    phi = np.linspace(0.04, 0.28, 30)
    for sw, label, style in ((1.0, "brine sand model", "-"), (0.2, "gas sand model", "--")):
        vp, vs, rho = rock(phi, 0.05 * np.ones_like(phi), sw * np.ones_like(phi))
        ax.plot(vp * rho, vp / vs, style, color="k", lw=1.8, label=label)
        ax.annotate("φ = 0.28", (vp[-1] * rho[-1], vp[-1] / vs[-1]), fontsize=8, xytext=(-30, 6),
                    textcoords="offset points")
    vsh = np.linspace(0.0, 0.9, 30)
    vp, vs, rho = rock(0.12 * np.ones_like(vsh), vsh, np.ones_like(vsh))
    ax.plot(vp * rho, vp / vs, ":", color="#555", lw=1.8, label="adding clay (φ = 0.12, brine)")
    key = ax.inset_axes([0.015, 0.64, 0.27, 0.34])
    key.set_xlim(-1, 1)
    key.set_ylim(-1, 1.25)
    key.set_xticks([])
    key.set_yticks([])
    key.set_facecolor("white")
    for spine in key.spines.values():
        spine.set_visible(True)
        spine.set_color("#999")
    key.text(0, 1.08, "Which way does a sample move?", ha="center", fontsize=8)
    for (dx, dy), text, ha, va in (((-0.75, 0), "more\nporosity", "right", "center"),
                                   ((0, 0.7), "more clay", "center", "bottom"),
                                   ((-0.6, -0.6), "gas instead\nof brine", "left", "top")):
        key.annotate("", (dx, dy), (0, 0), arrowprops=dict(arrowstyle="-|>", color="#c62828", lw=1.8))
        key.text(dx + (0.05 if ha == "left" else -0.05 if ha == "right" else 0), dy, text, color="#c62828",
                 fontsize=8, ha=ha if ha != "right" else "left", va=va)
    key.plot(0, 0, "ko", ms=4)
    ax.set_xlabel("Acoustic impedance AI = Vp × density  (m/s · g/cc)")
    ax.set_ylabel("Vp/Vs")
    ax.set_title(f"Plover Formation, {d['WELL_ID'].nunique()} Poseidon wells ({len(d):,} log samples)\n"
                 "with an (uncalibrated) rock-physics template")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.set_xlim(5500, 17500)
    ax.set_ylim(1.35, 2.4)
    save(fig, "fig04_elastic_crossplot.png")


# ---------------------------------------------------------------- 5. AVO curves
def interfaces():
    shale = rock(0.12, 0.8, 1.0)
    cases = {
        "brine sand": rock(0.20, 0.05, 1.0),
        "gas sand": rock(0.20, 0.05, 0.2),
        "tight (cemented) sand": rock(0.08, 0.05, 1.0, calcite=0.2),
    }
    return shale, cases


def fig_avo_curves():
    shale, cases = interfaces()
    theta = np.radians(np.linspace(0, 35, 71))
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 1.4]})
    ax0.axis("off")
    ax0.add_patch(Rectangle((0, 0.5), 1, 0.5, fc="#b0b0b0"))
    ax0.add_patch(Rectangle((0, 0), 1, 0.5, fc="#f2d27a"))
    ax0.text(0.05, 0.9, "Shale (cap rock)", weight="bold")
    ax0.text(0.05, 0.08, "Sand (reservoir): brine, gas or cemented", weight="bold")
    src, pt, rec = (0.15, 1.0), (0.5, 0.5), (0.85, 1.0)
    ax0.annotate("", pt, src, arrowprops=dict(arrowstyle="-|>", lw=2))
    ax0.annotate("", rec, pt, arrowprops=dict(arrowstyle="-|>", lw=2, color="#c62828"))
    ax0.plot([0.5, 0.5], [0.5, 0.95], "k:", lw=1)
    ax0.text(0.41, 0.62, "θ", fontsize=14)
    ax0.text(0.62, 0.85, "reflected energy\n= R(θ) × incoming", color="#c62828", fontsize=9)
    ax0.text(0.02, 1.03, "Source", fontsize=9)
    ax0.text(0.80, 1.03, "Receiver", fontsize=9)
    ax0.set_xlim(0, 1)
    ax0.set_ylim(0, 1.1)
    ax0.set_title("A seismic ray reflecting at a shale/sand boundary")
    for (name, sand), color in zip(cases.items(), ["#3b8bd9", "#e0483e", "#9b6fc3"]):
        r = fw.zoeppritz_rpp(shale[0][0], shale[1][0], shale[2][0], sand[0][0], sand[1][0], sand[2][0], theta)
        ax1.plot(np.degrees(theta), r, color=color, lw=2.2, label=f"shale over {name}")
    for st in STACKS:
        ax1.axvspan(st.angle_min, st.angle_max, color=STACK_COLORS[st.name], alpha=0.12)
        ax1.text((st.angle_min + st.angle_max) / 2, 0.97, st.name, transform=ax1.get_xaxis_transform(),
                 ha="center", color=STACK_COLORS[st.name], weight="bold")
    ax1.axhline(0, color="k", lw=0.8)
    ax1.set_xlabel("Incidence angle θ (degrees)")
    ax1.set_ylabel("Reflection coefficient R(θ)")
    ax1.set_title("AVO: how reflection strength changes with angle (exact Zoeppritz)")
    ax1.legend(loc="lower left")
    ax1.grid(alpha=0.3)
    save(fig, "fig05_avo_curves.png")


# ---------------------------------------------------------------- 6. intercept-gradient + MC cloud
def monte_carlo_cloud(n=700, seed=3):
    rng = np.random.default_rng(seed)
    theta = np.radians(np.linspace(2, 32, 7))
    sin2 = np.sin(theta) ** 2
    A = np.c_[np.ones_like(sin2), sin2]
    records = []
    for cls in range(4):
        sh = rock(rng.uniform(0.08, 0.20, n), rng.uniform(0.6, 0.9, n), np.ones(n))
        if cls == 0:
            lower = rock(rng.uniform(0.08, 0.20, n), rng.uniform(0.5, 0.9, n), np.ones(n))
        elif cls == 1:
            lower = rock(rng.uniform(0.02, 0.08, n), rng.uniform(0.0, 0.2, n), np.ones(n),
                         calcite=rng.uniform(0.3, 0.8, n))
        elif cls == 2:
            lower = rock(rng.uniform(0.08, 0.28, n), rng.uniform(0.0, 0.3, n), np.ones(n))
        else:
            lower = rock(rng.uniform(0.08, 0.28, n), rng.uniform(0.0, 0.3, n), rng.uniform(0.1, 0.4, n))
        R = fw.aki_richards(sh[0], sh[1], sh[2], lower[0], lower[1], lower[2], theta)
        coef, *_ = np.linalg.lstsq(A, R.T, rcond=None)
        I, G = coef + rng.normal(0, 0.012, coef.shape)
        records.append(pd.DataFrame({"I": I, "G": G, "cls": cls}))
    return pd.concat(records, ignore_index=True)


def fig_ig_classes(cloud):
    fig, ax = plt.subplots(figsize=(8.5, 7))
    for cls in range(4):
        c = cloud[cloud["cls"] == cls]
        ax.scatter(c["I"], c["G"], s=6, color=CLASS_COLORS[cls], alpha=0.45, label=CLASS_NAMES[cls])
    lim_i, lim_g = (-0.35, 0.5), (-1.0, 0.35)
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(0, color="k", lw=0.8)
    x = np.linspace(-0.35, 0.5, 10)
    ax.plot(x, -x, color="k", ls="--", lw=1, label="background (wet) trend")
    for text, xy in (("Class I\nhard (e.g. tight) sand", (0.33, -0.85)),
                     ("Class II\nnear-zero intercept", (0.02, -0.75)),
                     ("Class III\nsoft sand, bright", (-0.22, -0.45)), ("Class IV", (-0.25, 0.25))):
        ax.text(*xy, text, ha="center", fontsize=9, color="#333",
                bbox=dict(fc="white", ec="none", alpha=0.7))
    ax.set_xlim(*lim_i)
    ax.set_ylim(*lim_g)
    ax.set_xlabel("Intercept I (reflection at zero angle)")
    ax.set_ylabel("Gradient G (change with sin²θ)")
    ax.set_title("Intercept-gradient crossplot: 2,800 simulated shale/X interfaces\n"
                 "(classes overlap - that overlap is the uncertainty we must quantify)")
    ax.legend(loc="upper right", markerscale=2.5, fontsize=8.5)
    ax.grid(alpha=0.3)
    save(fig, "fig06_intercept_gradient.png")


# ---------------------------------------------------------------- 7. Gassmann
def fig_gassmann():
    sw = np.linspace(0, 1, 101)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for fluids, name, color in ((GAS, "gas", "#e0483e"), (OIL, "oil", "#2e7d32")):
        vp, vs, rho = rock(0.2 * np.ones_like(sw), 0.05 * np.ones_like(sw), sw, fluids=fluids)
        axes[0].plot(sw, vp, color=color, lw=2, label=f"brine + {name}")
        axes[1].plot(sw, rho, color=color, lw=2, label=f"brine + {name}")
        axes[2].plot(sw, vp / vs, color=color, lw=2, label=f"brine + {name}")
    for ax, label in zip(axes, ("Vp (m/s)", "Density (g/cc)", "Vp/Vs")):
        ax.set_xlabel("Water saturation Sw (1 = full brine)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8.5)
    axes[0].annotate("a little gas already\ncauses most of the\nVp drop ('fizz' effect)", (0.9, 3440),
                     xytext=(0.03, 3590), arrowprops=dict(arrowstyle="->"), fontsize=8.5)
    fig.suptitle("Gassmann fluid substitution: one sand (φ = 0.20, clean) filled with different fluids",
                 weight="bold")
    fig.tight_layout()
    save(fig, "fig07_gassmann.png")


# ---------------------------------------------------------------- 8. synthetic at Kronos-1
def kronos_time_depth():
    import lasio
    las = lasio.read(str(WELLS / "logs_extra" / "Kronos-1" / "CP_Kronos1_Logs_depth.las"))
    z = las.index.astype(float)
    data = las.data
    owt = data[:, 1]  # ADJS: VSP-adjusted one-way sonic time (s)
    return z, owt, las, data


def read_trace(stack, il, xl, half=1):
    path = SEIS / f"poseidon_{stack}.segy"
    if not path.is_file():
        return None, None
    import segyio
    with segyio.open(str(path), iline=189, xline=193) as f:
        f.mmap()
        line = f.iline[il]
        j = list(f.xlines).index(xl)
        tr = line[max(j - half, 0):j + half + 1].mean(axis=0)
        return np.asarray(f.samples), tr


def fig_synthetic():
    z, owt, las, data = kronos_time_depth()
    names = [c.mnemonic for c in las.curves]
    dtco = data[:, 4]  # DTCO (adjusted / VSP-calibrated)
    dtsm = data[:, names.index("DTSM")]
    rhob = data[:, names.index("RHOB")]
    gr = data[:, names.index("GR")]
    ok = np.isfinite(owt) & np.isfinite(dtco) & np.isfinite(dtsm) & np.isfinite(rhob) & (z > 4300)
    z, twt = z[ok], 2 * owt[ok]
    vp, vs, rho, gr = 304800 / dtco[ok], 304800 / dtsm[ok], rhob[ok], gr[ok]
    dt = 0.001
    t = np.arange(twt[0], twt[-1], dt)
    vp_t, vs_t, rho_t, gr_t = (np.interp(t, twt, a) for a in (vp, vs, rho, gr))
    refl = fw.reflectivity(vp_t, vs_t, rho_t, STACKS)
    wavelet = fw.ricker(25.0, dt)
    synth = np.array([np.convolve(r, wavelet, mode="same") for r in refl])

    tops = pd.read_csv(DASH / "well_tops.csv")
    kt = tops[(tops["WELL_ID"] == "Kronos-1") & tops["TVDSS_M"].between(z[0], z[-1])]
    zz, oo = kronos_time_depth()[:2]
    good = np.isfinite(oo)
    kt = kt.assign(TWT=np.interp(kt["TVDSS_M"], zz[good], 2000 * oo[good]))

    real = {s.name: read_trace(s.name, 2410, 2669) for s in STACKS}
    have_real = all(v[0] is not None for v in real.values())
    ncols = 3 + (2 if have_real else 1)
    fig, axes = plt.subplots(1, ncols, figsize=(3 * ncols + 1, 8.5), sharey=True,
                             gridspec_kw={"width_ratios": [1, 1, 1] + [2] * (ncols - 3)})
    axes[0].plot(gr_t, t * 1000, color="#2e7d32", lw=0.8)
    axes[0].set_title("Gamma ray")
    axes[1].plot(vp_t * rho_t, t * 1000, color="#ef6c00", lw=0.8)
    axes[1].set_title("Acoustic impedance")
    axes[2].plot(vp_t / vs_t, t * 1000, color="#00838f", lw=0.8)
    axes[2].set_title("Vp/Vs")

    def wiggles(ax, traces, times, title):
        scale = np.nanmax(np.abs(np.concatenate(traces))) or 1.0
        for k, (tr, st) in enumerate(zip(traces, STACKS)):
            x = k + 0.45 * tr / scale
            ax.plot(x, times, color="k", lw=0.7)
            ax.fill_betweenx(times, k, x, where=x > k, color=STACK_COLORS[st.name], alpha=0.8)
        ax.set_xticks(range(3))
        ax.set_xticklabels([s.name for s in STACKS])
        ax.set_title(title)

    wiggles(axes[3], list(synth), t * 1000, "Synthetic from logs\n(25 Hz Ricker)")
    if have_real:
        times = real["near"][0]
        sel = (times >= t[0] * 1000) & (times <= t[-1] * 1000)
        wiggles(axes[4], [real[s.name][1][sel] for s in STACKS], times[sel],
                "Real seismic at Kronos-1\n(il 2410 / xl 2669)")
    for ax in axes:
        ax.grid(alpha=0.3)
        for _, row in kt.iterrows():
            ax.axhline(row["TWT"], color="#555", ls="--", lw=0.7)
    for _, row in kt.iterrows():
        axes[-1].text(axes[-1].get_xlim()[1], row["TWT"], " " + row["SURFACE"].replace(" Formation", ""),
                      fontsize=7.5, ha="left", va="center", clip_on=False)
    axes[0].set_ylim(t[-1] * 1000, t[0] * 1000)
    axes[0].set_ylabel("Two-way time (ms), from the VSP-calibrated time-depth curve")
    fig.suptitle("From logs to synthetic seismic at Kronos-1 (a first look, not yet a calibrated well tie)",
                 weight="bold")
    save(fig, "fig08_synthetic_kronos.png")


# ---------------------------------------------------------------- 9. pseudo-wells
def fig_pseudo_wells():
    rng = np.random.default_rng(11)
    wells = priors.sample_prior_b(8, CFG, 0.5, rng)
    cmap = matplotlib.colors.ListedColormap(CLASS_COLORS)
    fig, axes = plt.subplots(1, len(wells), figsize=(13, 6.5), sharey=False)
    for k, (ax, w) in enumerate(zip(axes, wells)):
        from avo import FLUID_OIL, LITHO_SAND, LITHO_SHALE, LITHO_TIGHT
        cls = np.where(w["litho"] == LITHO_SHALE, 0, np.where(w["litho"] == LITHO_TIGHT, 1, 2))
        cls = np.where((w["litho"] == LITHO_SAND) & (w["fluid"] == FLUID_OIL), 3, cls)
        z = np.arange(len(cls)) * w["dz"]
        ax.imshow(cls[:, None], aspect="auto", cmap=cmap, vmin=-0.5, vmax=3.5,
                  extent=[0, 1, z[-1], z[0]])
        ai = w["vp"] * w["rho"]
        ax.plot((ai - 4000) / 14000, z, color="white", lw=0.9)
        ax.set_xticks([])
        ax.set_title(f"pseudo-well {k + 1}", fontsize=9)
        if k == 0:
            ax.set_ylabel("Depth below top of model (m)")
    handles = [Rectangle((0, 0), 1, 1, fc=c) for c in CLASS_COLORS]
    fig.legend(handles, ["shale", "tight", "brine sand", "hydrocarbon sand"], loc="lower center", ncol=4)
    fig.suptitle("Monte Carlo scenarios: 8 random pseudo-wells from the hand-picked prior (B)\n"
                 "white line = acoustic impedance; each pseudo-well gets its own synthetic seismic",
                 weight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "fig09_pseudo_wells.png")


# ---------------------------------------------------------------- 10. classifier posteriors
def fig_classifier(cloud):
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    model = QuadraticDiscriminantAnalysis(reg_param=1e-3).fit(cloud[["I", "G"]].to_numpy(), cloud["cls"])
    gi = np.linspace(-0.35, 0.5, 241)
    gg = np.linspace(-1.0, 0.35, 241)
    II, GG = np.meshgrid(gi, gg)
    P = model.predict_proba(np.c_[II.ravel(), GG.ravel()]).reshape(len(gg), len(gi), 4)
    entropy = -(P * np.log(np.clip(P, 1e-12, 1))).sum(-1) / np.log(4)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    cmap = matplotlib.colors.ListedColormap(CLASS_COLORS)
    axes[0].pcolormesh(II, GG, P.argmax(-1), cmap=cmap, vmin=-0.5, vmax=3.5, shading="auto", alpha=0.85)
    axes[0].set_title("Most likely class")
    m1 = axes[1].pcolormesh(II, GG, P[..., 3], cmap="Reds", vmin=0, vmax=1, shading="auto")
    fig.colorbar(m1, ax=axes[1], label="P(gas sand)")
    axes[1].set_title("Probability of gas sand")
    m2 = axes[2].pcolormesh(II, GG, entropy, cmap="magma_r", vmin=0, vmax=1, shading="auto")
    fig.colorbar(m2, ax=axes[2], label="normalised entropy (0 = certain, 1 = no idea)")
    axes[2].set_title("Uncertainty (entropy)")
    sample = cloud.sample(500, random_state=1)
    for ax in axes:
        ax.scatter(sample["I"], sample["G"], s=3, c="k", alpha=0.25)
        ax.set_xlabel("Intercept I")
        ax.axhline(0, color="k", lw=0.5)
        ax.axvline(0, color="k", lw=0.5)
    axes[0].set_ylabel("Gradient G")
    handles = [Rectangle((0, 0), 1, 1, fc=c) for c in CLASS_COLORS]
    axes[0].legend(handles, CLASS_NAMES, loc="upper right", fontsize=8)
    fig.suptitle("A classifier trained on the simulated scenarios turns (I, G) into class probabilities",
                 weight="bold")
    fig.tight_layout()
    save(fig, "fig10_classifier.png")


# ---------------------------------------------------------------- 11. blind wells
BLIND = ["Poseidon-2", "Boreas-1"]


def fig_blind_wells():
    wells = pd.read_csv(DASH / "wells.csv")
    wells = wells[wells["WELL_ID"] != "Torosa-1"]
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13.5, 5.8), gridspec_kw={"width_ratios": [1, 1.25]})
    outline = seismic_outline()
    if outline is not None:
        ax0.add_patch(Polygon(outline, closed=True, fc="#f3e9c8", ec="#b58b16", lw=1))
    for _, w in wells.iterrows():
        blind = w["WELL_ID"] in BLIND
        ax0.plot(w["X_M"], w["Y_M"], "*" if blind else "o", ms=16 if blind else 9,
                 color="#e0483e" if blind else "#1f4e79", mec="white", zorder=3)
        ax0.annotate(w["WELL_ID"], (w["X_M"], w["Y_M"]), xytext=(7, 5), textcoords="offset points",
                     fontsize=9, weight="bold" if blind else "normal")
    ax0.set_aspect("equal")
    ax0.ticklabel_format(style="plain")
    ax0.set_xlabel("Easting (m)")
    ax0.set_ylabel("Northing (m)")
    ax0.set_title("Proposed split: ★ blind (test) wells, ● training wells")
    ax0.grid(alpha=0.3)

    names = [n for n in wells["WELL_ID"] if n not in BLIND] + BLIND
    n_train = len(names) - len(BLIND)
    grid = np.zeros((n_train, len(names)))
    for r in range(n_train):
        grid[r, r] = 1
    grid[:, n_train:] = 2
    ax1.imshow(grid, cmap=matplotlib.colors.ListedColormap(["#cfe0f3", "#f2b134", "#9a9a9a"]), aspect="auto",
               vmin=-0.5, vmax=2.5)
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=35, ha="right")
    ax1.set_yticks(range(n_train))
    ax1.set_yticklabels([f"round {k + 1}" for k in range(n_train)])
    for r in range(n_train):
        ax1.text(r, r, "validation", ha="center", va="center", fontsize=8, weight="bold")
        for c in range(n_train, len(names)):
            ax1.text(c, r, "locked\naway", ha="center", va="center", color="white", fontsize=7.5)
    ax1.set_title("Development: leave-one-well-out over the training wells (yellow)\n"
                  "Final exam: the blind wells (grey) are only used once, at the very end")
    fig.tight_layout()
    save(fig, "fig11_blind_wells.png")


# ---------------------------------------------------------------- 12. real seismic
def fig_seismic_section(il=3180):
    path = SEIS / "poseidon_near.segy"
    if not path.is_file():
        print("skip fig12: near stack not downloaded")
        return
    import segyio
    with segyio.open(str(path), iline=189, xline=193) as f:
        f.mmap()
        section = f.iline[il]
        xlines = np.asarray(f.xlines)
        times = np.asarray(f.samples)
    clip = np.percentile(np.abs(section), 98)
    zz, oo = kronos_time_depth()[:2]
    good = np.isfinite(oo)
    plover = plover_top_by_well()
    t_top = float(np.interp(plover["Poseidon-1"], zz[good], 2000 * oo[good]))
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.imshow(section.T, cmap="RdBu_r", vmin=-clip, vmax=clip, aspect="auto",
              extent=[xlines[0], xlines[-1], times[-1], times[0]])
    ax.axvline(2337, color="k", lw=1.5)
    ax.text(2337, times[0] + 15, " Poseidon-1", fontsize=10, weight="bold", va="top")
    ax.plot([2337 - 25, 2337 + 25], [t_top, t_top], color="#f5b700", lw=3)
    ax.text(2337 + 30, t_top, "approx. top Plover reservoir", color="#8a6500", va="center", fontsize=9,
            weight="bold")
    ax.set_xlabel("Crossline")
    ax.set_ylabel("Two-way time (ms)")
    ax.set_title(f"Real Poseidon near-angle stack, inline {il} (through Poseidon-1)\n"
                 "red/blue = positive/negative amplitude; top Plover time from the Kronos-1 time-depth curve")
    save(fig, "fig12_seismic_section.png")

    stacks = {s.name: SEIS / f"poseidon_{s.name}.segy" for s in STACKS}
    if not all(p.is_file() for p in stacks.values()):
        print("skip fig12b: mid/far stacks not downloaded yet")
        return
    panels = {}
    for name, p in stacks.items():
        with segyio.open(str(p), iline=189, xline=193) as f:
            f.mmap()
            panels[name] = f.iline[il]
    x0, x1 = 2337 - 150, 2337 + 150
    j0, j1 = np.searchsorted(xlines, x0), np.searchsorted(xlines, x1)
    k0, k1 = np.searchsorted(times, t_top - 300), np.searchsorted(times, t_top + 400)
    clip = max(np.percentile(np.abs(v[j0:j1, k0:k1]), 98) for v in panels.values())
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5), sharey=True)
    for ax, (name, data) in zip(axes, panels.items()):
        ax.imshow(data[j0:j1, k0:k1].T, cmap="RdBu_r", vmin=-clip, vmax=clip, aspect="auto",
                  extent=[xlines[j0], xlines[j1 - 1], times[k1 - 1], times[k0]])
        ax.axvline(2337, color="k", lw=1)
        ax.axhline(t_top, color="#f5b700", lw=1.5, ls="--")
        ax.set_title(f"{name} stack", color=STACK_COLORS[name], weight="bold")
        ax.set_xlabel("Crossline")
    axes[0].set_ylabel("Two-way time (ms)")
    fig.suptitle("Same place, three angle ranges: amplitude differences between near, mid and far are the AVO "
                 "signal we classify", weight="bold")
    fig.tight_layout()
    save(fig, "fig12b_near_mid_far.png")


# ---------------------------------------------------------------- 13. synthetic end products
def fig_copy_synthetic_examples():
    src = ROOT / "outputs" / "synthetic"
    for name, target in (("maps.png", "fig13a_example_maps.png"), ("section.png", "fig13b_example_section.png"),
                         ("reliability.png", "fig13c_example_reliability.png"),
                         ("blind_W3.png", "fig13d_example_blind_well.png")):
        if (src / name).is_file():
            OUT.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src / name, OUT / target)
            print("copied", target)


# ---------------------------------------------------------------- 14. reading probabilities
def fig_uncertainty_examples():
    examples = [("A: confident gas sand", [0.03, 0.02, 0.10, 0.85]),
                ("B: brine or gas? (fluid ambiguity)", [0.05, 0.05, 0.47, 0.43]),
                ("C: rock type unclear (lithology ambiguity)", [0.38, 0.34, 0.18, 0.10])]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, (title, p) in zip(axes, examples):
        p = np.array(p)
        h = -(p * np.log(p)).sum() / np.log(len(p))
        ax.bar(CLASS_NAMES, p, color=CLASS_COLORS)
        ax.set_ylim(0, 1)
        ax.set_title(f"{title}\nentropy = {h:.2f}", fontsize=10)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Probability")
    fig.suptitle("How to read the output at one location: probabilities sum to 1; entropy summarises the spread",
                 weight="bold")
    fig.tight_layout()
    save(fig, "fig14_reading_probabilities.png")


# ---------------------------------------------------------------- 15. scenarios / what-if
def fig_scenarios():
    shale = rock(0.12, 0.8, 1.0)
    scenarios = {
        "base: gas sand φ 0.20": rock(0.20, 0.05, 0.2),
        "brine instead of gas": rock(0.20, 0.05, 1.0),
        "porosity -0.05": rock(0.15, 0.05, 0.2),
        "porosity +0.05": rock(0.25, 0.05, 0.2),
        "more clay (Vsh 0.25)": rock(0.20, 0.25, 0.2),
        "cemented (tight)": rock(0.05, 0.05, 1.0, calcite=0.5),
    }
    rows = []
    for name, sand in scenarios.items():
        r = []
        for st in STACKS:
            R = fw.aki_richards(shale[0], shale[1], shale[2], sand[0], sand[1], sand[2], fw.stack_angles(st))
            r.append(R.mean())
        rows.append((name, *r))
    df = pd.DataFrame(rows, columns=["scenario", *[s.name for s in STACKS]])
    fig, ax = plt.subplots(figsize=(11, 4.8))
    width = 0.26
    x = np.arange(len(df))
    for k, st in enumerate(STACKS):
        ax.bar(x + (k - 1) * width, df[st.name], width, color=STACK_COLORS[st.name], label=f"{st.name} stack")
    ax.set_xticks(x)
    ax.set_xticklabels(df["scenario"], rotation=15)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("Reflection coefficient at top sand")
    ax.set_title("What-if scenarios: each geological change gives a different near/mid/far signature",
                 weight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    save(fig, "fig15_scenarios.png")


def main():
    fig_workflow()
    fig_field_map()
    fig_log_panel()
    fig_crossplot()
    fig_avo_curves()
    cloud = monte_carlo_cloud()
    fig_ig_classes(cloud)
    fig_gassmann()
    fig_synthetic()
    fig_pseudo_wells()
    fig_classifier(cloud)
    fig_blind_wells()
    fig_seismic_section()
    fig_copy_synthetic_examples()
    fig_uncertainty_examples()
    fig_scenarios()


if __name__ == "__main__":
    main()
