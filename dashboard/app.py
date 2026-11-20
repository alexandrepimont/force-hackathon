"""Poseidon AVO workflow: a step-by-step app from data to a calibrated well tie.

Run:  .\\.venv\\Scripts\\streamlit.exe run dashboard\\app.py
The previous free-form explorer (incl. the synthetic dataset) is kept in dashboard/legacy_explorer.py.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from src.dashboard_data import (  # noqa: E402
    build_seismic_cache,
    find_seismic_stacks,
    read_seismic_geometry,
    read_seismic_section,
    read_seismic_traces,
    seismic_cache_ready,
    survey_outline,
    traverse_indices,
    wells_on_grid,
)

DASH = ROOT / "data" / "poseidon" / "dashboard"
QC_LOGS = ROOT / "data" / "interim" / "poseidon_logs"
TD_DIR = ROOT / "data" / "interim" / "poseidon_td"
QC_DIR = ROOT / "outputs" / "well_qc"
TIE_DIR = ROOT / "outputs" / "well_ties"
CAL_PATH = ROOT / "configs" / "calibration.yaml"
DECISION_LOG = ROOT / "outputs" / "decision_log.md"

BLUE, GREEN, AMBER, RED, GREY = "#1f6feb", "#2da44e", "#bf8700", "#cf222e", "#6e7781"
STACK_COLORS = {"near": "#1f77b4", "mid": "#2ca02c", "far": "#ff7f0e"}
LITHO_COLORS = {"shale": "#a3a8ae", "tight": "#8e6cc9", "sand_low_phi": "#f2c94c", "sand_high_phi": "#e8833a",
                "volcanic": "#3d3d3d", "unknown": "#ffffff"}
LITHO_LABELS = {"shale": "Shale", "tight": "Tight", "sand_low_phi": "Sand, low porosity",
                "sand_high_phi": "Sand, high porosity", "volcanic": "Volcanic", "unknown": "Unknown"}
DEFAULT_TRAVERSE = ["Kronos-1", "Poseidon-2", "Boreas-1", "Poseidon-1", "Poseidon-North-1"]

st.set_page_config(page_title="Poseidon AVO workflow", page_icon="🪨", layout="wide")
st.markdown(
    """
    <style>
      .block-container {padding-top: 4.2rem; padding-bottom: 3rem; max-width: 1500px;}
      .step-kicker {color:#1f6feb; font-weight:700; letter-spacing:.08em; font-size:.8rem; text-transform:uppercase;}
      .chip {display:inline-block; padding:6px 12px; margin:3px 6px 3px 0; border-radius:999px; font-size:.85rem;
             border:1px solid #d0d7de; background:#f6f8fa; color:#1f2933;}
      .chip.done {background:#dafbe1; border-color:#4ac26b;}
      .chip.attention {background:#fff8c5; border-color:#d4a72c;}
      .chip.todo {background:#f6f8fa; color:#6e7781;}
      div[data-testid="stMetricValue"] {font-size:1.5rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------------------------- data access
def _mtime(path: Path) -> int:
    return path.stat().st_mtime_ns if path.exists() else 0


@st.cache_data(show_spinner=False)
def _csv(path: str, mtime: int) -> pd.DataFrame:
    del mtime
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def _parquet(path: str, mtime: int) -> pd.DataFrame:
    del mtime
    return pd.read_parquet(path)


def csv(path: Path) -> pd.DataFrame | None:
    return _csv(str(path), _mtime(path)) if path.is_file() else None


def wells() -> pd.DataFrame:
    return csv(DASH / "wells.csv")


def tops() -> pd.DataFrame:
    return csv(DASH / "well_tops.csv")


def qc_logs(well: str) -> pd.DataFrame | None:
    path = QC_LOGS / f"{well}.parquet"
    return _parquet(str(path), _mtime(path)) if path.is_file() else None


def dashboard_logs() -> pd.DataFrame | None:
    path = DASH / "well_logs.parquet"
    return _parquet(str(path), _mtime(path)) if path.is_file() else None


@st.cache_data(show_spinner=False)
def _tie(well: str, mtime: int):
    del mtime
    with np.load(TIE_DIR / f"{well}.npz") as data:
        return {key: data[key] for key in data.files}


def tie(well: str):
    path = TIE_DIR / f"{well}.npz"
    return _tie(well, _mtime(path)) if path.is_file() else None


def tie_meta(well: str):
    path = TIE_DIR / f"{well}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _short_top(name: str) -> str:
    for word in (" Formation", " Member", " Fm"):
        name = name.replace(word, "")
    return name


def add_well_tops(fig, well, x, t_lo, t_hi, half_width, row=None, col=None, labels=True):
    """Tops of a tied well (TWT after the tie) as short ticks across the well line, labelled on the right.
    The ticks are markers, so on a clickable section a click on a top returns its time."""
    data = tie(well)
    if data is None:
        return
    tops = sorted((float(t), _short_top(str(n))) for n, t in zip(data["tops_names"], data["tops_twt_ms"])
                  if np.isfinite(t) and t_lo <= t <= t_hi)
    if not tops:
        return
    kw = {"row": row, "col": col} if row else {}
    xs, ys, txt = [], [], []
    for t, name in tops:
        xs += [x - half_width, x + half_width, None]
        ys += [t, t, None]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line={"color": "#111111", "width": 3}, hoverinfo="skip",
                             showlegend=False), **kw)
    fig.add_trace(go.Scatter(x=[x] * len(tops), y=[t for t, _ in tops], mode="markers",
                             marker={"symbol": "line-ew", "size": 30, "opacity": 0,
                                     "line": {"width": 8, "color": "#111111"}},
                             unselected={"marker": {"opacity": 0}}, selected={"marker": {"opacity": 0}},
                             text=[n for _, n in tops], showlegend=False,
                             hovertemplate=f"{well}: %{{text}}<br>%{{y:.0f}} ms<extra>top</extra>"), **kw)
    if labels:
        for (t, name), ty in zip(tops, spaced_labels([t for t, _ in tops], (t_hi - t_lo) / 28)):
            fig.add_annotation(x=x + half_width, y=ty, text=name, showarrow=False, xanchor="left",
                               font={"size": 10, "color": "#111111"}, bgcolor="rgba(255,255,255,0.8)",
                               borderpad=1, **kw)


def calibration():
    return yaml.safe_load(CAL_PATH.read_text()) if CAL_PATH.is_file() else None


def g2_status() -> str | None:
    if not DECISION_LOG.is_file():
        return None
    text = DECISION_LOG.read_text(encoding="utf-8")
    return "PASS" if "**PASS**" in text else "FAIL" if "**FAIL**" in text else None


@st.cache_data(show_spinner=False)
def _geometry(path: str, mtime: int):
    del mtime
    return read_seismic_geometry(Path(path))


@st.cache_data(show_spinner=False, max_entries=30)
def _section(path: str, kind: str, value: float, mtime: int):
    del mtime
    return read_seismic_section(Path(path), kind, value)


@st.cache_data(show_spinner=False, max_entries=12)
def _traverse(path: str, mtime: int, points: tuple):
    geometry = _geometry(path, mtime)
    route = traverse_indices(list(points), geometry["ilines"], geometry["xlines"], geometry)
    return route, read_seismic_traces(Path(path), route["i"], route["j"])


@st.cache_data(show_spinner=False)
def _positions(path: str, mtime: int, wells_frame: pd.DataFrame):
    return wells_on_grid(_geometry(path, mtime), wells_frame)


def seismic():
    """Ready stacks (cache built), geometry and well positions on the grid."""
    stacks = find_seismic_stacks(DASH)
    ready = {name: path for name, path in stacks.items() if seismic_cache_ready(path)}
    if not ready:
        return stacks, {}, None, None
    first = next(iter(ready.values()))
    geometry = _geometry(str(first), _mtime(first))
    w = wells()
    positions = _positions(str(first), _mtime(first), w[["WELL_ID", "X_M", "Y_M"]]) if w is not None else None
    return stacks, ready, geometry, positions


def section(ready, stack, kind, value):
    path = ready[stack]
    return _section(str(path), kind, value, _mtime(path))


# ----------------------------------------------------------------------------------------------- status
def step_status() -> dict[str, str]:
    stacks = find_seismic_stacks(DASH)
    data_ok = wells() is not None and len(stacks) == 3
    qc_ok = (QC_DIR / "summary.csv").is_file()
    tie_summary = csv(TIE_DIR / "summary.csv")
    tie_state = "todo"
    if tie_summary is not None:
        review = tie_summary.get("needs_review", pd.Series(dtype=bool)).fillna(False).astype(bool).any()
        tie_state = "attention" if review or g2_status() != "PASS" else "done"
    cache_ok = all(seismic_cache_ready(p) for p in stacks.values()) and bool(stacks)
    return {
        "data": "done" if data_ok else "todo",
        "logs": "done" if dashboard_logs() is not None else "todo",
        "qc": "done" if qc_ok else "todo",
        "seismic": "done" if cache_ok else ("attention" if stacks else "todo"),
        "tie": tie_state,
        "traverse": tie_state if tie_state != "todo" else ("done" if cache_ok else "todo"),
        "rockphysics": "done" if qc_ok else "todo",
        "next": "todo",
        "sim": "done" if sim_runs() else "todo",
        "sim3d": "done" if sim3d_runs() else "todo",
    }


STATUS_ICON = {"done": "✅", "attention": "⚠️", "todo": "⚪"}


# ----------------------------------------------------------------------------------------------- UI helpers
def step_header(number, title, purpose, look_for=None):
    step_nav(ORDER[number or 0], top=True)
    st.markdown(f"<div class='step-kicker'>{'Step ' + str(number) if number else 'Start'}</div>",
                unsafe_allow_html=True)
    st.title(title)
    st.markdown(purpose)
    if look_for:
        st.info("**What to look for:** " + look_for)


def well_picker(options: list[str], key: str, label: str = "Well") -> str:
    current = st.session_state.get("well")
    if current not in options:
        current = options[0]
    choice = st.segmented_control(label, options, default=current, key=f"well_{key}")
    choice = choice or current
    st.session_state["well"] = choice
    return choice


ORDER = ["overview", "data", "logs", "qc", "seismic", "tie", "traverse", "rockphysics", "next", "sim", "sim3d"]
NAV_LABELS = {"overview": "Overview", "data": "1 Data", "logs": "2 Well logs", "qc": "3 Well QC",
              "seismic": "4 Seismic", "tie": "5 Well tie", "traverse": "6 Seismic through wells",
              "rockphysics": "7 Rock physics", "next": "8 AVO classification",
              "sim": "9 Prospect simulation", "sim3d": "10 3D extrapolation"}


def step_nav(current: str, top: bool = False):
    """Back / next links so the workflow can be walked in both directions."""
    i = ORDER.index(current)
    if not top:
        st.divider()
    left, _, right = st.columns([1, 2, 1])
    if i > 0:
        left.page_link(PAGES[ORDER[i - 1]], label=f"Back: {NAV_LABELS[ORDER[i - 1]]}", icon="⬅️")
    if i < len(ORDER) - 1:
        right.page_link(PAGES[ORDER[i + 1]], label=f"Next: {NAV_LABELS[ORDER[i + 1]]}", icon="➡️")


def style(fig, height, legend_below=True, **kw):
    fig.update_layout(template="plotly_white", height=height, font={"size": 12},
                      margin={"l": 60, "r": 30, "t": 50, "b": 70 if legend_below else 40},
                      legend={"orientation": "h", "yanchor": "top", "y": -0.10, "x": 0} if legend_below else {},
                      **kw)
    return fig


def spaced_labels(values, min_gap):
    """Shift label positions so neighbouring labels are at least min_gap apart (input sorted)."""
    out = []
    for v in values:
        out.append(v if not out else max(v, out[-1] + min_gap))
    return out


def run_script(script: str, label: str):
    with st.status(f"Running {label}...", expanded=False) as status:
        result = subprocess.run([sys.executable, str(ROOT / "scripts" / script)], cwd=str(ROOT), capture_output=True,
                                text=True, timeout=1800)
        st.code((result.stdout or "")[-4000:] + (result.stderr or "")[-2000:], language="text")
        if result.returncode == 0:
            status.update(label=f"{label} finished", state="complete")
            st.cache_data.clear()
        else:
            status.update(label=f"{label} failed (see output)", state="error")


def basemap(geometry, positions, lines=(), highlight=False, height=380, title=None, clickable=False):
    w = wells()
    ox, oy = survey_outline(geometry)
    fig = go.Figure(go.Scatter(x=ox, y=oy, mode="lines", fill="toself", name="Seismic area",
                               fillcolor="rgba(31,111,235,0.18)" if highlight else "rgba(31,111,235,0.06)",
                               line={"color": BLUE, "width": 1.5}, hoverinfo="skip"))
    if clickable:
        # invisible clickable grid: plotly only reports clicks on markers
        step = 24
        ii, jj = np.meshgrid(geometry["ilines"][::step], geometry["xlines"][::step], indexing="ij")
        fig.add_trace(go.Scatter(x=geometry["cdp_x"][::step, ::step].ravel(),
                                 y=geometry["cdp_y"][::step, ::step].ravel(), mode="markers", name="click",
                                 marker={"size": 9, "color": "rgba(31,111,235,0.04)"},
                                 customdata=np.c_[ii.ravel(), jj.ravel()],
                                 hovertemplate="IL %{customdata[0]} / XL %{customdata[1]}<extra>click</extra>"))
    for x, y, label in lines:
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=label, line={"color": RED, "width": 4}))
    selected = st.session_state.get("well")
    inside_ids = set(positions.dropna(subset=["INLINE"])["WELL_ID"]) if positions is not None else set(w["WELL_ID"])
    inside = w[w["WELL_ID"].isin(inside_ids)]
    fig.add_trace(go.Scatter(
        x=inside["X_M"], y=inside["Y_M"], mode="markers+text", text=inside["WELL_ID"], textposition="top center",
        textfont={"size": 10}, name="Wells", hovertemplate="%{text}<extra></extra>",
        marker={"size": [13 if n == selected else 9 for n in inside["WELL_ID"]],
                "color": [RED if n == selected else "#24292f" for n in inside["WELL_ID"]]}))
    pad = 1500
    fig.update_xaxes(range=[ox.min() - pad, ox.max() + pad], showticklabels=False, title=None)
    fig.update_yaxes(range=[oy.min() - pad, oy.max() + pad], showticklabels=False, scaleanchor="x", title=None)
    style(fig, height, legend_below=False, showlegend=False,
          title={"text": title or "Location", "font": {"size": 13}})
    fig.update_layout(margin={"l": 10, "r": 10, "t": 40, "b": 10})
    if clickable:
        fig.update_layout(clickmode="event+select", dragmode="pan")
        fig.update_traces(unselected={"marker": {"opacity": 1}}, selector={"type": "scatter", "mode": "markers"})
        fig.update_traces(unselected={"marker": {"opacity": 1}}, selector={"type": "scatter", "mode": "markers+text"})
    return fig


def seismic_heatmap(z, x, y, clip, colorscale, showscale, hover):
    return go.Heatmap(x=x, y=y, z=np.round(z), zmin=-clip, zmax=clip, zmid=0, colorscale=colorscale,
                      showscale=showscale, colorbar={"title": "Amp.", "len": 0.8}, hovertemplate=hover)


def _step(n, max_points):
    return max(1, int(np.ceil(n / max_points)))


# ----------------------------------------------------------------------------------------------- pages
def page_overview():
    step_header(None, "Poseidon AVO workflow",
                "We want to map **rock type** (lithotype) and, where the data allow it, **fluid** in the Plover "
                "reservoir of the Poseidon field in 3D, using the near, mid and far angle stacks calibrated to the "
                "wells. This app walks through the workflow one step at a time. Use the menu on the left or the "
                "*Next* link at the bottom of each page.")
    status = step_status()
    chips = [("1 Data", "data"), ("2 Well logs", "logs"), ("3 Well QC", "qc"), ("4 Seismic", "seismic"),
             ("5 Well tie", "tie"), ("6 Seismic through wells", "traverse"), ("7 Rock physics", "rockphysics"),
             ("8 AVO classification", "next")]
    st.markdown("".join(f"<span class='chip {status[k]}'>{STATUS_ICON[status[k]]} {t}</span>" for t, k in chips),
                unsafe_allow_html=True)
    st.caption("✅ done · ⚠️ done, but something needs your review · ⚪ not started")

    w = wells()
    if w is None:
        st.error("No well data found. Run the download and preparation scripts (see step 1).")
        return
    tie_summary = csv(TIE_DIR / "summary.csv")
    stacks, ready, geometry, positions = seismic()
    m = st.columns(5)
    m[0].metric("Wells", len(w))
    m[1].metric("Inside seismic", int(positions["INLINE"].notna().sum()) if positions is not None else 0)
    m[2].metric("Seismic stacks", f"{len(ready)} / 3")
    m[3].metric("Wells tied", int((tie_summary["status"] == "tied").sum()) if tie_summary is not None else 0)
    g2 = g2_status()
    m[4].metric("Gate G2 (tie)", g2 or "not run")

    if geometry is not None:
        left, right = st.columns([1.4, 1])
        with left:
            fig = basemap(geometry, positions, height=520, title="Wells and seismic area")
            st.plotly_chart(fig, width="stretch", key="overview_map")
        with right:
            st.markdown("#### Well status")
            table = w[["WELL_ID"]].rename(columns={"WELL_ID": "Well"})
            table["In seismic"] = table["Well"].map(
                lambda n: "yes" if positions is not None and n in set(positions.dropna(subset=["INLINE"])["WELL_ID"])
                else "no")
            if tie_summary is not None:
                ts = tie_summary.set_index("well")
                table["Tie"] = table["Well"].map(lambda n: _tie_label(ts, n))
                table["Role"] = table["Well"].map(
                    lambda n: "held out (blind)" if n in ts.index and str(ts.loc[n].get("held_out")) == "True"
                    else ("training" if n in ts.index and ts.loc[n]["status"] == "tied" else "-"))
            st.dataframe(table, hide_index=True, width="stretch", height=330)
    step_nav("overview")


def _tie_label(ts, name):
    if name not in ts.index:
        return "-"
    r = ts.loc[name]
    if r["status"] != "tied":
        return "not tied"
    return f"{'⚠️ review' if bool(r.get('needs_review')) else '✅ ok'} (r = {r['corr_after_mean']:.2f})"


def page_data():
    step_header(1, "Data",
                "Everything starts with two data sets: **well logs** (what the rock is at the wells) and the "
                "**near, mid and far angle stacks** (how the subsurface reflects at small, medium and large angles).",
                "every well should have P-sonic, S-sonic and density in the reservoir; all three stacks should be "
                "present with a display cache.")
    w, logs = wells(), dashboard_logs()
    stacks, ready, geometry, positions = seismic()
    t = tops()
    st.subheader("Wells")
    if w is None or logs is None:
        st.error("Run `scripts/download_poseidon_wells.py` and `scripts/prepare_poseidon_dashboard.py`.")
        return
    zone = logs[logs["TVDSS_M"] >= 4000]
    rows = []
    in_seismic = set(positions.dropna(subset=["INLINE"])["WELL_ID"]) if positions is not None else set()
    for name in w["WELL_ID"]:
        z = zone[zone["WELL_ID"] == name]

        def has(col):
            return "✅" if z[col].notna().mean() > 0.2 else "–"
        rows.append({"Well": name, "Gamma ray": has("GR_API"), "Density": has("RHOB_G_CC"),
                     "Neutron": has("NPHI_VV"), "P-sonic": has("DTC_US_FT"), "S-sonic": has("DTS_US_FT"),
                     "Tops": int((t["WELL_ID"] == name).sum()) if t is not None else 0,
                     "Own time-depth": "✅" if name in ("Kronos-1", "Pharos-1") else "–",
                     "In seismic": "✅" if name in in_seismic else "–"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Curves count as present when more than 20 % of the interval below 4000 m TVDSS has values. "
               "Wells without their own time-depth curve borrow one from the nearest well in step 5.")

    st.subheader("Seismic")
    seis_cfg = yaml.safe_load((ROOT / "configs" / "poseidon_seismic.yaml").read_text())
    c1, c2 = st.columns([1, 1])
    with c1:
        st.dataframe(pd.DataFrame([{
            "Stack": name, "Angles": f"{seis_cfg['stacks'][name]['angle_min']}–{seis_cfg['stacks'][name]['angle_max']}°",
            "File": stacks[name].name if name in stacks else "missing",
            "Fast display cache": "✅" if name in ready else ("build needed" if name in stacks else "–")}
            for name in ("near", "mid", "far")]), hide_index=True, width="stretch")
        st.markdown(
            f"- **Area:** inlines {seis_cfg['geometry']['window']['inlines'][0]}–"
            f"{seis_cfg['geometry']['window']['inlines'][1]}, crosslines "
            f"{seis_cfg['geometry']['window']['crosslines'][0]}–{seis_cfg['geometry']['window']['crosslines'][1]}, "
            f"{seis_cfg['geometry']['window']['twt_ms'][0]}–{seis_cfg['geometry']['window']['twt_ms'][1]} ms\n"
            f"- **Polarity:** reverse (a trough is an increase in impedance)\n"
            f"- **Coordinates:** {seis_cfg['crs']}\n"
            "- **Amplitudes:** post-stack time-variant scaling is not documented, so each stack gets one global "
            "scale from the training wells (step 5).")
        missing = [n for n in stacks if n not in ready]
        if missing and st.button(f"Build display cache for {', '.join(missing)} (about 1 minute each)"):
            with st.spinner("Building cache..."):
                for n in missing:
                    build_seismic_cache(stacks[n])
            st.cache_data.clear()
            st.rerun()
    with c2:
        if geometry is not None:
            st.plotly_chart(basemap(geometry, positions, height=360, title="Seismic area and wells"),
                            width="stretch", key="data_map")
    with st.expander("How to download or rebuild the data"):
        st.code("\n".join([
            r".\.venv\Scripts\python.exe scripts\download_poseidon_wells.py",
            r".\.venv\Scripts\python.exe scripts\download_poseidon_seismic.py",
            r".\.venv\Scripts\python.exe scripts\prepare_poseidon_dashboard.py"]), language="powershell")
    step_nav("data")


def page_logs():
    step_header(2, "Well logs",
                "The logs tell us what the rock is: **gamma ray** separates sand from shale, **density** and "
                "**neutron** show porosity and gas, and **Vp, Vs** give the elastic properties that seismic "
                "responds to. The right-hand column shows the rock type from the well QC (step 3).",
                "low gamma ray + low density + low Vp/Vs together usually means porous, possibly gas-bearing sand.")
    w = wells()
    if w is None:
        st.error("No wells found.")
        return
    well = well_picker(list(w["WELL_ID"]), "logs")
    frame = qc_logs(well)
    if frame is None:
        st.warning("Well QC has not been run yet, showing the raw prepared logs.")
        d = dashboard_logs()
        d = d[d["WELL_ID"] == well].rename(columns={"TVDSS_M": "tvdss", "GR_API": "gr", "RHOB_G_CC": "rho",
                                                    "NPHI_VV": "nphi", "VP_M_S": "vp", "VS_M_S": "vs"})
        frame = d.assign(lithotype="unknown")
    zmin, zmax = float(np.nanmin(frame["tvdss"])), float(np.nanmax(frame["tvdss"]))
    depth = st.slider("Depth interval (m TVDSS)", zmin, zmax, (max(zmin, 3500.0), zmax), 10.0, key=f"logs_depth_{well}")
    d = frame[frame["tvdss"].between(*depth)]
    well_tops = tops()
    well_tops = well_tops[(well_tops["WELL_ID"] == well) & well_tops["TVDSS_M"].between(*depth)].sort_values("TVDSS_M")
    st.plotly_chart(log_figure(d, well_tops), width="stretch", key=f"logs_fig_{well}")
    st.caption("Depth is TVDSS assuming vertical wells (no deviation surveys available; error < 16 m at TD). "
               "Dashed lines are formation tops.")
    step_nav("logs")


def log_figure(d, well_tops, height=760):
    tracks = [("Gamma ray (API)", [("gr", "#2da44e", None)], (0, 200)),
              ("Density (g/cc)", [("rho", "#cf222e", None)], (1.95, 2.95)),
              ("Neutron (v/v)", [("nphi", "#1f6feb", None)], (0.45, -0.05)),
              ("Velocity (m/s)", [("vp", "#1f6feb", "Vp"), ("vs", "#8250df", "Vs")], (1000, 6500)),
              ("Acoustic impedance", [("ai", "#bf8700", None)], None),
              ("Vp/Vs", [("vpvs", "#0a7b83", None)], (1.3, 2.6))]
    d = d.assign(ai=d["vp"] * d["rho"], vpvs=d["vp"] / d["vs"])
    n = len(tracks) + 2
    fig = make_subplots(rows=1, cols=n, shared_yaxes=True, horizontal_spacing=0.012,
                        column_widths=[1] * len(tracks) + [0.35, 0.9],
                        subplot_titles=[t[0] for t in tracks] + ["Rock", "Tops"])
    for col, (_, curves, rng) in enumerate(tracks, start=1):
        for name, color, label in curves:
            if name in d and d[name].notna().any():
                fig.add_trace(go.Scatter(x=d[name], y=d["tvdss"], mode="lines", line={"color": color, "width": 1},
                                           name=label or name, showlegend=bool(label),
                                           hovertemplate=f"{label or name} %{{x:.3g}}<br>%{{y:.0f}} m<extra></extra>"),
                              1, col)
        if rng:
            fig.update_xaxes(range=list(rng), row=1, col=col)
    if "lithotype" in d:
        codes = {k: i for i, k in enumerate(LITHO_COLORS)}
        z = d["lithotype"].map(codes).fillna(codes["unknown"]).to_numpy()[:, None]
        scale = []
        for i, color in enumerate(LITHO_COLORS.values()):
            scale += [[i / len(LITHO_COLORS), color], [(i + 1) / len(LITHO_COLORS), color]]
        fig.add_trace(go.Heatmap(z=z, y=d["tvdss"], x=[0], colorscale=scale, zmin=-0.5,
                                 zmax=len(LITHO_COLORS) - 0.5, showscale=False,
                                 customdata=d["lithotype"].map(LITHO_LABELS).to_numpy()[:, None],
                                 hovertemplate="%{customdata}<br>%{y:.0f} m<extra></extra>"), 1, n - 1)
        for label, color in LITHO_COLORS.items():
            if label != "unknown" and (d["lithotype"] == label).any():
                fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=LITHO_LABELS[label],
                                         marker={"size": 12, "symbol": "square", "color": color}), 1, n - 1)
    fig.update_xaxes(showticklabels=False, row=1, col=n - 1)
    if len(well_tops):
        span = float(d["tvdss"].max() - d["tvdss"].min()) or 1.0
        label_y = spaced_labels(list(well_tops["TVDSS_M"]), span / 38)
        for (_, top), y in zip(well_tops.iterrows(), label_y):
            for col in range(1, n):
                fig.add_hline(y=top["TVDSS_M"], line={"color": "#57606a", "width": 0.8, "dash": "dash"}, row=1, col=col)
            fig.add_trace(go.Scatter(x=[0, 0.12], y=[top["TVDSS_M"], y], mode="lines", line={"color": "#8c959f", "width": 0.8},
                                     showlegend=False, hoverinfo="skip"), 1, n)
            fig.add_trace(go.Scatter(x=[0.14], y=[y], mode="text", text=[top["SURFACE"].replace(" Formation", "")],
                                     textposition="middle right", textfont={"size": 10}, showlegend=False,
                                     hoverinfo="skip"), 1, n)
        fig.update_xaxes(range=[0, 1], showticklabels=False, showgrid=False, zeroline=False, row=1, col=n)
    fig.update_yaxes(autorange="reversed", title_text="TVDSS (m)", row=1, col=1)
    style(fig, height)
    fig.update_annotations(font_size=11)
    return fig


def page_qc():
    step_header(3, "Well QC and petrophysics",
                "Before logs can be trusted for a well tie, we check them: **bad hole** (washouts) is flagged, "
                "**shale volume**, **porosity** and **water saturation** are computed, and every sample gets a "
                "**rock type** (a petrophysical scenario, not ground truth). Fluid is *not* labelled from logs; only "
                "a gas indicator is stored.",
                "few flagged samples in the reservoir, sensible porosities (sands 0.08–0.25) and whether S-wave "
                "velocity is measured or estimated.")
    c = st.columns([1, 3])
    if c[0].button("Run well QC", type="primary", help="scripts/run_well_qc.py, ~20 s"):
        run_script("run_well_qc.py", "well QC")
    c[1].caption("Parameters: `configs/petrophysics.yaml` (cut-offs, Archie, held-out wells).")
    summary = csv(QC_DIR / "summary.csv")
    if summary is None:
        st.warning("Well QC has not been run yet.")
        return
    table = summary.rename(columns={
        "well": "Well", "flagged_fraction_zone": "Flagged", "gross_m": "Gross (m)", "net_sand_m": "Net sand (m)",
        "net_to_gross": "N/G", "mean_phi_net": "Porosity", "mean_sw_net": "Sw", "gas_indicated_m": "Gas indicated (m)",
        "vs_measured_fraction_res": "Vs measured", "held_out": "Held out"})
    cols = ["Well", "Held out", "Flagged", "Gross (m)", "Net sand (m)", "N/G", "Porosity", "Sw", "Gas indicated (m)",
            "Vs measured"]
    st.dataframe(table[cols], hide_index=True, width="stretch", column_config={
        "Flagged": st.column_config.ProgressColumn("Flagged (zone)", format="percent", min_value=0, max_value=0.2),
        "Vs measured": st.column_config.ProgressColumn("Vs measured (reservoir)", format="percent", min_value=0,
                                                       max_value=1),
        "N/G": st.column_config.NumberColumn(format="%.2f"), "Porosity": st.column_config.NumberColumn(format="%.3f"),
        "Sw": st.column_config.NumberColumn(format="%.2f")})
    st.caption("Reservoir = Plover top to Nome (or TD). Empty rows: no formation tops for that well yet.")

    well = well_picker(list(summary["well"]), "qc")
    row = summary[summary["well"] == well].iloc[0]
    m = st.columns(5)
    m[0].metric("Net sand", f"{row['net_sand_m']:.0f} m" if pd.notna(row.get("net_sand_m")) else "–")
    m[1].metric("Net-to-gross", f"{row['net_to_gross']:.2f}" if pd.notna(row.get("net_to_gross")) else "–")
    m[2].metric("Mean porosity", f"{row['mean_phi_net']:.3f}" if pd.notna(row.get("mean_phi_net")) else "–")
    m[3].metric("Flagged (zone)", f"{row['flagged_fraction_zone']:.1%}")
    m[4].metric("Max depth error", f"{row['tvdss_max_error_m']:.0f} m", help="Vertical-well assumption, at TD")
    image = QC_DIR / f"{well}.png"
    if image.is_file():
        st.image(str(image), width="stretch")
    meta_path = QC_LOGS / f"{well}.meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        with st.expander("Where every curve comes from (provenance)"):
            prov = pd.DataFrame(meta["provenance"]["curves"]).T.reset_index().rename(columns={"index": "curve"})
            st.dataframe(prov.astype(str), hide_index=True, width="stretch")
            st.caption(f"Source file: {meta['provenance']['file']}")
    step_nav("qc")


@st.fragment
def page_seismic():
    step_header(4, "Seismic",
                "Browse the near, mid and far stacks. The **map** on the right always shows where the section is. "
                "Showing all three stacks on one colour scale lets you see how amplitudes change with angle (AVO).",
                "the strong events around 3.1–3.3 s (top Plover reservoir) and how their brightness changes from near "
                "to far.")
    stacks, ready, geometry, positions = seismic()
    if not ready:
        st.warning("The seismic display cache is missing. Build it in step 1.")
        return
    ilines, xlines, samples = geometry["ilines"], geometry["xlines"], geometry["samples"]
    pos = positions.dropna(subset=["INLINE"])
    c = st.columns([1.3, 1.3, 1])
    view = c[0].segmented_control("View", ["Inline", "Crossline", "Time slice", "3D"], default="Inline", key="seis_view")
    view = view or "Inline"
    stack = c[1].segmented_control("Stack", ["near", "mid", "far", "all three"], default="near", key="seis_stack")
    stack = stack or "near"
    well = c[2].selectbox("Centre on well", list(pos["WELL_ID"]),
                          index=list(pos["WELL_ID"]).index(st.session_state["well"])
                          if st.session_state.get("well") in list(pos["WELL_ID"]) else 0, key="seis_well")
    st.session_state["well"] = well
    wrow = pos[pos["WELL_ID"] == well].iloc[0]
    with st.expander("Display settings"):
        s = st.columns(3)
        clip_pct = s[0].slider("Contrast (clip percentile)", 90.0, 99.9, 98.0, 0.1, key="seis_clip")
        colorscale = s[1].selectbox("Colours", ["RdBu_r", "Greys"], key="seis_cmap")
        proj = s[2].slider("Show wells within (lines)", 0, 200, 30, 5, key="seis_proj")
    names = ["near", "mid", "far"] if stack == "all three" else [stack]
    names = [n for n in names if n in ready]

    if view == "3D":
        render_3d(ready[names[0]], geometry, pos, samples, clip_pct, colorscale, int(wrow["INLINE"]),
                  int(wrow["CROSSLINE"]))
        step_nav("seismic")
        return
    clicked = _map_click_ilxl("seis_map", geometry)
    if clicked and view == "Inline":
        st.session_state[f"seis_il_{well}"] = clicked[0]
    elif clicked and view == "Crossline":
        st.session_state[f"seis_xl_{well}"] = clicked[1]
    if view == "Inline":
        kind, value = "inline", st.slider("Inline", int(ilines[0]), int(ilines[-1]), int(wrow["INLINE"]),
                                          key=f"seis_il_{well}")
    elif view == "Crossline":
        kind, value = "crossline", st.slider("Crossline", int(xlines[0]), int(xlines[-1]), int(wrow["CROSSLINE"]),
                                             key=f"seis_xl_{well}")
    else:
        kind, value = "time", float(st.select_slider("Time slice (ms)", options=[float(t) for t in samples],
                                                     value=3152.0, key="seis_t"))
    if kind != "time":
        tw = st.slider("Time window (ms)", float(samples[0]), float(samples[-1]), (2900.0, 3600.0), 4.0, key="seis_tw")
    sections = {n: section(ready, n, kind, value) for n in names}
    clip = max(float(np.percentile(np.abs(sections[names[0]]), clip_pct)), 1e-6)
    left, right = st.columns([3, 1.1])
    if kind == "time":
        fig = make_subplots(rows=1, cols=len(names), shared_yaxes=True, horizontal_spacing=0.03,
                            subplot_titles=[f"{n} stack" for n in names])
        for col, n in enumerate(names, 1):
            sx, sy = _step(len(xlines), 500), _step(len(ilines), 500)
            fig.add_trace(seismic_heatmap(sections[n][::sy, ::sx], xlines[::sx], ilines[::sy], clip, colorscale,
                                          col == len(names), "XL %{x}<br>IL %{y}<br>%{z:.0f}<extra></extra>"), 1, col)
            fig.add_trace(go.Scatter(x=pos["CROSSLINE"], y=pos["INLINE"], mode="markers+text", text=pos["WELL_ID"],
                                     textposition="top center", textfont={"size": 10, "color": "black"},
                                     marker={"size": 8, "color": "#ffd33d", "line": {"color": "black", "width": 1}},
                                     showlegend=False, hoverinfo="text"), 1, col)
            fig.update_xaxes(title_text="Crossline", row=1, col=col)
        fig.update_yaxes(title_text="Inline", row=1, col=1)
        style(fig, 560 if len(names) == 1 else 480, legend_below=False)
        lines = []
    else:
        lateral = xlines if kind == "inline" else ilines
        t_mask = (samples >= tw[0]) & (samples <= tw[1])
        along, across = ("CROSSLINE", "INLINE") if kind == "inline" else ("INLINE", "CROSSLINE")
        near = pos[(pos[across] - value).abs() <= proj]
        fig = make_subplots(rows=len(names), cols=1, shared_xaxes=True, vertical_spacing=0.06,
                            subplot_titles=[f"{n} stack" for n in names])
        for row, n in enumerate(names, 1):
            sx = _step(len(lateral), 900)
            fig.add_trace(seismic_heatmap(sections[n][::sx][:, t_mask].T, lateral[::sx], samples[t_mask], clip,
                                          colorscale, row == 1,
                                          f"{along.title()} %{{x}}<br>%{{y:.0f}} ms<br>%{{z:.0f}}<extra></extra>"),
                          row, 1)
            for _, wl in near.iterrows():
                fig.add_vline(x=float(wl[along]), line={"color": "black", "width": 2 if wl["WELL_ID"] == well else 1},
                              row=row, col=1)
                add_well_tops(fig, wl["WELL_ID"], float(wl[along]), float(tw[0]), float(tw[1]),
                              max(4.0, (lateral[-1] - lateral[0]) / 120), row=row, col=1)
                if row == 1:
                    fig.add_annotation(x=float(wl[along]), y=1.0, yref="y domain", yanchor="bottom", showarrow=False,
                                       text=wl["WELL_ID"], font={"size": 10}, bgcolor="rgba(255,255,255,.85)")
            fig.update_yaxes(autorange="reversed", title_text="TWT (ms)", row=row, col=1)
        fig.update_xaxes(title_text=along.title(), row=len(names), col=1)
        style(fig, 560 if len(names) == 1 else 300 * len(names) + 60, legend_below=False)
        i = int(np.searchsorted(ilines, value)) if kind == "inline" else None
        j = int(np.searchsorted(xlines, value)) if kind == "crossline" else None
        lines = [(geometry["cdp_x"][i, :], geometry["cdp_y"][i, :], f"Inline {value}")] if kind == "inline" else \
            [(geometry["cdp_x"][:, j], geometry["cdp_y"][:, j], f"Crossline {value}")]
    left.plotly_chart(fig, width="stretch", key="seis_fig")
    clickable = kind != "time"
    right.plotly_chart(basemap(geometry, pos, lines, highlight=kind == "time", clickable=clickable,
                               title=f"{view} {value:.0f}" + (" ms" if kind == "time" else "")),
                       width="stretch", key="seis_map", on_select="rerun", selection_mode="points")
    if clickable:
        right.caption(f"Click the map to choose the {kind}.")
    step_nav("seismic")


def _map_click_ilxl(key, geometry):
    """(inline, crossline) of the trace nearest to a new click on a clickable basemap, else None."""
    ev = st.session_state.get(key)
    try:
        pts = list(ev["selection"]["points"]) if ev else []
    except (KeyError, TypeError):
        pts = []
    sig = tuple((round(p.get("x", 0)), round(p.get("y", 0))) for p in pts)
    if not pts or sig == st.session_state.get(f"{key}_sig"):
        return None
    st.session_state[f"{key}_sig"] = sig
    d = (geometry["cdp_x"] - pts[-1]["x"]) ** 2 + (geometry["cdp_y"] - pts[-1]["y"]) ** 2
    i, j = np.unravel_index(np.argmin(d), d.shape)
    return int(geometry["ilines"][i]), int(geometry["xlines"][j])


def render_3d(path, geometry, pos, samples, clip_pct, colorscale, il0, xl0):
    ilines, xlines = geometry["ilines"], geometry["xlines"]
    c = st.columns(4)
    il = c[0].slider("Inline", int(ilines[0]), int(ilines[-1]), il0, key="d3_il")
    xl = c[1].slider("Crossline", int(xlines[0]), int(xlines[-1]), xl0, key="d3_xl")
    t_slice = float(c[2].select_slider("Time slice (ms)", options=[float(t) for t in samples], value=3400.0, key="d3_t"))
    vscale = c[3].slider("Vertical scale", 0.1, 2.0, 0.6, 0.05, key="d3_v")
    mt = _mtime(path)
    inl, xln, tsl = (_section(str(path), k, v, mt) for k, v in (("inline", il), ("crossline", xl), ("time", t_slice)))
    clip = max(float(np.percentile(np.abs(inl), clip_pct)), 1e-6)
    cx, cy = float(np.nanmean(geometry["cdp_x"])), float(np.nanmean(geometry["cdp_y"]))
    i, j = int(np.searchsorted(ilines, il)), int(np.searchsorted(xlines, xl))
    st_t = _step(len(samples), 160)
    tt = samples[::st_t]
    fig = go.Figure()
    sj, si = _step(len(xlines), 220), _step(len(ilines), 220)
    for xs, ys, vals in ((geometry["cdp_x"][i, ::sj], geometry["cdp_y"][i, ::sj], inl[::sj, ::st_t].T),
                         (geometry["cdp_x"][::si, j], geometry["cdp_y"][::si, j], xln[::si, ::st_t].T)):
        fig.add_trace(go.Surface(x=np.tile(xs - cx, (len(tt), 1)), y=np.tile(ys - cy, (len(tt), 1)),
                                 z=-np.tile(tt[:, None], (1, len(xs))), surfacecolor=vals, colorscale=colorscale,
                                 cmin=-clip, cmax=clip, showscale=False, hoverinfo="skip"))
    si2, sj2 = _step(len(ilines), 200), _step(len(xlines), 200)
    gx, gy = geometry["cdp_x"][::si2, ::sj2] - cx, geometry["cdp_y"][::si2, ::sj2] - cy
    fig.add_trace(go.Surface(x=gx, y=gy, z=np.full(gx.shape, -t_slice), surfacecolor=tsl[::si2, ::sj2],
                             colorscale=colorscale, cmin=-clip, cmax=clip, showscale=False, hoverinfo="skip"))
    for _, w in pos.iterrows():
        wi, wj = int(np.searchsorted(ilines, w["INLINE"])), int(np.searchsorted(xlines, w["CROSSLINE"]))
        x0, y0 = geometry["cdp_x"][wi, wj] - cx, geometry["cdp_y"][wi, wj] - cy
        selected = w["WELL_ID"] == st.session_state.get("well")
        fig.add_trace(go.Scatter3d(x=[x0, x0], y=[y0, y0], z=[-samples[0], -samples[-1]], mode="lines",
                                   line={"color": RED if selected else "#24292f", "width": 6}, name=w["WELL_ID"],
                                   hoverinfo="name", showlegend=False))
        fig.add_trace(go.Scatter3d(x=[x0], y=[y0], z=[-samples[0] + 30], mode="text", text=[w["WELL_ID"]],
                                   textfont={"size": 11, "color": "#24292f"}, showlegend=False, hoverinfo="skip"))
    dx = float(np.nanmax(geometry["cdp_x"]) - np.nanmin(geometry["cdp_x"]))
    dy = float(np.nanmax(geometry["cdp_y"]) - np.nanmin(geometry["cdp_y"]))
    mx = max(dx, dy)
    fig.update_layout(template="plotly_white", height=680, margin={"l": 0, "r": 0, "t": 10, "b": 0},
                      scene={"aspectmode": "manual", "aspectratio": {"x": dx / mx, "y": dy / mx, "z": vscale},
                             "xaxis_title": "East (m)", "yaxis_title": "North (m)", "zaxis_title": "-TWT (ms)",
                             "camera": {"eye": {"x": -1.3, "y": -1.5, "z": 0.9}}})
    left, right = st.columns([3, 1.1])
    left.plotly_chart(fig, width="stretch", key="seis_3d")
    right.plotly_chart(basemap(geometry, pos, [(geometry["cdp_x"][i, :], geometry["cdp_y"][i, :], "inline"),
                                               (geometry["cdp_x"][:, j], geometry["cdp_y"][:, j], "crossline")],
                               title=f"IL {il} · XL {xl} · {t_slice:.0f} ms"), width="stretch", key="seis_3d_map")
    st.caption("Drag to rotate, scroll to zoom. Coordinates are metres from the centre of the seismic area.")


def page_tie():
    step_header(5, "Well tie",
                "A **synthetic seismogram** is made from each well's logs (Vp, Vs, density → reflectivity at the near, "
                "mid and far angles → convolved with a wavelet) and compared with the real seismic at the well. The "
                "automated tie then finds the **time shift and stretch** that line them up, using the well's own "
                "time-depth curve or, if it has none, the nearest well's. Wavelets and amplitude scales are estimated "
                "on the training wells only; the two blind wells are kept out.",
                "a high correlation after the tie, a small shift, and the same events in the synthetic and the "
                "seismic. Wells marked ⚠️ need a manual look.")
    c = st.columns([1, 3])
    if c[0].button("Run automated well tie", type="primary", help="scripts/run_well_tie.py, 1–2 minutes"):
        run_script("run_well_tie.py", "automated well tie")
    c[1].caption("Parameters: `configs/well_tie.yaml`. Results: `outputs/well_ties/`.")
    summary = csv(TIE_DIR / "summary.csv")
    if summary is None:
        st.warning("The well tie has not been run yet (run well QC first).")
        return
    g2 = g2_status()
    cal = calibration()
    if g2 == "PASS":
        st.success(f"Gate G2 passed: the training wells tie with a mean correlation of "
                   + ", ".join(f"{k} {v['mean_corr_training']:.2f}" for k, v in cal["stacks"].items()) + ".")
    elif g2 == "FAIL":
        st.error("Gate G2 failed: ties are not good enough to calibrate the seismic.")

    tied = summary[summary["status"] == "tied"].copy()
    overview = pd.DataFrame({
        "Well": summary["well"],
        "Status": summary.apply(lambda r: "⛔ " + str(r["status"]).replace("stopped: ", "") if r["status"] != "tied"
                                else ("⚠️ review" if bool(r.get("needs_review")) else "✅ ok"), axis=1),
        "Role": summary.apply(lambda r: "blind" if str(r.get("held_out")) == "True" else
                              ("training" if r["status"] == "tied" else "-"), axis=1),
        "Time-depth from": summary.apply(lambda r: "-" if r["status"] != "tied" or pd.isna(r.get("td_source")) else
                                         (f"{r['td_source']} ({r['td_distance_km']:.1f} km)"
                                          if str(r.get("td_transferred")) == "True" else "own VSP"), axis=1),
        "Shift (ms)": pd.to_numeric(summary.get("shift_ms"), errors="coerce"),
        "Stretch (%)": pd.to_numeric(summary.get("stretch_pct"), errors="coerce"),
        "r before": pd.to_numeric(summary.get("corr_before_mean"), errors="coerce"),
        "r after": pd.to_numeric(summary.get("corr_after_mean"), errors="coerce"),
        "Why review": summary.get("qc_flags").fillna("").replace("ok", ""),
    })
    st.dataframe(overview, hide_index=True, width="stretch", column_config={
        "r before": st.column_config.NumberColumn(format="%.2f"),
        "r after": st.column_config.ProgressColumn("r after", format="%.2f", min_value=0, max_value=1)})

    well = well_picker(list(tied["well"]), "tie")
    r = tied[tied["well"] == well].iloc[0]
    data, meta = tie(well), tie_meta(well)
    if str(r.get("qc_flags", "ok")) != "ok":
        st.warning(f"**{well}** needs a manual look: {r['qc_flags']}.")
    m = st.columns(5)
    m[0].metric("Time-depth from", "own VSP" if str(r["td_transferred"]) != "True" else f"{r['td_source']}",
                help=None if str(r["td_transferred"]) != "True" else f"{r['td_distance_km']:.1f} km away")
    m[1].metric("Time shift", f"{r['shift_ms']:+.0f} ms", help=f"Allowed ±{r['max_shift_ms']:.0f} ms")
    m[2].metric("Stretch", f"{r['stretch_pct']:+.1f} %")
    m[3].metric("Correlation", f"{r['corr_after_mean']:.2f}", f"{r['corr_after_mean'] - r['corr_before_mean']:+.2f}")
    m[4].metric("Role", "blind" if str(r["held_out"]) == "True" else "training")
    tie_line_view(well, data, meta)
    st.markdown("##### Synthetic vs seismic at the well")
    st.plotly_chart(tie_figure(data, r), width="stretch", key=f"tie_fig_{well}")
    st.caption("White/black = real seismic at the well (3×3 trace average). Dashed grey = synthetic with the "
               "initial time-depth. Coloured = synthetic after the automated tie. Dotted lines = formation tops.")

    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("##### Time-depth: before and after")
        st.plotly_chart(td_figure(well, data), width="stretch", key=f"td_fig_{well}")
    with right:
        st.markdown("##### What the automated tie changed")
        stacks = [str(s) for s in data["stacks"]]
        st.dataframe(pd.DataFrame({
            "Stack": stacks,
            "r before": [r[f"corr_before_{s}"] for s in stacks],
            "r after": [r[f"corr_after_{s}"] for s in stacks],
            "Wavelet": [f"{meta['wavelets'][s]['freq_hz']:.0f} Hz, {meta['wavelets'][s]['phase_deg']:+.0f}°"
                        for s in stacks]}), hide_index=True, width="stretch",
            column_config={"r before": st.column_config.NumberColumn(format="%.2f"),
                           "r after": st.column_config.NumberColumn(format="%.2f")})
        tdm = meta["time_depth"]
        st.markdown(
            f"- **Time-depth:** {'own checkshot-calibrated curve' if not tdm['transferred'] else 'borrowed from ' + tdm['donor'] + f' ({tdm['distance_km']} km away)'}\n"
            f"- **Shift / stretch:** {r['shift_ms']:+.0f} ms (limit ±{r['max_shift_ms']:.0f}), {r['stretch_pct']:+.1f} %\n"
            f"- **Seismic used:** {meta['trace_selection']}\n"
            f"- **Logs:** {meta['log_conditioning']['interpolated_samples']} bad-hole/gap samples interpolated; "
            f"Vs estimated for {meta['log_conditioning']['vs_estimated_fraction']:.0%} of the interval")
        with st.expander("Technical details (raw)"):
            st.json(meta)
    with st.expander("Global calibration from the training wells"):
        if cal:
            st.dataframe(pd.DataFrame(cal["stacks"]).T.reset_index().rename(columns={"index": "stack"}),
                         hide_index=True, width="stretch")
            st.caption(f"Training: {', '.join(cal['training_wells'])}. Blind: {', '.join(cal['held_out_wells'])}. "
                       f"Polarity check (mean r): {cal['polarity_check_mean_corr']}.")
    step_nav("tie")


@st.fragment
def tie_line_view(well, data, meta):
    """Seismic line through the well with the tie switched on or off."""
    st.markdown("##### The tie on a seismic line")
    stacks, ready, geometry, positions = seismic()
    if not ready:
        st.info("Build the seismic display cache (step 1) to see the tie on a seismic line.")
        return
    pos = positions.dropna(subset=["INLINE"]).set_index("WELL_ID")
    if well not in pos.index:
        return
    samples, ilines, xlines = geometry["samples"], geometry["ilines"], geometry["xlines"]
    others = [w for w in pos.index if w != well]
    c = st.columns([1.6, 1.1, 1.3, 1.3])
    line = c[0].segmented_control("Line through the well", ["Inline", "Crossline", "To another well"],
                                  default="Inline", key=f"tl_line_{well}") or "Inline"
    stack = c[1].segmented_control("Stack", [s for s in ("near", "mid", "far") if s in ready], default="near",
                                   key=f"tl_stack_{well}") or "near"
    mode = c[2].segmented_control("Well tie", ["Off", "On", "Before | after"], default="Before | after",
                                  key=f"tl_mode_{well}") or "Before | after"
    other = c[3].selectbox("Other well", others, key=f"tl_other_{well}", disabled=line != "To another well")
    il, xl = float(pos.loc[well, "INLINE"]), float(pos.loc[well, "CROSSLINE"])
    half = 120
    if line == "Inline":
        sec = section(ready, stack, "inline", il)
        j0 = int(np.searchsorted(xlines, xl))
        lo_i, hi_i = max(j0 - half, 0), min(j0 + half, len(xlines) - 1)
        x_axis, traces, well_x = xlines[lo_i:hi_i + 1], sec[lo_i:hi_i + 1], float(xl)
        x_title = "Crossline"
        lines = [(geometry["cdp_x"][int(np.searchsorted(ilines, il)), lo_i:hi_i + 1],
                  geometry["cdp_y"][int(np.searchsorted(ilines, il)), lo_i:hi_i + 1], "line")]
    elif line == "Crossline":
        sec = section(ready, stack, "crossline", xl)
        i0 = int(np.searchsorted(ilines, il))
        lo_i, hi_i = max(i0 - half, 0), min(i0 + half, len(ilines) - 1)
        x_axis, traces, well_x = ilines[lo_i:hi_i + 1], sec[lo_i:hi_i + 1], float(il)
        x_title = "Inline"
        lines = [(geometry["cdp_x"][lo_i:hi_i + 1, int(np.searchsorted(xlines, xl))],
                  geometry["cdp_y"][lo_i:hi_i + 1, int(np.searchsorted(xlines, xl))], "line")]
    else:
        il2, xl2 = float(pos.loc[other, "INLINE"]), float(pos.loc[other, "CROSSLINE"])
        back = (float(np.clip(il - 0.25 * (il2 - il), ilines[0], ilines[-1])),
                float(np.clip(xl - 0.25 * (xl2 - xl), xlines[0], xlines[-1])))
        points = (back, (il, xl), (il2, xl2))
        route, traces = _traverse(str(ready[stack]), _mtime(ready[stack]), points)
        x_axis = route["distance_m"] / 1000
        well_x, x_title = float(x_axis[route["vertex"][1]]), f"Distance along the line towards {other} (km)"
        lines = [(route["x"], route["y"], "line")]

    tie_meta_ = meta["auto_tie"]
    shift, stretch, tc = tie_meta_["shift_s"] * 1000, tie_meta_["stretch"], tie_meta_["tc"] * 1000
    tops_after = np.asarray(data["tops_twt_ms"], dtype=float)
    tops_before = (tops_after - shift - tc) / (1 + stretch) + tc
    t_lo = max(float(min(data["log_twt_initial_ms"].min(), data["log_twt_ms"].min())) - 120, float(samples[0]))
    t_hi = min(float(max(data["log_twt_initial_ms"].max(), data["log_twt_ms"].max())) + 120, float(samples[-1]))
    t_mask = (samples >= t_lo) & (samples <= t_hi)
    clip = max(float(np.percentile(np.abs(traces[:, t_mask]), 98)), 1e-6)
    k = [str(s) for s in data["stacks"]].index(stack)
    span = float(x_axis[-1] - x_axis[0]) if len(x_axis) > 1 else 1.0
    width = 0.06 * abs(span)
    states = {"Off": [False], "On": [True], "Before | after": [False, True]}[mode]
    titles = ["Without well tie (initial time-depth)" if not s else "With automated well tie" for s in states]
    fig = make_subplots(rows=1, cols=len(states), shared_yaxes=True, horizontal_spacing=0.04, subplot_titles=titles)
    ai = data["ai"]
    ai_norm = (ai - np.nanmean(ai)) / (np.nanstd(ai) or 1.0)
    for col, applied in enumerate(states, 1):
        fig.add_trace(seismic_heatmap(traces[:, t_mask].T, x_axis, samples[t_mask], clip, "RdBu_r", col == len(states),
                                      f"{x_title.split(' (')[0]} %{{x}}<br>%{{y:.0f}} ms<br>%{{z:.0f}}<extra></extra>"),
                      1, col)
        log_t = data["log_twt_ms"] if applied else data["log_twt_initial_ms"]
        fig.add_trace(go.Scatter(x=well_x - 3.2 * width + 0.7 * width * np.clip(ai_norm, -2, 2), y=log_t,
                                 mode="lines", line={"color": "#116329", "width": 1.6},
                                 name="Acoustic impedance log (left of well)", showlegend=col == 1,
                                 hoverinfo="skip"), 1, col)
        syn = (data["syn_after"][k] if applied else data["syn_before"][k]) * data["scales"][k]
        win = data["win_after"] if applied else data["win_before"]
        syn_x = well_x + 1.2 * width + width * np.clip(syn[win] / clip, -1.5, 1.5)
        fig.add_trace(go.Scatter(x=syn_x, y=data["t_ms"][win], mode="lines", line={"color": "black", "width": 5},
                                 showlegend=False, hoverinfo="skip"), 1, col)
        fig.add_trace(go.Scatter(x=syn_x, y=data["t_ms"][win], mode="lines", line={"color": "#ffd33d", "width": 2.5},
                                 name="Synthetic (right of well)", showlegend=col == 1, hoverinfo="skip"), 1, col)
        fig.add_vline(x=well_x, line={"color": "black", "width": 1.5}, row=1, col=col)
        tt = tops_after if applied else tops_before
        for n_top, (name, t) in enumerate(zip(data["tops_names"], tt)):
            if t_lo <= t <= t_hi:
                fig.add_trace(go.Scatter(x=[well_x - 4.2 * width, well_x + 3.2 * width], y=[t, t], mode="lines",
                                         line={"color": "#24292f", "width": 2.5},
                                         name="Formation tops (hover for name)",
                                         legendgroup="tops", showlegend=col == 1 and n_top == 0,
                                         hovertemplate=f"{name}<br>{t:.0f} ms<extra></extra>"), 1, col)
        fig.update_xaxes(title_text=x_title, row=1, col=col)
    fig.update_yaxes(autorange="reversed", title_text="TWT (ms)", row=1, col=1)
    style(fig, 600)
    left, right = st.columns([3, 1])
    left.plotly_chart(fig, width="stretch", key=f"tl_fig_{well}")
    right.plotly_chart(basemap(geometry, positions.dropna(subset=["INLINE"]), lines, height=330,
                               title=f"{line} through {well}"), width="stretch", key=f"tl_map_{well}")
    right.markdown(f"**The tie moved this well by {shift:+.0f} ms** and stretched it {stretch * 100:+.1f} %. "
                   "Compare where the yellow synthetic and the black top markers sit relative to the seismic "
                   "events with the tie off and on.")


def tie_figure(data, r, height=680):
    t = data["t_ms"]
    stacks = [str(s) for s in data["stacks"]]
    lo = max(float(data["log_twt_ms"].min()) - 60, float(t[0]))
    hi = min(float(data["log_twt_ms"].max()) + 40, float(t[-1]))
    fig = make_subplots(rows=1, cols=3 + len(stacks), shared_yaxes=True, horizontal_spacing=0.015,
                        column_widths=[0.8, 0.8] + [1] * len(stacks) + [0.7],
                        subplot_titles=["Acoustic imp.", "Vp/Vs"] +
                        [f"{s}  (r {r[f'corr_before_{s}']:.2f} → {r[f'corr_after_{s}']:.2f})" for s in stacks] + ["Tops"])
    fig.add_trace(go.Scatter(x=data["ai"], y=data["log_twt_ms"], line={"color": "#bf8700", "width": 1},
                             showlegend=False, hoverinfo="skip"), 1, 1)
    fig.add_trace(go.Scatter(x=data["vpvs"], y=data["log_twt_ms"], line={"color": "#0a7b83", "width": 1},
                             showlegend=False, hoverinfo="skip"), 1, 2)
    win = data["win_after"]
    for k, s in enumerate(stacks):
        obs = data["obs"][k] / data["scales"][k]
        norm = float(np.abs(obs[win]).max()) or 1.0
        fig.add_trace(go.Scatter(x=obs / norm, y=t, line={"color": "#24292f", "width": 1.6}, name="Seismic",
                                 showlegend=k == 0), 1, 3 + k)
        fig.add_trace(go.Scatter(x=np.where(data["win_before"], data["syn_before"][k], np.nan) / norm, y=t,
                                 line={"color": "#8c959f", "width": 1.2, "dash": "dash"}, name="Synthetic before",
                                 showlegend=k == 0), 1, 3 + k)
        fig.add_trace(go.Scatter(x=np.where(win, data["syn_after"][k], np.nan) / norm, y=t,
                                 line={"color": STACK_COLORS.get(s, RED), "width": 2.4}, name="Synthetic after tie",
                                 showlegend=k == 0), 1, 3 + k)
        fig.update_xaxes(range=[-1.4, 1.4], showticklabels=False, row=1, col=3 + k)
    names = [(float(tt), str(n)) for n, tt in zip(data["tops_names"], data["tops_twt_ms"]) if lo <= tt <= hi]
    names.sort()
    n_col = 3 + len(stacks)
    label_y = spaced_labels([tt for tt, _ in names], (hi - lo) / 30)
    for (tt, name), y in zip(names, label_y):
        for col in range(1, n_col):
            fig.add_hline(y=tt, line={"color": "#57606a", "width": 0.8, "dash": "dot"}, row=1, col=col)
        fig.add_trace(go.Scatter(x=[0, 0.1], y=[tt, y], mode="lines", line={"color": "#8c959f", "width": 0.8},
                                 showlegend=False, hoverinfo="skip"), 1, n_col)
        fig.add_trace(go.Scatter(x=[0.12], y=[y], mode="text", text=[name.replace(" Formation", "")],
                                 textposition="middle right", textfont={"size": 10}, showlegend=False,
                                 hoverinfo="skip"), 1, n_col)
    fig.update_xaxes(range=[0, 1], showticklabels=False, showgrid=False, zeroline=False, row=1, col=n_col)
    fig.update_yaxes(range=[hi, lo], title_text="TWT (ms)", row=1, col=1)
    style(fig, height)
    fig.update_annotations(font_size=11)
    return fig


def td_figure(well, data):
    td = csv(TD_DIR / f"{well}.csv")
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, column_widths=[1.4, 1], horizontal_spacing=0.05,
                        subplot_titles=["TWT vs depth", "Change (ms)"])
    if td is not None and "twt_final_ms" in td:
        td = td.dropna(subset=["twt_initial_ms", "twt_final_ms"])
        lo, hi = float(data["log_twt_ms"].min()) - 150, float(data["log_twt_ms"].max()) + 100
        td = td[td["twt_final_ms"].between(lo, hi)]
        fig.add_trace(go.Scatter(x=td["twt_initial_ms"], y=td["tvdss"], name="Before",
                                 line={"color": "#8c959f", "dash": "dash"}), 1, 1)
        fig.add_trace(go.Scatter(x=td["twt_final_ms"], y=td["tvdss"], name="After tie", line={"color": RED}), 1, 1)
        fig.add_trace(go.Scatter(x=td["twt_final_ms"] - td["twt_initial_ms"], y=td["tvdss"], showlegend=False,
                                 line={"color": RED}), 1, 2)
    fig.update_yaxes(autorange="reversed", title_text="TVDSS (m)", row=1, col=1)
    fig.update_xaxes(title_text="TWT (ms)", row=1, col=1)
    style(fig, 400)
    return fig


@st.fragment
def page_traverse():
    step_header(6, "Seismic through the wells",
                "A line through several wells shows how the tied wells fit the seismic between them. Yellow traces "
                "are the tied synthetics at each well; dotted lines join the same formation top from well to well.",
                "synthetics that match the seismic at every well, and tops that follow the same seismic event "
                "between wells.")
    stacks, ready, geometry, positions = seismic()
    if not ready:
        st.warning("The seismic display cache is missing. Build it in step 1.")
        return
    pos = positions.dropna(subset=["INLINE"])
    samples = geometry["samples"]
    choices = list(pos["WELL_ID"])
    chosen = st.multiselect("Wells along the line, in order", choices,
                            default=[w for w in DEFAULT_TRAVERSE if w in choices], key="trav_wells")
    if len(chosen) < 2:
        st.info("Choose at least two wells.")
        return
    c = st.columns([1.3, 2, 1, 1])
    stack = c[0].segmented_control("Stack", ["near", "mid", "far", "all three"], default="near", key="trav_stack") or "near"
    tw = c[1].slider("Time window (ms)", float(samples[0]), float(samples[-1]), (2950.0, 3450.0), 4.0, key="trav_tw")
    show_syn = c[2].toggle("Synthetics", value=True, key="trav_syn")
    show_tops = c[3].toggle("Tops", value=True, key="trav_tops")
    with st.expander("Display settings"):
        s = st.columns(2)
        clip_pct = s[0].slider("Contrast (clip percentile)", 90.0, 99.9, 98.0, 0.1, key="trav_clip")
        colorscale = s[1].selectbox("Colours", ["RdBu_r", "Greys"], key="trav_cmap")
    names = ["near", "mid", "far"] if stack == "all three" else [stack]
    rows = pos.set_index("WELL_ID").loc[chosen]
    points = tuple((float(r.INLINE), float(r.CROSSLINE)) for r in rows.itertuples())
    t_mask = (samples >= tw[0]) & (samples <= tw[1])
    data = {n: _traverse(str(ready[n]), _mtime(ready[n]), points) for n in names}
    route = data[names[0]][0]
    dist = route["distance_m"] / 1000
    well_x = dist[route["vertex"]]
    clip = max(float(np.percentile(np.abs(data[names[0]][1][:, t_mask]), clip_pct)), 1e-6)
    ties = {w: tie(w) for w in chosen}
    fig = make_subplots(rows=len(names), cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        subplot_titles=[f"{n} stack" for n in names])
    palette = ["#1f6feb", "#cf222e", "#2da44e", "#8250df", "#bf8700", "#0a7b83", "#e8833a", "#57606a", "#d63384"]
    for row, n in enumerate(names, 1):
        traces = data[n][1]
        sx = _step(len(dist), 1000)
        fig.add_trace(seismic_heatmap(traces[::sx][:, t_mask].T, dist[::sx], samples[t_mask], clip, colorscale,
                                      row == 1, "%{x:.2f} km<br>%{y:.0f} ms<br>%{z:.0f}<extra></extra>"), row, 1)
        k = ["near", "mid", "far"].index(n)
        width = 0.035 * float(dist[-1])
        surfaces = {}
        for w, x in zip(chosen, well_x):
            fig.add_vline(x=float(x), line={"color": "black", "width": 1}, row=row, col=1)
            if row == 1:
                fig.add_annotation(x=float(x), y=1.0, yref="y domain", yanchor="bottom", showarrow=False, text=w,
                                   font={"size": 11}, bgcolor="rgba(255,255,255,.9)")
            d = ties.get(w)
            if d is None:
                continue
            if show_syn:
                win = d["win_after"] & (d["t_ms"] >= tw[0]) & (d["t_ms"] <= tw[1])
                syn = d["syn_after"][k] * d["scales"][k]
                fig.add_trace(go.Scatter(x=x + width * np.clip(syn[win] / clip, -1.5, 1.5), y=d["t_ms"][win],
                                         mode="lines", line={"color": "#ffd33d", "width": 2.5},
                                         name="Synthetic (tied)", showlegend=False, hoverinfo="skip"), row, 1)
            if show_tops:
                for s_name, tt in zip(d["tops_names"], d["tops_twt_ms"]):
                    if tw[0] <= tt <= tw[1]:
                        surfaces.setdefault(str(s_name), []).append((float(x), float(tt)))
        for idx, (surface, pts) in enumerate(sorted(surfaces.items(), key=lambda kv: min(p[1] for p in kv[1]))):
            pts.sort()
            fig.add_trace(go.Scatter(x=[p[0] for p in pts], y=[p[1] for p in pts], mode="lines+markers",
                                     line={"color": palette[idx % len(palette)], "width": 2, "dash": "dot"},
                                     marker={"size": 7}, name=surface, showlegend=row == 1,
                                     hovertemplate=f"{surface}<br>%{{y:.0f}} ms<extra></extra>"), row, 1)
        fig.update_yaxes(autorange="reversed", title_text="TWT (ms)", row=row, col=1)
    fig.update_xaxes(title_text="Distance along the line (km)", row=len(names), col=1)
    style(fig, 520 if len(names) == 1 else 330 * len(names) + 80)
    left, right = st.columns([3, 1.1])
    left.plotly_chart(fig, width="stretch", key="trav_fig")
    right.plotly_chart(basemap(geometry, pos, [(route["x"], route["y"], "line")], title="The line on the map"),
                       width="stretch", key="trav_map")
    untied = [w for w in chosen if ties.get(w) is None]
    if untied:
        st.caption(f"No tie available for: {', '.join(untied)}.")
    step_nav("traverse")


def page_rockphysics():
    step_header(7, "Rock physics: what makes rocks look different",
                "Every log sample from the Plover reservoir is plotted as **acoustic impedance** against **Vp/Vs**. "
                "Porous sand moves left, shale moves up, gas moves down-left. Where the rock types overlap, the "
                "seismic cannot tell them apart - that overlap is the uncertainty the AVO classification must report.",
                "whether high-porosity sands (orange) separate from shale (grey) and tight rock (purple).")
    w = wells()
    frames = [qc_logs(n) for n in w["WELL_ID"]]
    frames = [f for f in frames if f is not None]
    if not frames:
        st.warning("Run the well QC (step 3) first.")
        return
    d = pd.concat(frames, ignore_index=True)
    d = d[d["formation"].astype(str).str.contains("Plover", na=False)].copy()
    d["ai"], d["vpvs"] = d["vp"] * d["rho"], d["vp"] / d["vs"]
    d = d.dropna(subset=["ai", "vpvs"])
    c = st.columns([2, 1])
    chosen = c[0].multiselect("Wells", sorted(d["well"].unique()), default=sorted(d["well"].unique()), key="rp_wells")
    color_by = c[1].segmented_control("Colour by", ["Rock type", "Well", "Gas indicator"], default="Rock type",
                                      key="rp_color") or "Rock type"
    d = d[d["well"].isin(chosen) & ~d["flag_bad"]]
    fig = go.Figure()
    if color_by == "Rock type":
        groups = [(LITHO_LABELS[k], d[d["lithotype"] == k], LITHO_COLORS[k]) for k in LITHO_COLORS if k != "unknown"]
    elif color_by == "Well":
        cols = ["#1f6feb", "#cf222e", "#2da44e", "#8250df", "#bf8700", "#0a7b83", "#e8833a", "#57606a"]
        groups = [(n, d[d["well"] == n], cols[i % len(cols)]) for i, n in enumerate(sorted(d["well"].unique()))]
    else:
        groups = [("Gas indicated", d[d["hc_indicator"] == "gas_indicated"], RED),
                  ("Possible", d[d["hc_indicator"] == "possible"], AMBER),
                  ("None / not sand", d[d["hc_indicator"].isin(["none", "not_applicable"])], "#a3a8ae")]
    for label, g, color in groups:
        if len(g):
            fig.add_trace(go.Scatter(x=g["ai"], y=g["vpvs"], mode="markers", name=f"{label} ({len(g):,})",
                                       marker={"size": 4, "color": color, "opacity": 0.6,
                                               "line": {"width": 0.3, "color": "#57606a"}},
                                       customdata=np.c_[g["well"], g["tvdss"].round(0)],
                                       hovertemplate="%{customdata[0]}<br>%{customdata[1]} m<br>AI %{x:.0f}<br>"
                                                     "Vp/Vs %{y:.2f}<extra></extra>"))
    fig.update_xaxes(title_text="Acoustic impedance (m/s · g/cc)")
    fig.update_yaxes(title_text="Vp/Vs", range=[1.35, 2.4])
    style(fig, 620)
    st.plotly_chart(fig, width="stretch", key="rp_fig")
    st.caption(f"{len(d):,} good-hole samples from the Plover Formation. Rock types are the petrophysical scenario "
               "from step 3.")
    step_nav("rockphysics")


def page_next():
    step_header(8, "AVO classification (next)",
                "With calibrated wells and seismic, the remaining steps build the actual prediction. They reuse the "
                "workflow that was already tested end-to-end on a synthetic earth.")
    items = [
        ("4 Fluid sensitivity", "Gassmann: does gas vs brine change near/mid/far amplitudes more than the tie noise? "
                                "Needs a gas fluid model at Plover pressure and temperature (gate G3)."),
        ("5 Scenarios", "Thousands of pseudo-wells: porosity, thickness, clay, fluid varied around the QC'd wells."),
        ("6 Forward modelling", "Synthetic near/mid/far for every scenario with the wavelets and scales from step 5."),
        ("7 Classification", "Methods A (intercept-gradient), B (hand-picked scenarios), C (learned from wells)."),
        ("8 Blind test", "Score once on Poseidon-2 and Boreas-1, which were kept out of everything (gate G4)."),
        ("9 3D cubes", "Rock-type and fluid probabilities plus uncertainty for every trace in the reservoir window."),
    ]
    for title, text in items:
        st.markdown(f"**{title}** – {text}")
    st.info("Open items before starting: review the ⚠️ ties in step 5 (Poseidon-North-1, Proteus-1, Pharos-1) and "
            "pick formation tops for Pharos-1 and Poseidon-North-1.")
    step_nav("next")


# ----------------------------------------------------------------------------------------------- step 9
SIM_DIR = ROOT / "outputs" / "simulation"
SIM_FACIES = ["shale", "tight", "volcanic", "sand_low_phi", "sand_high_phi"]
SIM_TRACKS = [("phi", "Porosity (v/v)"), ("vsh", "Vsh (v/v)"), ("sw", "Sw (sands)"), ("vp", "Vp (m/s)"),
              ("vs", "Vs (m/s)"), ("rho", "Density (g/cc)"), ("ai", "AI"), ("vpvs", "Vp/Vs")]


@st.cache_resource(show_spinner=False)
def _sim_prior(cfg_mtime: int, logs_mtime: int, tie_mtime: int):
    from avo import prospect_sim as ps
    del cfg_mtime, logs_mtime, tie_mtime
    cfg, calib, petro = ps.load_configs(ROOT)
    return ps.prepare_prior(ROOT, cfg, calib, petro["held_out_wells"])


def sim_prior():
    logs_m = max((_mtime(p) for p in QC_LOGS.glob("*.parquet")), default=0)
    return _sim_prior(_mtime(ROOT / "configs" / "prospect_sim.yaml"), logs_m, _mtime(CAL_PATH))


@st.cache_data(show_spinner=False, max_entries=8)
def _sim_result(path: str, mtime: int):
    del mtime
    d = np.load(path)
    return {k: d[k] for k in d.files}


def sim_runs() -> list[Path]:
    if not SIM_DIR.is_dir():
        return []
    runs = [p for p in SIM_DIR.glob("*.json") if p.stem != "prior" and p.with_suffix(".npz").is_file()]
    return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)


@st.cache_data(show_spinner=False)
def _trace_at(il: int, xl: int, mtime: int):
    from avo.welltie import well_traces
    del mtime
    cache = ROOT / "data" / "poseidon" / "seismic" / "cache"
    return well_traces(cache, ["near", "mid", "far"], il, xl, 3)


def _selection_points(key):
    ev = st.session_state.get(key)
    try:
        return list(ev["selection"]["points"]) if ev else []
    except (KeyError, TypeError):
        return []


def _vertex(pos, il, xl, snap=40):
    """Line vertex at (il, xl), snapped to a well within snap traces."""
    d = np.hypot(pos["INLINE"] - il, pos["CROSSLINE"] - xl)
    if len(d) and d.min() <= snap:
        r = pos.loc[d.idxmin()]
        return {"label": r["WELL_ID"], "il": int(r["INLINE"]), "xl": int(r["CROSSLINE"])}
    return {"label": f"IL {il}/XL {xl}", "il": int(il), "xl": int(xl)}


def _consume_map_click(geometry, pos):
    pts = _selection_points("sim_map")
    sig = tuple((round(p.get("x", 0)), round(p.get("y", 0))) for p in pts)
    if not pts or sig == st.session_state.get("sim_map_sig"):
        return
    st.session_state["sim_map_sig"] = sig
    x, y = pts[-1]["x"], pts[-1]["y"]
    d = (geometry["cdp_x"] - x) ** 2 + (geometry["cdp_y"] - y) ** 2
    i, j = np.unravel_index(np.argmin(d), d.shape)
    il, xl = int(geometry["ilines"][i]), int(geometry["xlines"][j])
    if st.session_state.get("sim_click_mode") == "Add line vertex":
        st.session_state["sim_vertices"] = list(st.session_state.get("sim_vertices") or []) + [_vertex(pos, il, xl)]
        st.session_state["sim_mode"] = "Composite line"
    else:
        st.session_state["sim_il"], st.session_state["sim_xl"] = il, xl


def _route_points(geometry, mode, il, xl, vertices):
    ilines, xlines = geometry["ilines"], geometry["xlines"]
    if mode == "Inline":
        return ((float(il), float(xlines[0])), (float(il), float(xlines[-1])))
    if mode == "Crossline":
        return ((float(ilines[0]), float(xl)), (float(ilines[-1]), float(xl)))
    return tuple((float(v["il"]), float(v["xl"])) for v in vertices)


def _route_wells(route, pos, max_traces=3):
    """Wells within max_traces of the route: name -> distance along the route (km)."""
    out = {}
    for r in pos.itertuples():
        d = np.hypot(route["inline"] - r.INLINE, route["crossline"] - r.CROSSLINE)
        k = int(np.argmin(d))
        if d[k] <= max_traces:
            out[r.WELL_ID] = float(route["distance_m"][k]) / 1000
    return out


@st.cache_data(show_spinner=False)
def _zone_wells(cfg_mtime: int):
    from avo import prospect_sim as ps
    from avo.prospect_sim3d import zone_twt_from_logs
    del cfg_mtime
    cfg, _, petro = ps.load_configs(ROOT)
    zone = cfg["training"].get("zone_formation_prefix")
    skip = set(petro["held_out_wells"]) | set(cfg["training"].get("exclude_wells") or [])
    out = {}
    if zone:
        for w in wells()["WELL_ID"]:
            z = zone_twt_from_logs(ROOT, w, zone) if w not in skip else None
            if z:
                out[w] = z
    return zone, out


def zone_wells():
    """(zone formation, {well: (top TWT ms, base TWT ms)}) for the wells the simulation uses."""
    return _zone_wells(_mtime(ROOT / "configs" / "prospect_sim.yaml"))


def _zone_defaults(route_wells, zw):
    top = sorted([d, zw[w][0]] for w, d in route_wells.items() if w in zw)
    base = sorted([d, zw[w][1]] for w, d in route_wells.items() if w in zw)
    return top, base


def _zone_line(picks, dist_km):
    if not picks:
        return None
    p = sorted(picks)
    return np.interp(dist_km, [a for a, _ in p], [b for _, b in p])


def _consume_route_click(dist_km, route):
    pts = [p for p in _selection_points("sim_sec") if p.get("x") is not None and p.get("y") is not None]
    sig = tuple((round(float(p["x"]), 3), round(float(p["y"]))) for p in pts)
    if not pts or sig == st.session_state.get("sim_sec_sig"):
        return
    st.session_state["sim_sec_sig"] = sig
    x, y = float(pts[-1]["x"]), round(float(pts[-1]["y"]) / 4) * 4
    mode = st.session_state.get("sim_pick_mode") or "Top"
    if mode == "Pseudo-well":
        k = int(np.argmin(np.abs(dist_km - x)))
        st.session_state["sim_il"], st.session_state["sim_xl"] = int(route["inline"][k]), int(route["crossline"][k])
        return
    key = "sim_top" if mode == "Top" else "sim_base"
    tol = max(0.01 * float(dist_km[-1]), 0.05)
    keep = [p for p in (st.session_state.get(key) or []) if abs(p[0] - x) > tol]
    st.session_state[key] = sorted(keep + [[x, float(y)]])
    st.session_state[f"{key}_last"] = x


def sim_route_map(geometry, pos, route, il, xl, vertices, mode):
    ox, oy = survey_outline(geometry)
    step = 24
    gx, gy = geometry["cdp_x"][::step, ::step].ravel(), geometry["cdp_y"][::step, ::step].ravel()
    ii, jj = np.meshgrid(geometry["ilines"][::step], geometry["xlines"][::step], indexing="ij")
    fig = go.Figure(go.Scatter(x=ox, y=oy, mode="lines", fill="toself", fillcolor="rgba(31,111,235,0.06)",
                               line={"color": BLUE, "width": 1.5}, hoverinfo="skip", name="Seismic area"))
    fig.add_trace(go.Scatter(x=gx, y=gy, mode="markers", name="click",
                             marker={"size": 9, "color": "rgba(31,111,235,0.07)"},
                             customdata=np.c_[ii.ravel(), jj.ravel()],
                             hovertemplate="IL %{customdata[0]} / XL %{customdata[1]}<extra>click</extra>"))
    if route is not None:
        fig.add_trace(go.Scatter(x=route["x"], y=route["y"], mode="lines", line={"color": RED, "width": 3},
                                 hoverinfo="skip", name="Section"))
    if mode == "Composite line" and vertices:
        i = [int(np.searchsorted(geometry["ilines"], v["il"])) for v in vertices]
        j = [int(np.searchsorted(geometry["xlines"], v["xl"])) for v in vertices]
        fig.add_trace(go.Scatter(x=geometry["cdp_x"][i, j], y=geometry["cdp_y"][i, j], mode="markers",
                                 marker={"size": 11, "color": "white", "line": {"color": RED, "width": 2}},
                                 hovertemplate="vertex<extra></extra>", name="Vertices"))
    held = set((calibration() or {}).get("held_out_wells", []))
    fig.add_trace(go.Scatter(x=pos["X_M"], y=pos["Y_M"], mode="markers+text", textposition="top center",
                             text=[w + (" (blind)" if w in held else "") for w in pos["WELL_ID"]],
                             textfont={"size": 10}, name="Wells", hovertemplate="%{text}<extra></extra>",
                             marker={"size": 9, "color": ["white" if w in held else "#24292f" for w in pos["WELL_ID"]],
                                     "symbol": ["diamond" if w in held else "circle" for w in pos["WELL_ID"]],
                                     "line": {"color": "#24292f", "width": 1.5}}))
    i, j = int(np.searchsorted(geometry["ilines"], il)), int(np.searchsorted(geometry["xlines"], xl))
    fig.add_trace(go.Scatter(x=[geometry["cdp_x"][i, j]], y=[geometry["cdp_y"][i, j]], mode="markers",
                             marker={"symbol": "star", "size": 20, "color": "#ffd33d",
                                     "line": {"color": "black", "width": 1.5}},
                             name="Pseudo-well", hovertemplate=f"Pseudo-well IL {il} / XL {xl}<extra></extra>"))
    pad = 1500
    fig.update_xaxes(range=[ox.min() - pad, ox.max() + pad], showticklabels=False)
    fig.update_yaxes(range=[oy.min() - pad, oy.max() + pad], showticklabels=False, scaleanchor="x")
    style(fig, 470, legend_below=False, showlegend=False, title={"text": "Map", "font": {"size": 13}})
    fig.update_layout(margin={"l": 10, "r": 10, "t": 40, "b": 10}, clickmode="event+select", dragmode="pan")
    return fig


def sim_route_fig(route, traces, samples, tw, rwells, held, zone_lines, picks, point_d, t0, t1, stack):
    dist = route["distance_m"] / 1000
    tm = (samples >= tw[0]) & (samples <= tw[1])
    sx = _step(len(dist), 900)
    clip = max(float(np.percentile(np.abs(traces[::sx][:, tm]), 98)), 1e-6)
    fig = go.Figure(seismic_heatmap(traces[::sx][:, tm].T, dist[::sx], samples[tm], clip, "RdBu_r", False, None))
    fig.data[0].hoverinfo = "skip"
    fig.data[0].hovertemplate = None
    # invisible clickable grid (heatmaps cannot deliver click events)
    gx, gy = np.meshgrid(dist[::max(1, len(dist) // 150)], samples[tm][::2])
    fig.add_trace(go.Scatter(x=gx.ravel(), y=gy.ravel(), mode="markers", name="click grid",
                             marker={"size": 12, "color": "rgba(0,0,0,0.003)"},
                             unselected={"marker": {"opacity": 1}}, selected={"marker": {"opacity": 1}},
                             hovertemplate="%{x:.2f} km · %{y:.0f} ms<extra>click</extra>"))
    top_line, base_line = zone_lines
    if top_line is not None and base_line is not None:
        fig.add_trace(go.Scatter(x=np.r_[dist, dist[::-1]], y=np.r_[top_line, base_line[::-1]], fill="toself",
                                 fillcolor="rgba(255,211,61,0.22)", line={"width": 0}, hoverinfo="skip"))
    for line, color, name in ((top_line, RED, "Zone top"), (base_line, BLUE, "Zone base")):
        if line is not None:
            fig.add_trace(go.Scatter(x=dist, y=line, mode="lines", line={"color": color, "width": 2},
                                     hoverinfo="skip", name=name))
    for key, color, symbol in (("top", RED, "triangle-down"), ("base", BLUE, "triangle-up")):
        p = picks[key]
        if p:
            fig.add_trace(go.Scatter(x=[a for a, _ in p], y=[b for _, b in p], mode="markers",
                                     marker={"size": 11, "color": color, "symbol": symbol,
                                             "line": {"color": "white", "width": 1}},
                                     hovertemplate=f"{key} pick %{{x:.2f}} km · %{{y:.0f}} ms<extra></extra>"))
    hw = 0.012 * float(dist[-1])
    for w, d in rwells.items():
        fig.add_vline(x=d, line={"color": "#24292f", "width": 1, "dash": "dot" if w in held else "solid"})
        fig.add_annotation(x=d, y=1.0, yref="y domain", yanchor="bottom", showarrow=False,
                           text=w + (" (blind – tops hidden)" if w in held else ""), font={"size": 10},
                           bgcolor="rgba(255,255,255,.85)")
        if w not in held:
            add_well_tops(fig, w, d, tw[0], tw[1], hw)
    fig.add_vline(x=point_d, line={"color": "#bf8700", "width": 2, "dash": "dash"})
    if t0 is not None:
        fig.add_shape(type="rect", x0=point_d - hw / 2, x1=point_d + hw / 2, y0=t0, y1=t1,
                      line={"color": "black", "width": 2}, fillcolor="rgba(255,211,61,0.55)")
    fig.update_yaxes(autorange="reversed", title_text="TWT (ms)", range=[tw[1], tw[0]])
    fig.update_xaxes(title_text="Distance along the section (km)", range=[float(dist[0]), float(dist[-1])])
    style(fig, 520, legend_below=False, showlegend=False,
          title={"text": f"Section ({stack} stack) · yellow: reservoir zone · dashed: pseudo-well",
                 "font": {"size": 13}})
    fig.update_layout(clickmode="event+select", dragmode="pan", hovermode="closest", hoverdistance=30)
    return fig



SIM_LINE_DIR = ROOT / "outputs" / "simulation_lines"
LINE_VIEWS = {"P(sand)": ("p_sand", "YlOrBr", 0, 1), "P(gas | sand)": ("p_gas_sand", "Reds", 0, 1),
              "Porosity P50": ("phi_p50", "Viridis", None, None),
              "Porosity uncertainty (P90-P10)": ("phi_spread", "Magma", None, None),
              "Acoustic impedance (mean)": ("ai_mean", "Cividis", None, None),
              "What the seismic adds: P(sand) posterior - prior": ("d_sand", "RdBu_r", -0.5, 0.5),
              "Most likely facies": ("ml_facies", None, None, None)}


@st.cache_data(show_spinner=False, max_entries=6)
def _sim_line(path: str, mtime: int):
    del mtime
    d = np.load(path)
    r = {k: d[k] for k in d.files}
    names = list(r["facies_names"])
    sand = [names.index(f) for f in ("sand_high_phi", "sand_low_phi") if f in names]
    fp, pp = r["facies_prob"].astype(np.float32), r["prior_facies_prob"].astype(np.float32)
    r["p_sand"] = fp[:, sand].sum(axis=1)
    r["d_sand"] = r["p_sand"] - pp[:, sand].sum(axis=1)
    r["p_gas_sand"] = np.where(r["p_sand"] > 0.05, r["p_gas"].astype(np.float32) / np.maximum(r["p_sand"], 1e-6),
                               np.nan)
    r["phi_p50"] = r["phi_q"][:, 1].astype(np.float32)
    r["phi_spread"] = (r["phi_q"][:, 2] - r["phi_q"][:, 0]).astype(np.float32)
    r["ai_mean"] = r["ai_mean"].astype(np.float32)
    r["ml_facies"] = fp.argmax(axis=1).astype(np.float32)
    return r


def sim_line_fig(r, view, pos, held):
    key, cmap, zmin, zmax = LINE_VIEWS[view]
    dist, rel, score = r["distance_m"] / 1000, r["rel_ms"], r["score"].astype(bool)
    t_axis = np.arange(np.floor(r["top_ms"].min() / 4) * 4 - 20, np.ceil(r["base_ms"].max() / 4) * 4 + 24, 4.0)
    img = np.full((len(t_axis), len(dist)), np.nan, np.float32)
    vol = r[key]
    for c in range(len(dist)):
        t = r["top_ms"][c] + rel[score[c]]
        idx = np.round((t - t_axis[0]) / 4).astype(int)
        ok = (idx >= 0) & (idx < len(t_axis))
        img[idx[ok], c] = vol[c][score[c]][ok]
    if key == "ml_facies":
        names = list(r["facies_names"])
        n = len(names)
        scale = []
        for i, f in enumerate(names):
            scale += [[i / n, LITHO_COLORS.get(f, "#fff")], [(i + 1) / n, LITHO_COLORS.get(f, "#fff")]]
        hm = go.Heatmap(z=img, x=dist, y=t_axis, colorscale=scale, zmin=-0.5, zmax=n - 0.5, showscale=False,
                        customdata=np.vectorize(lambda v: LITHO_LABELS.get(names[int(v)], "") if np.isfinite(v)
                                                else "")(img),
                        hovertemplate="%{x:.2f} km · %{y:.0f} ms<br>%{customdata}<extra></extra>")
    else:
        hm = go.Heatmap(z=img, x=dist, y=t_axis, colorscale=cmap, zmin=zmin, zmax=zmax, colorbar={"title": ""},
                        zmid=0 if key == "d_sand" else None,
                        hovertemplate="%{x:.2f} km · %{y:.0f} ms<br>%{z:.3f}<extra></extra>")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.8, 0.2], vertical_spacing=0.04)
    fig.add_trace(hm, 1, 1)
    fig.add_trace(go.Scatter(x=dist, y=r["top_ms"], mode="lines", line={"color": "black", "width": 1.5},
                             hoverinfo="skip"), 1, 1)
    fig.add_trace(go.Scatter(x=dist, y=r["base_ms"], mode="lines", line={"color": "black", "width": 1.5,
                                                                          "dash": "dot"}, hoverinfo="skip"), 1, 1)
    fig.add_trace(go.Scatter(x=dist, y=r["beta"], mode="lines", line={"color": BLUE}, name="seismic weight",
                             hovertemplate="seismic weight %{y:.2f}<extra></extra>"), 2, 1)
    fig.add_trace(go.Scatter(x=dist, y=r["fit_corr"], mode="lines", line={"color": GREEN}, name="fit",
                             hovertemplate="fit corr %{y:.2f}<extra></extra>"), 2, 1)
    route = {"inline": r["inline"], "crossline": r["crossline"], "distance_m": r["distance_m"]}
    for w, d in _route_wells(route, pos, max_traces=40).items():
        fig.add_vline(x=d, line={"color": "#24292f", "width": 1, "dash": "dot" if w in held else "solid"})
        fig.add_annotation(x=d, y=1.0, yref="paper", yanchor="bottom", showarrow=False, font={"size": 10},
                           text=w + (" (blind)" if w in held else ""), bgcolor="rgba(255,255,255,.85)")
    fig.update_yaxes(autorange="reversed", title_text="TWT (ms)", row=1, col=1)
    fig.update_yaxes(title_text="support", range=[0, 1.05], row=2, col=1)
    fig.update_xaxes(title_text="Distance along the section (km)", row=2, col=1)
    style(fig, 620, legend_below=False, showlegend=False)
    return fig


def _sim_line_results(pos, held):
    runs = sorted([p for p in SIM_LINE_DIR.glob("*.json") if p.with_suffix(".npz").is_file()],
                  key=lambda p: p.stat().st_mtime, reverse=True) if SIM_LINE_DIR.is_dir() else []
    if not runs:
        return
    st.subheader("Results along a line")
    labels = [p.stem for p in runs]
    default = st.session_state.get("sim_line_pick")
    c = st.columns([1.5, 2.5])
    choice = c[0].selectbox("Line simulation", labels, index=labels.index(default) if default in labels else 0,
                            key="sim_line_choice")
    view = c[1].segmented_control("Show", list(LINE_VIEWS), default="P(sand)", key="sim_line_view") or "P(sand)"
    path = SIM_LINE_DIR / f"{choice}.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    r = _sim_line(str(path.with_suffix(".npz")), _mtime(path.with_suffix(".npz")))
    k = st.columns(5)
    k[0].metric("Traces", f"{meta['n_traces']}", help=f"every {meta['trace_step']}th trace")
    k[1].metric("Length", f"{meta['length_km']:.1f} km")
    k[2].metric("Median seismic weight", f"{meta['beta_median']:.2f}",
                help="1 = full likelihood; lower: softened because the wells could not reproduce the seismic.")
    k[3].metric("Low support", f"{meta['low_support_fraction']:.0%}")
    k[4].metric("Median fit to seismic", f"{meta['fit_corr_median']:.2f}")
    st.plotly_chart(sim_line_fig(r, view, pos, held), width="stretch", key="sim_line_fig")
    st.caption(f"SIMULATED – not measured data. Each trace re-weights pseudo-wells from the {meta.get('zone_formation')} "
               f"of {', '.join(meta['training_wells'])} against its own near/mid/far inside the zone (solid: top, "
               "dotted: base). Bottom: seismic weight (blue) and fit (green) per trace. Probabilities are per trace; "
               "they are not a single geological realization.")


def _run_simulation_ui(il, xl, t0, t1, n, method="localised_is"):
    from avo import prospect_sim as ps
    bar = st.progress(0.0, text="Preparing prior from the training wells...")
    try:
        prior = sim_prior()
        meta = ps.run_at_location(ROOT, il, xl, t0, t1, n=n, prior=prior, overrides={"method": method},
                                  progress=lambda f, msg: bar.progress(min(f, 1.0), text=f"{msg} ({f:.0%})"))
    except ValueError as exc:
        bar.empty()
        st.error(str(exc))
        return None
    bar.empty()
    st.session_state["sim_selected"] = meta["name"]
    st.session_state["sim_pick_run"] = meta["name"]
    return meta


def _near_well(pos, il, xl, max_traces=3):
    p = pos.dropna(subset=["INLINE"])
    d = np.hypot(p["INLINE"] - il, p["CROSSLINE"] - xl)
    if len(d) and d.min() <= max_traces:
        return p.loc[d.idxmin(), "WELL_ID"]
    return None


def _band(fig, row, col, q, t, color, name, show):
    fig.add_trace(go.Scatter(x=np.r_[q[0], q[2][::-1]], y=np.r_[t, t[::-1]], fill="toself", fillcolor=color,
                             line={"width": 0}, hoverinfo="skip", name=name, legendgroup=name, showlegend=show),
                  row, col)


def sim_wells_figure(r, meta, actual):
    t = r["t_model_ms"]
    t0, t1 = meta["window_twt_ms"]
    m = (t >= t0 - 8) & (t <= t1 + 8)
    t = t[m]
    names = list(r["facies_names"])
    ncol = 3 + len(SIM_TRACKS)
    titles = ["Facies (prior)", "Facies (posterior)", "P(gas | sand)"] + [lab for _, lab in SIM_TRACKS]
    fig = make_subplots(rows=1, cols=ncol, shared_yaxes=True, horizontal_spacing=0.012, subplot_titles=titles)
    for col, ens in ((1, "prior"), (2, "post")):
        prob = r[f"{ens}_facies_prob"][:, m]
        cum = np.zeros(len(t))
        for fname in SIM_FACIES:
            k = names.index(fname)
            fig.add_trace(go.Scatter(x=cum.copy(), y=t, mode="lines", line={"width": 0}, hoverinfo="skip",
                                     showlegend=False), 1, col)
            cum = cum + prob[k]
            fig.add_trace(go.Scatter(x=cum.copy(), y=t, mode="lines", fill="tonextx", line={"width": 0},
                                     fillcolor=LITHO_COLORS[fname], name=LITHO_LABELS[fname], legendgroup=fname,
                                     showlegend=col == 2, customdata=prob[k],
                                     hovertemplate=f"{LITHO_LABELS[fname]}: %{{customdata:.2f}}<br>%{{y:.0f}} ms"
                                                   "<extra></extra>"), 1, col)
        fig.update_xaxes(range=[0, 1], row=1, col=col)
    si = [names.index("sand_high_phi"), names.index("sand_low_phi")]
    for ens, color, dash in (("prior", GREY, "dot"), ("post", RED, "solid")):
        psand = r[f"{ens}_facies_prob"][si][:, m].sum(axis=0)
        pg = np.where(psand > 0.03, r[f"{ens}_p_gas"][m] / np.maximum(psand, 1e-6), np.nan)
        fig.add_trace(go.Scatter(x=pg, y=t, mode="lines", line={"color": color, "width": 2, "dash": dash},
                                 name="P(gas | sand) prior" if ens == "prior" else "P(gas | sand) posterior",
                                 hovertemplate="%{x:.2f}<extra></extra>"), 1, 3)
    fig.update_xaxes(range=[0, 1], row=1, col=3)
    for c, (key, _) in enumerate(SIM_TRACKS, start=4):
        _band(fig, 1, c, r[f"prior_{key}_q"][:, m], t, "rgba(110,119,129,0.18)", "Prior P10–P90", c == 4)
        _band(fig, 1, c, r[f"post_{key}_q"][:, m], t, "rgba(31,111,235,0.28)", "Posterior P10–P90", c == 4)
        fig.add_trace(go.Scatter(x=r[f"post_{key}_q"][1, m], y=t, mode="lines", line={"color": BLUE, "width": 2},
                                 name="Posterior P50", legendgroup="p50", showlegend=c == 4,
                                 hovertemplate="%{x:.3g}<extra>P50</extra>"), 1, c)
        if actual is not None and key in actual:
            fig.add_trace(go.Scatter(x=actual[key], y=actual.index, mode="lines", line={"color": "black", "width": 1.3},
                                     name=f"Actual log ({meta.get('_actual_well')})", legendgroup="actual",
                                     showlegend=c == 4, hovertemplate="%{x:.3g}<extra>actual</extra>"), 1, c)
    if actual is not None:
        lith = actual["lithotype"].dropna()
        for fname in SIM_FACIES:
            sel = lith[lith == fname]
            if len(sel):
                fig.add_trace(go.Scatter(x=np.full(len(sel), 1.04), y=sel.index, mode="markers", cliponaxis=False,
                                         marker={"symbol": "square", "size": 5, "color": LITHO_COLORS[fname]},
                                         showlegend=False, hoverinfo="skip"), 1, 2)
    for c in range(1, ncol + 1):
        fig.add_hline(y=t0, line={"color": "black", "width": 1, "dash": "dash"}, row=1, col=c)
        fig.add_hline(y=t1, line={"color": "black", "width": 1, "dash": "dash"}, row=1, col=c)
    fig.update_yaxes(autorange="reversed", title_text="TWT (ms)", row=1, col=1)
    fig.update_annotations(font_size=11)
    style(fig, 760)
    fig.update_layout(legend={"orientation": "h", "y": -0.07, "x": 0})
    return fig


def sim_avo_figure(r, meta):
    t = r["t_out_ms"]
    t0, t1 = meta["window_twt_ms"]
    stacks = list(r["stacks"])
    fig = make_subplots(rows=1, cols=len(stacks), shared_yaxes=True, horizontal_spacing=0.03,
                        subplot_titles=[f"{s} stack" for s in stacks])
    for k, s in enumerate(stacks, start=1):
        _band(fig, 1, k, r["prior_syn_q"][:, k - 1], t, "rgba(110,119,129,0.18)", "Prior P10–P90", k == 1)
        _band(fig, 1, k, r["post_syn_q"][:, k - 1], t, "rgba(31,111,235,0.28)", "Posterior P10–P90", k == 1)
        fig.add_trace(go.Scatter(x=r["post_syn_q"][1, k - 1], y=t, mode="lines", line={"color": BLUE, "width": 2},
                                 name="Posterior P50", showlegend=k == 1), 1, k)
        fig.add_trace(go.Scatter(x=r["obs_norm"][k - 1], y=t, mode="lines", line={"color": "black", "width": 2},
                                 name="Observed (calibrated)", showlegend=k == 1), 1, k)
        if "ex_best_syn" in r:
            fig.add_trace(go.Scatter(x=r["ex_best_syn"][0, k - 1], y=t, mode="lines",
                                     line={"color": RED, "width": 1.5, "dash": "dot"},
                                     name="Best-matching pseudo-well", showlegend=k == 1), 1, k)
        fig.add_hrect(y0=t0, y1=t1, fillcolor="rgba(255,211,61,0.12)", line_width=0, row=1, col=k)
    fig.update_yaxes(autorange="reversed", title_text="TWT (ms)", row=1, col=1)
    style(fig, 560)
    return fig


def sim_avo_curve(r, meta, t_pick):
    t = r["t_out_ms"]
    k = int(np.argmin(np.abs(t - t_pick)))
    calib = calibration() or {}
    angles = [0.5 * (v["angle_min"] + v["angle_max"]) for v in calib.get("stacks", {}).values()] or [12, 24, 36]
    fig = go.Figure()
    for ens, color, name in (("prior", GREY, "Prior"), ("post", BLUE, "Posterior")):
        q = r[f"{ens}_syn_q"][:, :, k]
        fig.add_trace(go.Scatter(x=angles, y=q[1], mode="lines+markers", name=f"{name} P50",
                                 line={"color": color, "width": 2},
                                 error_y={"type": "data", "symmetric": False, "array": q[2] - q[1],
                                          "arrayminus": q[1] - q[0], "color": color, "thickness": 1.5}))
    fig.add_trace(go.Scatter(x=angles, y=r["obs_norm"][:, k], mode="markers", name="Observed",
                             marker={"size": 13, "color": "black", "symbol": "diamond"}))
    fig.add_hline(y=0, line={"color": "#d0d7de"})
    fig.update_xaxes(title_text="Mean incidence angle of the stack (deg)")
    fig.update_yaxes(title_text="Amplitude (calibrated synthetic units)")
    style(fig, 420, title={"text": f"AVO response at {t[k]:.0f} ms", "font": {"size": 13}})
    return fig


def sim_examples_figure(r, meta):
    t = r["t_model_ms"]
    t0, t1 = meta["window_twt_ms"]
    m = (t >= t0) & (t <= t1)
    names = list(r["facies_names"])
    order = [names.index(f) for f in SIM_FACIES]
    remap = np.full(len(names), 0)
    for new, old in enumerate(order):
        remap[old] = new
    fac = r["ex_best_facies"][:20, m].astype(int)
    z = remap[np.clip(fac, 0, len(names) - 1)].T
    n = len(SIM_FACIES)
    scale = []
    for i, f in enumerate(SIM_FACIES):
        scale += [[i / n, LITHO_COLORS[f]], [(i + 1) / n, LITHO_COLORS[f]]]
    fig = go.Figure(go.Heatmap(z=z, x=np.arange(1, z.shape[1] + 1), y=t[m], colorscale=scale, zmin=-0.5,
                               zmax=n - 0.5, showscale=False,
                               customdata=np.array([[LITHO_LABELS[SIM_FACIES[v]] for v in row] for row in z]),
                               hovertemplate="#%{x} %{y:.0f} ms<br>%{customdata}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", title_text="TWT (ms)")
    fig.update_xaxes(title_text="Best-matching pseudo-wells (rank)", dtick=1)
    style(fig, 420, legend_below=False, title={"text": "Example pseudo-wells (facies)", "font": {"size": 13}})
    return fig


def _actual_logs(well, t0, t1):
    d = qc_logs(well)
    if d is None or "twt" not in d:
        return None
    d = d[(d["twt"] >= t0 - 8) & (d["twt"] <= t1 + 8)].copy()
    if d.empty:
        return None
    d["ai"], d["vpvs"] = d["vp"] * d["rho"], d["vp"] / d["vs"]
    d.loc[~d["lithotype"].isin(["sand_high_phi", "sand_low_phi"]), "sw"] = np.nan
    ms = np.round(d["twt"])
    out = d.groupby(ms)[["phi", "vsh", "sw", "vp", "vs", "rho", "ai", "vpvs"]].mean()
    out["lithotype"] = d.groupby(ms)["lithotype"].agg(lambda s: s.mode().iat[0] if len(s.mode()) else None)
    return out


def page_simulation():
    step_header(9, "Prospect simulation (pseudo-wells in the reservoir zone)",
                "Draw a **section** – an inline, a crossline or a **composite line** through one or more wells – and "
                "mark the **reservoir zone** on it (Plover Fm; pre-filled from the tops of the wells the simulation "
                "uses). Thousands of **pseudo-wells** are simulated from the rock physics of the **Plover Formation in "
                "the training wells only** (facies sequences, porosity, clay, velocities, density, gas or brine), "
                "forward-modelled to near/mid/far and weighted by how well they match the **observed AVO signal** "
                "inside the zone. Simulate one pseudo-well, or every trace along the line.",
                "where the posterior (blue) is narrower than the prior (grey): there the seismic tells us "
                "something. Where they overlap, the answer comes from the wells alone.")
    stacks, ready, geometry, positions = seismic()
    if not ready or geometry is None:
        st.warning("The seismic display cache is missing. Build it in step 1.")
        return
    if not CAL_PATH.is_file():
        st.warning("Run the well tie (step 5) first: the simulation uses its wavelets and amplitude scales.")
        return
    pos = positions.dropna(subset=["INLINE"])
    if "X_M" not in pos:
        pos = pos.merge(wells()[["WELL_ID", "X_M", "Y_M"]], on="WELL_ID", how="left")
    held = set((calibration() or {}).get("held_out_wells", []))
    zone, zw = zone_wells()
    samples = geometry["samples"]

    if "sim_il" not in st.session_state:
        w = st.session_state.get("well")
        row = pos[pos["WELL_ID"] == w]
        row = row.iloc[0] if len(row) else pos.iloc[0]
        st.session_state["sim_il"], st.session_state["sim_xl"] = int(row["INLINE"]), int(row["CROSSLINE"])
    if "sim_vertices" not in st.session_state:
        st.session_state["sim_vertices"] = [_vertex(pos, int(r.INLINE), int(r.CROSSLINE))
                                            for w in DEFAULT_TRAVERSE for r in pos[pos["WELL_ID"] == w].itertuples()]
    st.session_state.setdefault("sim_mode", "Composite line")
    _consume_map_click(geometry, pos)

    # ------------------------------------------------------------------ 1. section
    st.subheader("1 · Choose the section")
    c = st.columns([1.5, 1.4, 3])
    mode = c[0].segmented_control("Section", ["Composite line", "Inline", "Crossline"], key="sim_mode") \
        or "Composite line"
    c[1].segmented_control("A click on the map", ["Move pseudo-well", "Add line vertex"],
                           default="Move pseudo-well", key="sim_click_mode")
    vertices = st.session_state["sim_vertices"]
    if mode == "Composite line":
        c[2].caption("**Line:** " + (" → ".join(v["label"] for v in vertices) or "empty") +
                     ". Add wells below, or choose *Add line vertex* and click the map (clicks near a well snap to it).")
        v = st.columns([1.6, 0.5, 0.5, 0.5, 2.5])
        add = v[0].selectbox("Add a well to the line", ["(choose)"] + list(pos["WELL_ID"]), key="sim_add_well",
                             label_visibility="collapsed")
        if v[1].button("Add", key="sim_add_btn") and add != "(choose)":
            r = pos[pos["WELL_ID"] == add].iloc[0]
            st.session_state["sim_vertices"] = vertices + [_vertex(pos, int(r["INLINE"]), int(r["CROSSLINE"]))]
            st.rerun()
        if v[2].button("Undo", key="sim_undo_vertex") and vertices:
            st.session_state["sim_vertices"] = vertices[:-1]
            st.rerun()
        if v[3].button("Clear", key="sim_clear_line"):
            st.session_state["sim_vertices"] = []
            st.rerun()
    if mode == "Composite line" and len(vertices) < 2:
        st.info("The line needs at least two points.")
        st.plotly_chart(sim_route_map(geometry, pos, None, st.session_state["sim_il"], st.session_state["sim_xl"],
                                      vertices, mode), width="stretch", key="sim_map", on_select="rerun",
                        selection_mode="points")
        step_nav("sim")
        return

    stack = st.session_state.get("sim_stack") or "far"
    pts = _route_points(geometry, mode, st.session_state["sim_il"], st.session_state["sim_xl"], vertices)
    route, traces = _traverse(str(ready[stack]), _mtime(ready[stack]), pts)
    dist = route["distance_m"] / 1000
    rwells = _route_wells(route, pos)
    sig = (mode, pts)
    if st.session_state.get("sim_route_sig") != sig:
        st.session_state["sim_route_sig"] = sig
        st.session_state["sim_top"], st.session_state["sim_base"] = _zone_defaults(rwells, zw)
        st.session_state["sim_sec_sig"] = None
    _consume_route_click(dist, route)
    # keep the pseudo-well on the section
    k = int(np.argmin(np.hypot(route["inline"] - st.session_state["sim_il"],
                               route["crossline"] - st.session_state["sim_xl"])))
    st.session_state["sim_il"], st.session_state["sim_xl"] = int(route["inline"][k]), int(route["crossline"][k])
    il, xl, point_d = st.session_state["sim_il"], st.session_state["sim_xl"], float(dist[k])
    top_line = _zone_line(st.session_state["sim_top"], dist)
    base_line = _zone_line(st.session_state["sim_base"], dist)
    t0 = t1 = None
    if top_line is not None and base_line is not None:
        t0, t1 = float(np.floor(top_line[k] / 4) * 4), float(np.ceil(base_line[k] / 4) * 4)

    # ------------------------------------------------------------------ 2. zone
    st.subheader("2 · Mark the reservoir zone")
    p = st.columns([1.8, 1.1, 0.8, 0.8, 1.3, 1.6])
    pick_mode = p[0].segmented_control("A click on the section sets", ["Top", "Base", "Pseudo-well"], default="Top",
                                       key="sim_pick_mode") or "Top"
    if p[1].button("Zone from well tops", key="sim_zone_wells",
                   help=f"Top and base of {zone} in the wells on this section that the simulation uses."):
        st.session_state["sim_top"], st.session_state["sim_base"] = _zone_defaults(rwells, zw)
        st.rerun()
    if p[2].button("Undo pick", key="sim_undo_pick"):
        key = "sim_base" if pick_mode == "Base" else "sim_top"
        last = st.session_state.get(f"{key}_last")
        picks = st.session_state.get(key) or []
        if picks:
            i_last = int(np.argmin([abs(a - last) for a, _ in picks])) if last is not None else len(picks) - 1
            st.session_state[key] = picks[:i_last] + picks[i_last + 1:]
        st.rerun()
    if p[3].button("Clear", key="sim_clear_picks"):
        st.session_state["sim_top"], st.session_state["sim_base"] = [], []
        st.rerun()
    p[4].segmented_control("Stack", ["near", "mid", "far"], default="far", key="sim_stack")
    centre = float(np.nanmean(top_line)) if top_line is not None else 3150.0
    tw = p[5].slider("Display (ms)", float(samples[0]), float(samples[-1]),
                     (max(float(samples[0]), centre - 250), min(float(samples[-1]), centre + 450)), 4.0,
                     key=f"sim_tw_{mode}")
    on_route = [w for w in rwells if w in zw]
    blind_on = [w for w in rwells if w in held]
    no_tops = [w for w in rwells if w not in zw and w not in held]
    st.caption(
        (f"Zone pre-filled from {', '.join(on_route)}. " if on_route else
         "No well with reservoir tops on this section – pick the zone top and base yourself. ")
        + (f"Blind wells (tops hidden): {', '.join(blind_on)}. " if blind_on else "")
        + (f"Not used (no {zone} tops or excluded): {', '.join(no_tops)}. " if no_tops else "")
        + "Clicks add a pick (a click near an existing pick replaces it); the zone is interpolated between picks.")

    left, right = st.columns([1, 2.3])
    left.plotly_chart(sim_route_map(geometry, pos, route, il, xl, vertices, mode), width="stretch", key="sim_map",
                      on_select="rerun", selection_mode="points")
    right.plotly_chart(sim_route_fig(route, traces, samples, tw, rwells, held, (top_line, base_line),
                                     {"top": st.session_state["sim_top"], "base": st.session_state["sim_base"]},
                                     point_d, t0, t1, stack),
                       width="stretch", key="sim_sec", on_select="rerun", selection_mode="points")

    # ------------------------------------------------------------------ 3. simulate
    st.subheader("3 · Simulate")
    if t0 is None:
        st.info("Mark both the top and the base of the zone first.")
    else:
        _simulate_panel(route, dist, vertices, mode, il, xl, point_d, t0, t1, top_line, base_line, pos, held)
    _sim_line_results(pos, held)


def _simulate_panel(route, dist, vertices, mode, il, xl, point_d, t0, t1, top_line, base_line, pos, held):
    valid = t1 - t0 >= 20
    st.markdown(f"**Pseudo-well:** IL {il} / XL {xl}, {point_d:.2f} km along the section · zone "
                f"**{t0:.0f}–{t1:.0f} ms** (choose *Pseudo-well* above and click the section to move it).")
    if not valid:
        st.error("The zone must be at least 20 ms thick at the pseudo-well.")
    near = _near_well(pos, il, xl)
    if near:
        st.info(f"This location is at **{near}**"
                + (" – a **blind well**: its logs were not used anywhere in the simulation, so this is a fair test."
                   if near in held else " – a training well, so this is not an independent test."))
    tab_pt, tab_line = st.tabs(["One pseudo-well", "Every trace along the line"])
    with tab_pt:
        r1 = st.columns([1.3, 1.4, 1, 2.3])
        method = r1[0].segmented_control(
            "Method", ["Localised weighting", "Sequential (SMC)"], default="Localised weighting", key="sim_method",
            help="Localised weighting (recommended): best facies and log predictions in tests on training wells. "
                 "Sequential Monte Carlo: pseudo-wells match the seismic more closely, but tend to explain weak "
                 "amplitudes with stiffer, more uniform rock.") or "Localised weighting"
        smc = method.startswith("Sequential")
        options = [2000, 5000, 10000] if smc else [5000, 10000, 20000, 40000]
        n = r1[1].select_slider("Number of pseudo-wells", options, value=5000 if smc else 20000,
                                key=f"sim_n_{'smc' if smc else 'is'}")
        win_s = max(t1 - t0, 20.0) / 1000
        secs = n / 5000 * (25 + 120 * win_s) if smc else n / 20000 * 45
        r1[3].caption(f"About {max(15, int(secs))} s (plus ~15 s the first time, to build the prior). Saved to "
                      "outputs/simulation/ and shown under *Results* below.")
        if r1[1].button("Run simulation", type="primary", disabled=not valid, key="sim_run"):
            meta = _run_simulation_ui(int(il), int(xl), t0, t1, int(n), "smc" if smc else "localised_is")
            if meta:
                st.success(f"Saved **{meta['name']}** ({meta['runtime_s']} s).")
    with tab_line:
        r2 = st.columns([1, 1.3, 1.5, 2.2])
        step = r2[0].select_slider("Every n-th trace", [2, 4, 8, 16, 32], value=8, key="sim_line_step")
        n_lib = r2[1].select_slider("Pseudo-wells per depth band", [2000, 5000, 10000], value=5000, key="sim_line_n")
        label = "_".join(v["label"] for v in vertices) if mode == "Composite line" else f"{mode.lower()}_{il if mode == 'Inline' else xl}"
        name = r2[2].text_input("Name", "line_" + "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in label)[:60],
                                key="sim_line_name")
        n_tr = len(dist) // step + 1
        r2[3].caption(f"{n_tr} traces over {dist[-1]:.1f} km, each scored only inside its own zone. About "
                      f"{max(10, int(n_tr * n_lib / 5000 * 0.12 + 10))} s. Saved to outputs/simulation_lines/.")
        ok_line = bool(np.all(base_line - top_line >= 8))
        if not ok_line:
            st.error("The zone must be at least 8 ms thick everywhere along the line.")
        if r2[0].button("Simulate the line", type="primary", disabled=not ok_line, key="sim_line_run"):
            from avo import prospect_sim3d as p3
            bar = st.progress(0.0, text="Preparing...")
            try:
                meta = p3.run_line(ROOT, name.strip() or "line", route["i"], route["j"], route["distance_m"],
                                   top_line, base_line, step=int(step), n=int(n_lib), prior=sim_prior(),
                                   progress=lambda f, msg: bar.progress(min(f, 1.0), text=f"{msg} ({f:.0%})"),
                                   extra_meta={"section": mode, "vertices": [v["label"] for v in vertices]
                                               if mode == "Composite line" else None,
                                               "top_picks": st.session_state["sim_top"],
                                               "base_picks": st.session_state["sim_base"]})
                st.session_state["sim_line_pick"] = meta["name"]
                st.success(f"Saved **{meta['name']}** ({meta['runtime_s']} s).")
            except ValueError as exc:
                st.error(str(exc))
            bar.empty()
    _sim_line_results(pos, held)

    runs = sim_runs()
    if not runs:
        st.caption("No simulations yet.")
        step_nav("sim")
        return
    st.subheader("Results")
    labels = [p.stem for p in runs]
    default = st.session_state.get("sim_selected")
    choice = st.selectbox("Simulation", labels, index=labels.index(default) if default in labels else 0,
                          key="sim_pick_run")
    path = SIM_DIR / f"{choice}.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    r = _sim_result(str(path.with_suffix(".npz")), _mtime(path.with_suffix(".npz")))
    t0r, t1r = meta["window_twt_ms"]
    z0, z1 = meta.get("z_window_tvdss_m", [np.nan, np.nan])
    k = st.columns(6)
    k[0].metric("Location", f"IL {meta['inline']} / XL {meta['crossline']}")
    k[1].metric("Window (TWT)", f"{t0r:.0f}–{t1r:.0f} ms")
    k[2].metric("Approx. depth (TVDSS)", f"{z0:,.0f}–{z1:,.0f} m",
                help=f"Time-depth from {meta['time_depth_from']} ({meta['time_depth_distance_km']} km away).")
    k[3].metric("Pseudo-wells", f"{meta['n_realizations']:,}",
                help="Effective number after weighting (lowest in the window): "
                     f"{meta['ess_local_min']:,.0f}; median {meta['ess_local_median']:,.0f}.")
    k[4].metric("P(gas column)", f"{meta['p_gas_column_posterior_global']:.2f}",
                f"{meta['p_gas_column_posterior_global'] - meta['p_gas_column_prior']:+.2f} vs prior",
                delta_color="off", help="Global weighting; read together with the P(gas | sand) track.")
    k[5].metric("Seismic weight (beta)", f"{meta.get('adaptive_beta', 1.0):.2f}",
                help="1 = full weight. Lower values mean the wells could not reproduce the seismic here, so the "
                     "likelihood was softened to keep enough pseudo-wells.")
    if meta.get("adaptive_beta", 1.0) < 0.5:
        st.warning("The amplitudes here are outside what the training wells can produce (seismic weight "
                   f"{meta['adaptive_beta']:.2f}). The result leans on the wells more than on the seismic – treat it "
                   "as a weakly conditioned estimate.")
    method_label = {"smc": "sequential Monte Carlo"}.get(meta.get("method"), "localised importance weighting")
    st.caption(f"SIMULATED pseudo-well – not measured data. Method: {method_label}. Blind wells "
               f"({', '.join(meta.get('held_out_wells_excluded', []))}) are never used.")

    actual = None
    near_r = _near_well(pos, meta["inline"], meta["crossline"])
    if near_r:
        actual = _actual_logs(near_r, t0r, t1r)
        meta["_actual_well"] = near_r
    tab1, tab2, tab3 = st.tabs(["Pseudo-well logs", "AVO signal", "Example pseudo-wells"])
    with tab1:
        st.plotly_chart(sim_wells_figure(r, meta, actual), width="stretch", key="sim_logs")
        st.caption("Grey: prior (wells only). Blue: posterior (wells + seismic). Facies tracks are probabilities. "
                   "Sw is shown for sands; non-reservoir rock is water-bearing (Sw = 1)."
                   + (f" Black: actual logs of {near_r}; coloured squares next to the posterior facies: its log "
                      "facies." if actual is not None else ""))
        blind = SIM_DIR / f"{choice}_blind.csv"
        if blind.is_file():
            with st.expander("Blind-test scores against the actual logs"):
                st.dataframe(pd.read_csv(blind), hide_index=True, width="stretch")
    with tab2:
        st.plotly_chart(sim_avo_figure(r, meta), width="stretch", key="sim_avo")
        t_out = r["t_out_ms"][r["score_mask"].astype(bool)]
        k_def = int(np.argmax(np.abs(r["obs_norm"][:, r["score_mask"].astype(bool)]).sum(axis=0)))
        t_pick = st.select_slider("Time for the AVO curve (ms)", options=[float(v) for v in t_out],
                                  value=float(t_out[k_def]), key=f"sim_avo_t_{choice}")
        a, b = st.columns([1.3, 1])
        a.plotly_chart(sim_avo_curve(r, meta, t_pick), width="stretch", key="sim_avo_curve")
        b.markdown("**How to read this**\n\n"
                   "- Black diamonds: observed near/mid/far amplitude at the chosen time.\n"
                   "- Grey: what the wells alone predict here (prior).\n"
                   "- Blue: pseudo-wells that match the seismic (posterior).\n"
                   "- Red dotted (traces): the single pseudo-well that matches the whole window best.\n\n"
                   "Amplitudes are in calibrated synthetic units (seismic divided by the global scale from step 5); "
                   "reverse polarity is applied, so a trough is an increase in impedance.")
    with tab3:
        st.plotly_chart(sim_examples_figure(r, meta), width="stretch", key="sim_examples")
        st.caption("The 20 individual pseudo-wells that match the whole window best. They show realistic layering; "
                   "the probability tracks summarise all pseudo-wells.")
    step_nav("sim")


# ----------------------------------------------------------------------------------------------- step 10
SIM3D_DIR = ROOT / "outputs" / "simulation3d"
SIM3D_SPACING = {"150 m (slow)": (8, 12), "300 m": (16, 24), "600 m (quick look)": (32, 48)}
SIM3D_ATTRS = {
    "Porosity P50": ("phi_p50", "Viridis", "Porosity (v/v)"),
    "Porosity uncertainty (P90-P10)": ("phi_spread", "Magma", "P90-P10 (v/v)"),
    "P(sand)": ("p_sand", "YlOrBr", "P(sand)"),
    "P(gas)": ("p_gas", "Reds", "P(gas)"),
    "P(gas | sand)": ("p_gas_sand", "Reds", "P(gas | sand)"),
    "Acoustic impedance (mean)": ("ai_mean", "Cividis", "AI"),
    "Realization: porosity": ("real_phi", "Viridis", "Porosity (v/v)"),
    "Realization: gas": ("real_gas", "Reds", "Gas (0/1)"),
}


def sim3d_runs() -> list[Path]:
    if not SIM3D_DIR.is_dir():
        return []
    runs = [p for p in SIM3D_DIR.glob("*.json") if not p.stem.endswith("_blind") and p.with_suffix(".npz").is_file()]
    return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)


@st.cache_data(show_spinner=False, max_entries=4)
def _sim3d(path: str, mtime: int):
    del mtime
    d = np.load(path)
    r = {k: d[k] for k in d.files}
    names = list(r["facies_names"])
    sand = [names.index(f) for f in ("sand_high_phi", "sand_low_phi") if f in names]
    fp = r["facies_prob"].astype(np.float32)                  # (ni, nj, F, nt)
    r["p_sand"] = fp[:, :, sand].sum(axis=2)
    r["p_gas"] = r["p_gas"].astype(np.float32)
    r["p_gas_sand"] = np.where(r["p_sand"] > 0.05, r["p_gas"] / np.maximum(r["p_sand"], 1e-6), np.nan)
    r["phi_p50"] = r["phi_q"][:, :, 1].astype(np.float32)
    r["phi_spread"] = (r["phi_q"][:, :, 2] - r["phi_q"][:, :, 0]).astype(np.float32)
    r["ai_mean"] = r["ai_mean"].astype(np.float32)
    r["ml_facies"] = np.where(np.isfinite(fp).all(axis=2), np.nan_to_num(fp).argmax(axis=2), -1)
    return r


def _sim3d_volume(r, key, realization):
    if key.startswith("real_"):
        v = r[key][realization].astype(np.float32)
        return np.where(r["real_facies"][realization] >= 0, v, np.nan)
    return r[key]


def sim3d_map_fig(r, key, label, cmap, agg, rel_t, realization, pos, mask_low, min_beta, line):
    sc = r["score_mask"].astype(bool)
    if key in ("beta", "fit_corr"):
        z = r[key].astype(np.float32)
    else:
        vol = _sim3d_volume(r, key, realization)
        if agg == "Mean over window":
            z = np.nanmean(vol[:, :, sc], axis=2)
        elif agg == "Max over window":
            z = np.nanmax(vol[:, :, sc], axis=2)
        else:
            z = vol[:, :, int(np.argmin(np.abs(r["rel_ms"] - rel_t)))]
    if mask_low:
        z = np.where(r["beta"] >= min_beta, z, np.nan)
    fig = go.Figure(go.Heatmap(z=z, x=r["xlines"], y=r["ilines"], colorscale=cmap, colorbar={"title": label},
                               hovertemplate="IL %{y} / XL %{x}<br>%{z:.3f}<extra></extra>"))
    p = pos.dropna(subset=["INLINE"])
    held = set((calibration() or {}).get("held_out_wells", []))
    fig.add_trace(go.Scatter(x=p["CROSSLINE"], y=p["INLINE"], mode="markers+text", text=p["WELL_ID"],
                             textposition="top center", textfont={"size": 10}, hovertemplate="%{text}<extra></extra>",
                             marker={"size": 10, "symbol": ["diamond" if w in held else "circle" for w in p["WELL_ID"]],
                                     "color": ["#ffffff" if w in held else "#24292f" for w in p["WELL_ID"]],
                                     "line": {"color": "#24292f", "width": 1.5}}))
    kind, value = line
    if kind == "Inline":
        fig.add_hline(y=value, line={"color": RED, "width": 2})
    else:
        fig.add_vline(x=value, line={"color": RED, "width": 2})
    fig.update_xaxes(title_text="Crossline")
    fig.update_yaxes(title_text="Inline", scaleanchor="x", scaleratio=18.75 / 12.5)
    style(fig, 560, legend_below=False, showlegend=False)
    return fig


def sim3d_section_fig(r, key, label, cmap, realization, kind, value):
    a = int(np.argmin(np.abs((r["ilines"] if kind == "Inline" else r["xlines"]) - value)))
    lateral = r["xlines"] if kind == "Inline" else r["ilines"]
    take = (lambda v: v[a]) if kind == "Inline" else (lambda v: v[:, a])
    surf = take(r["surface_twt_ms"])
    rel = r["rel_ms"]
    ok = np.isfinite(surf)
    if not ok.any():
        return None
    t_axis = np.arange(np.nanmin(surf) + rel[0], np.nanmax(surf) + rel[-1] + 4, 4.0)
    facies_mode = key in ("ml_facies", "real_facies")
    vol = take(r[key][realization] if key == "real_facies" else
               (r[key] if facies_mode else _sim3d_volume(r, key, realization))).astype(np.float32)
    img = np.full((len(t_axis), len(lateral)), np.nan, np.float32)
    for c in np.flatnonzero(ok):
        k0 = int(round((surf[c] + rel[0] - t_axis[0]) / 4))
        col = vol[c].copy()
        if facies_mode:
            col[col < 0] = np.nan
        img[k0:k0 + len(rel), c] = col[:len(t_axis) - k0]
    if facies_mode:
        names = list(r["facies_names"])
        n = len(names)
        scale = []
        for i, f in enumerate(names):
            scale += [[i / n, LITHO_COLORS.get(f, "#ffffff")], [(i + 1) / n, LITHO_COLORS.get(f, "#ffffff")]]
        hm = go.Heatmap(z=img, x=lateral, y=t_axis, colorscale=scale, zmin=-0.5, zmax=n - 0.5, showscale=False,
                        hovertemplate="%{x} · %{y:.0f} ms<br>facies %{z}<extra></extra>")
    else:
        hm = go.Heatmap(z=img, x=lateral, y=t_axis, colorscale=cmap, colorbar={"title": label},
                        hovertemplate="%{x} · %{y:.0f} ms<br>%{z:.3f}<extra></extra>")
    fig = go.Figure(hm)
    sc = r["score_mask"].astype(bool)
    for off, dash in ((rel[sc][0], "dot"), (0.0, "solid"), (rel[sc][-1], "dot")):
        fig.add_trace(go.Scatter(x=lateral[ok], y=surf[ok] + off, mode="lines", hoverinfo="skip",
                                 line={"color": "black", "width": 1.5 if off == 0 else 1, "dash": dash}))
    fig.update_yaxes(autorange="reversed", title_text="TWT (ms)")
    fig.update_xaxes(title_text="Crossline" if kind == "Inline" else "Inline")
    style(fig, 460, legend_below=False, showlegend=False,
          title={"text": f"{kind} {int(r['ilines'][a] if kind == 'Inline' else r['xlines'][a])} · solid: interpolated "
                         "top Plover, dotted: scored window", "font": {"size": 13}})
    return fig


def _run_sim3d_ui(name, steps, n):
    args = [sys.executable, str(ROOT / "scripts" / "run_prospect_sim3d.py"), "--name", name,
            "--il-step", str(steps[0]), "--xl-step", str(steps[1]), "--n", str(n)]
    with st.status("Running the 3D simulation (several minutes)...", expanded=False) as status:
        result = subprocess.run(args, cwd=str(ROOT), capture_output=True, text=True, timeout=7200)
        st.code((result.stdout or "")[-4000:] + (result.stderr or "")[-2000:], language="text")
        if result.returncode == 0:
            status.update(label="3D simulation finished", state="complete")
            st.cache_data.clear()
            st.session_state["sim3d_pick"] = name
        else:
            status.update(label="3D simulation failed (see output)", state="error")


def page_sim3d():
    step_header(10, "3D extrapolation",
                "The pseudo-well simulation from step 9, repeated on a grid of traces between the wells. Pseudo-wells "
                "are simulated **once per depth band** (a shared library) and re-weighted at every trace against its "
                "own near/mid/far signal, in a window that follows **top Plover interpolated from the training "
                "wells**. Laterally consistent **realizations** pick similar pseudo-wells at neighbouring traces.",
                "where the support map is high (the wells can explain the seismic) and whether sands and gas line up "
                "with the structure. The blind wells (white diamonds) were never used.")
    stacks, ready, geometry, positions = seismic()
    if not ready or geometry is None or not CAL_PATH.is_file():
        st.warning("Needs the seismic cache (step 4) and the well tie (step 5).")
        return
    pos = positions.dropna(subset=["INLINE"])

    with st.expander("Run a new 3D simulation", expanded=not sim3d_runs()):
        c = st.columns([1.2, 1.2, 1, 1])
        spacing = c[0].selectbox("Trace spacing", list(SIM3D_SPACING), index=1, key="sim3d_spacing")
        n = c[1].select_slider("Pseudo-wells per depth band", [2000, 5000, 10000], value=5000, key="sim3d_n")
        name = c[2].text_input("Name", "grid3d", key="sim3d_name")
        c[3].caption("300 m spacing with 5000 pseudo-wells takes a few minutes; 150 m about 4x longer.")
        if c[3].button("Run 3D simulation", type="primary", key="sim3d_run"):
            _run_sim3d_ui(name.strip() or "grid3d", SIM3D_SPACING[spacing], int(n))

    runs = sim3d_runs()
    if not runs:
        st.caption("No 3D simulations yet.")
        step_nav("sim3d")
        return
    labels = [p.stem for p in runs]
    default = st.session_state.get("sim3d_pick")
    choice = st.selectbox("3D simulation", labels, index=labels.index(default) if default in labels else 0)
    path = SIM3D_DIR / f"{choice}.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    r = _sim3d(str(path.with_suffix(".npz")), _mtime(path.with_suffix(".npz")))
    min_beta = meta["config"]["support"]["min_beta"]

    k = st.columns(6)
    k[0].metric("Traces", f"{meta['n_traces']:,}", help=f"Grid {meta['grid_shape'][0]} x {meta['grid_shape'][1]}")
    k[1].metric("Spacing", f"{meta['node_spacing_m'][0]:.0f} x {meta['node_spacing_m'][1]:.0f} m")
    k[2].metric("Pseudo-wells / band", f"{meta['library_n_per_band']:,}",
                help=f"Depth bands: {', '.join(f'{b:.0f}' for b in meta['depth_bands_m'])} m")
    k[3].metric("Median seismic weight", f"{meta['beta_median']:.2f}",
                help="1 = full likelihood. Lower values: the likelihood was softened to keep enough pseudo-wells.")
    k[4].metric("Low support", f"{meta['low_support_fraction']:.0%}", help=f"Traces with seismic weight < {min_beta}")
    k[5].metric("Median fit to seismic", f"{meta['fit_corr_median']:.2f}",
                help="Correlation between the observed near/mid/far window and the posterior-mean synthetic.")
    pts = ", ".join(f"{p['well']} ({p['twt_ms']:.0f} ms)" for p in meta["surface"]["points"])
    st.caption(f"SIMULATED – not measured data. Target surface: first '{meta['surface']['surface_prefix']}' pick in "
               f"{pts}; plane fit. Window {meta['window_rel_ms'][0]:.0f} to +{meta['window_rel_ms'][1]:.0f} ms. "
               f"Blind wells ({', '.join(meta['held_out_wells_excluded'])}) are never used. Runtime "
               f"{meta['runtime_s'] / 60:.1f} min.")

    c = st.columns([1.6, 1.3, 1.2, 1, 1])
    extra = {"Support (seismic weight)": ("beta", "Blues", "Seismic weight"),
             "Fit to seismic": ("fit_corr", "Blues", "Correlation")}
    attr = c[0].selectbox("Property", list(SIM3D_ATTRS) + list(extra), key="sim3d_property")
    key, cmap, label = SIM3D_ATTRS[attr] if attr in SIM3D_ATTRS else extra[attr]
    agg = c[1].segmented_control("Map", ["Mean over window", "Max over window", "Time slice"],
                                 default="Mean over window", key="sim3d_agg") or "Mean over window"
    sc_rel = r["rel_ms"][r["score_mask"].astype(bool)]
    rel_t = c[2].select_slider("Slice (ms below top)", [float(v) for v in sc_rel], value=0.0, key="sim3d_rel",
                               disabled=agg != "Time slice")
    realization = int(c[3].number_input("Realization", 1, int(r["real_facies"].shape[0]), 1, key="sim3d_real")) - 1
    mask_low = c[4].toggle("Hide low support", value=True, key="sim3d_mask")

    s = st.columns([1, 1, 2.2])
    kind = s[0].segmented_control("Section", ["Inline", "Crossline"], default="Inline", key="sim3d_kind") or "Inline"
    axis = r["ilines"] if kind == "Inline" else r["xlines"]
    mid = int(axis[len(axis) // 2])
    value = s[1].select_slider(kind, [int(v) for v in axis], value=mid, key=f"sim3d_line_{kind}")
    sec_opts = ["Realization: facies", "Most likely facies"] + list(SIM3D_ATTRS)
    sec_attr = s[2].selectbox("Section shows", sec_opts, key="sim3d_sec_attr")

    left, right = st.columns([1, 1.25])
    left.plotly_chart(sim3d_map_fig(r, key, label, cmap, agg, rel_t, realization, pos, mask_low, min_beta,
                                    (kind, value)), width="stretch", key="sim3d_map")
    if sec_attr == "Most likely facies":
        skey, scmap, slabel = "ml_facies", None, "facies"
    elif sec_attr == "Realization: facies":
        skey, scmap, slabel = "real_facies", None, "facies"
    else:
        skey, scmap, slabel = SIM3D_ATTRS[sec_attr]
    fig = sim3d_section_fig(r, skey, slabel, scmap, realization, kind, value)
    if fig is not None:
        right.plotly_chart(fig, width="stretch", key="sim3d_sec")
    right.caption("Facies colours: " + " · ".join(f"<span style='color:{LITHO_COLORS[f]}'>■</span> {LITHO_LABELS[f]}"
                                                  for f in SIM_FACIES), unsafe_allow_html=True)

    blind = SIM3D_DIR / f"{choice}_blind.csv"
    if blind.is_file():
        with st.expander("Blind test at the held-out wells (same engine, scored against their logs)", expanded=True):
            info_p = SIM3D_DIR / f"{choice}_blind.json"
            if info_p.is_file():
                info = pd.DataFrame(json.loads(info_p.read_text(encoding="utf-8")))
                st.dataframe(info, hide_index=True, width="stretch")
            st.dataframe(pd.read_csv(blind), hide_index=True, width="stretch")
            st.caption("prior = wells only; post = wells + seismic. The surface columns compare the interpolated top "
                       "with the well's own pick (used for scoring only). A large difference means the window is "
                       "placed differently relative to the geology than in the training wells.")
    step_nav("sim3d")


PAGES = {
    "overview": st.Page(page_overview, title="Overview", icon="🏠", default=True, url_path="overview"),
    "data": st.Page(page_data, title="1 · Data", icon="🗂️", url_path="data"),
    "logs": st.Page(page_logs, title="2 · Well logs", icon="📈", url_path="logs"),
    "qc": st.Page(page_qc, title="3 · Well QC", icon="🧪", url_path="qc"),
    "seismic": st.Page(page_seismic, title="4 · Seismic", icon="🌊", url_path="seismic"),
    "tie": st.Page(page_tie, title="5 · Well tie", icon="🔗", url_path="tie"),
    "traverse": st.Page(page_traverse, title="6 · Seismic through wells", icon="🧭", url_path="traverse"),
    "rockphysics": st.Page(page_rockphysics, title="7 · Rock physics", icon="🪨", url_path="rockphysics"),
    "next": st.Page(page_next, title="8 · AVO classification", icon="🎯", url_path="next"),
    "sim": st.Page(page_simulation, title="9 · Prospect simulation", icon="🎲", url_path="simulation"),
    "sim3d": st.Page(page_sim3d, title="10 · 3D extrapolation", icon="🧊", url_path="simulation3d"),
}

if "well" not in st.session_state:
    st.session_state["well"] = "Poseidon-1"
status = step_status()
with st.sidebar:
    st.markdown("### Poseidon AVO")
    st.caption("Plover reservoir · Browse Basin")
navigation = st.navigation({
    "Start": [PAGES["overview"]],
    "Workflow": [PAGES[k] for k in ("data", "logs", "qc", "seismic", "tie", "traverse", "rockphysics", "next", "sim",
                                    "sim3d")],
})
with st.sidebar:
    st.divider()
    st.markdown("**Progress**")
    for key, label in (("data", "Data"), ("qc", "Well QC"), ("seismic", "Seismic cache"), ("tie", "Well tie"),
                       ("sim", "Prospect simulation"), ("sim3d", "3D extrapolation")):
        st.markdown(f"{STATUS_ICON[status[key]]} {label}")
    st.caption(f"Selected well: **{st.session_state['well']}**")
# Test hook: POSEIDON_APP_PAGE=<page key> renders one page directly (used by headless tests).
_forced_page = os.environ.get("POSEIDON_APP_PAGE")
if _forced_page:
    {"overview": page_overview, "data": page_data, "logs": page_logs, "qc": page_qc, "seismic": page_seismic,
     "tie": page_tie, "traverse": page_traverse, "rockphysics": page_rockphysics, "next": page_next,
     "sim": page_simulation, "sim3d": page_sim3d}[_forced_page]()
else:
    navigation.run()
