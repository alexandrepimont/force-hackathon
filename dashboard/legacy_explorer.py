"""Interactive local dashboard for synthetic/authorized AVO input examples."""

from __future__ import annotations

import sys
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard_data import (  # noqa: E402
    find_seismic_stacks,
    is_synthetic,
    read_gather,
    read_manifest,
    read_reservoir_model,
    read_seismic_geometry,
    read_seismic_section,
    read_seismic_traces,
    read_table,
    seismic_cache_ready,
    survey_outline,
    table_candidates,
    traverse_indices,
    wells_on_grid,
)

st.set_page_config(
    page_title="AVO Data Explorer",
    page_icon="🪨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
            .stApp {
                background: radial-gradient(
                    ellipse at 12% 0%, #142d3a 0%, #0c1520 42%, #09111a 100%
                );
            }
            [data-testid="stSidebar"] {
                background: rgba(13, 25, 36, 0.96);
                border-right: 1px solid #253a4a;
            }
            [data-testid="stMetric"] {
                background: linear-gradient(
                    145deg, rgba(28,48,64,.92), rgba(17,31,44,.92)
                );
                border: 1px solid #2a4355;
                border-radius: 16px;
                padding: 14px 18px;
            }
            [data-testid="stMetricLabel"] { color: #a9bdcb; }
            .hero { padding: 8px 0 16px 0; }
            .hero-kicker {
                color: #51d6bd;
                text-transform: uppercase;
                letter-spacing: .16em;
                font-size: .75rem;
                font-weight: 700;
            }
            .hero h1 { margin: .25rem 0; font-size: 2.45rem; color: #eef6fb; }
            .hero p { color: #a8bbc9; font-size: 1.05rem; margin-top: .2rem; }
            .data-badge {
                display: inline-block;
                border-radius: 999px;
                padding: 5px 12px;
                background: #4a2d16;
                color: #ffd08a;
                border: 1px solid #98622b;
                font-size: .8rem;
                font-weight: 700;
            }
            .data-badge.real {
                background: #123d36;
                border-color: #268b75;
                color: #8af0d5;
            }
            div[data-testid="stTabs"] button { color: #b7c8d4; }
            div[data-testid="stTabs"] button[aria-selected="true"] {
                color: #64e0ca;
            }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def cached_table(path_string: str, modified_ns: int):
    del modified_ns
    return read_table(Path(path_string))


@st.cache_data(show_spinner=False)
def cached_manifest(directory_string: str, modified_ns: int):
    del modified_ns
    return read_manifest(Path(directory_string))


@st.cache_data(show_spinner=False)
def cached_gather(path_string: str, modified_ns: int):
    del modified_ns
    return read_gather(Path(path_string))


@st.cache_data(show_spinner=False)
def cached_reservoir_model(path_string: str, modified_ns: int):
    del modified_ns
    return read_reservoir_model(Path(path_string))


@st.cache_data(show_spinner=False)
def cached_local_csv(path_string: str, modified_ns: int):
    del modified_ns
    return pd.read_csv(Path(path_string))


@st.cache_data(show_spinner=False)
def cached_seismic_geometry(path_string: str, modified_ns: int):
    del modified_ns
    return read_seismic_geometry(Path(path_string))


@st.cache_data(show_spinner=False, max_entries=24)
def cached_seismic_section(path_string: str, kind: str, value: float, modified_ns: int):
    del modified_ns
    return read_seismic_section(Path(path_string), kind, value)


def _cached_file(path: Path):
    return cached_table(str(path), path.stat().st_mtime_ns)


def _color_column(frame):
    for column in ("SYNTHETIC_LITHOLOGY", "LITHOLOGY", "FACIES", "FORMATION", "SYNTHETIC_FLUID"):
        if column in frame:
            return column
    return None


def _depth_options(frame):
    return [column for column in ("MD_M", "TVD_M", "TVDSS_M") if column in frame]


def _label(column: str) -> str:
    labels = {
        "DTC_US_FT": "Compressional slowness (µs/ft)",
        "DTS_US_FT": "Shear slowness (µs/ft)",
        "RHOB_G_CC": "Bulk density (g/cc)",
        "GR_API": "Gamma ray (API)",
        "PHIT_VV": "Total porosity (fraction)",
        "RT_OHM_M": "Deep resistivity (Ω·m)",
        "CALI_IN": "Caliper (in)",
        "DRHO_G_CC": "Density correction (g/cc)",
        "PEF_B_E": "Photoelectric factor (barn/e)",
        "NPHI_VV": "Neutron porosity (fraction)",
        "VP_M_S": "P-velocity (m/s)",
        "VS_M_S": "S-velocity (m/s)",
        "AI": "Acoustic impedance (m/s·g/cc)",
        "VPVS": "Vp/Vs ratio",
    }
    return labels.get(column, column.replace("_", " ").title())


with st.sidebar:
    st.markdown("### Data source")
    poseidon_wells = PROJECT_ROOT / "data" / "poseidon" / "dashboard"
    reservoir_examples = PROJECT_ROOT / "data" / "synthetic_reservoir"
    legacy_examples = PROJECT_ROOT / "data" / "volve" / "examples"
    default_directory = next(
        (path for path in (poseidon_wells, reservoir_examples) if path.is_dir()), legacy_examples
    )
    directory_input = st.text_input("Local data folder", str(default_directory))
    data_directory = Path(directory_input).expanduser()
    if not data_directory.is_absolute():
        data_directory = PROJECT_ROOT / data_directory
    data_directory = data_directory.resolve()
    well_candidates = table_candidates(data_directory, "well_logs")
    checkshot_candidates = table_candidates(data_directory, "checkshot")
    well_path = (
        st.selectbox(
            "Well-log table",
            well_candidates,
            format_func=lambda path: path.name,
        )
        if well_candidates
        else None
    )
    checkshot_path = (
        st.selectbox(
            "Checkshot table (optional)",
            checkshot_candidates,
            format_func=lambda path: path.name,
        )
        if checkshot_candidates
        else None
    )
    gather_candidates = sorted(data_directory.glob("*.npz"))
    gather_path = (
        st.selectbox(
            "Angle gather (optional)",
            gather_candidates,
            format_func=lambda path: path.name,
        )
        if gather_candidates
        else data_directory / "angle_gather_synthetic.npz"
    )
    reservoir_model_path = data_directory / "model" / "earth_model_synthetic.npz"

    st.caption(
        "Use a folder containing a well-log CSV, Parquet, or LAS file. "
        "Checkshot, gather, and reservoir model views are optional."
    )
    if well_path:
        st.success(f"Well logs: {well_path.name}")
    else:
        st.warning("No supported well-log CSV, Parquet, or LAS file found in this folder.")

if well_path is None:
    st.markdown(
        '<div class="hero"><div class="hero-kicker">AVO uncertainty workflow</div>'
        "<h1>Local Data Explorer</h1>"
        "<p>Inspect well-log, time-depth, QC, and seismic inputs.</p></div>",
        unsafe_allow_html=True,
    )
    st.info("Select a local folder with a well-log CSV, Parquet, or LAS file in the sidebar.")
    st.stop()

try:
    logs = _cached_file(well_path)
except Exception as exc:
    st.error(f"Could not read the selected well-log table ({type(exc).__name__}).")
    st.stop()

if logs.empty or "MD_M" not in logs:
    st.error("The selected well-log table is empty or does not include MD_M depth values.")
    st.stop()

manifest = cached_manifest(
    str(data_directory),
    max(
        (path.stat().st_mtime_ns for path in data_directory.glob("*manifest.json")),
        default=0,
    ),
)
synthetic = is_synthetic(logs, manifest)
well_id = (
    str(logs["WELL_ID"].dropna().iloc[0])
    if "WELL_ID" in logs and logs["WELL_ID"].notna().any()
    else "Local well"
)
selected_well_id = None
logs_view = logs
if "WELL_ID" in logs and logs["WELL_ID"].nunique(dropna=True) > 1:
    well_ids = sorted(logs["WELL_ID"].dropna().astype(str).unique())
    with st.sidebar:
        selected_well_id = st.selectbox("Well", well_ids, index=0)
    well_id = selected_well_id
    logs_view = logs.loc[logs["WELL_ID"].astype(str) == selected_well_id].copy()

depth_options = _depth_options(logs_view)
if not depth_options:
    depth_options = ["MD_M"]

with st.sidebar:
    st.markdown("### View controls")
    depth_column = st.selectbox("Depth reference", depth_options, index=0)
    min_depth = float(logs_view[depth_column].min())
    max_depth = float(logs_view[depth_column].max())
    depth_window = st.slider(
        f"Depth window ({depth_column.replace('_', ' ').lower()})",
        min_value=min_depth,
        max_value=max_depth,
        value=(min_depth, max_depth),
        step=max((max_depth - min_depth) / 500.0, 0.01),
        format="%.1f",
    )

filtered = logs_view.loc[logs_view[depth_column].between(*depth_window)].copy()
if filtered.empty:
    st.warning("No log samples fall inside the selected depth window.")
    st.stop()

field_tops_path = data_directory / "well_tops.csv"
field_wells_path = data_directory / "wells.csv"
field_tops = (
    cached_local_csv(str(field_tops_path), field_tops_path.stat().st_mtime_ns)
    if field_tops_path.is_file()
    else None
)
field_wells = (
    cached_local_csv(str(field_wells_path), field_wells_path.stat().st_mtime_ns)
    if field_wells_path.is_file()
    else None
)
well_tops_in_view = pd.DataFrame()
if field_tops is not None and depth_column in field_tops and "WELL_ID" in field_tops:
    well_tops_in_view = field_tops.loc[
        (field_tops["WELL_ID"].astype(str) == str(well_id))
        & field_tops[depth_column].between(*depth_window)
    ]
show_field_overview = not reservoir_model_path.is_file() and field_wells is not None

badge_class = "real" if not synthetic else ""
badge_text = (
    "LOCAL DATA · VERIFY PROVENANCE" if not synthetic else "SYNTHETIC EXAMPLE · NOT FIELD DATA"
)
dataset_name = manifest.get("dataset") if isinstance(manifest.get("dataset"), str) else ""
st.markdown(
    f'<div class="hero"><div class="hero-kicker">Lithology · Fluids · Uncertainty</div>'
    f"<h1>AVO Data Explorer</h1><p>{escape(well_id)} &nbsp; "
    f'<span class="data-badge {badge_class}">{badge_text}</span>'
    f'{"<br>" + escape(dataset_name) if dataset_name else ""}</p></div>',
    unsafe_allow_html=True,
)

if synthetic:
    st.warning(
        "Synthetic scenario for software testing only. It is not derived from or "
        "representative of real Volve measurements."
    )

qc_count = (
    int(filtered["QC_BAD_HOLE"].fillna(False).astype(bool).sum())
    if "QC_BAD_HOLE" in filtered
    else 0
)
missing_percent = float(filtered.isna().mean().mean() * 100.0)
metric_columns = st.columns(4)
metric_columns[0].metric("Samples in view", f"{len(filtered):,}")
metric_columns[1].metric("Depth range", f"{depth_window[0]:,.1f}–{depth_window[1]:,.1f} m")
metric_columns[2].metric(
    "QC flagged", f"{qc_count:,}", help="Rows marked QC_BAD_HOLE in the source table."
)
metric_columns[3].metric("Missing values", f"{missing_percent:.2f}%")

TIE_DIR = PROJECT_ROOT / "outputs" / "well_ties"
QC_DIR = PROJECT_ROOT / "outputs" / "well_qc"
STACK_COLORS = {"near": "#4c9be8", "mid": "#43b581", "far": "#e8833a"}
DEFAULT_TRAVERSE = ["Kronos-1", "Poseidon-2", "Boreas-1", "Poseidon-1", "Poseidon-North-1"]


def _step(axis_values, max_points: int) -> int:
    return max(1, int(np.ceil(len(axis_values) / max_points)))


@st.cache_data(show_spinner=False)
def cached_wells_on_grid(path_string: str, modified_ns: int, wells: pd.DataFrame):
    geometry = cached_seismic_geometry(path_string, modified_ns)
    return wells_on_grid(geometry, wells).dropna(subset=["INLINE"])


@st.cache_data(show_spinner=False, max_entries=12)
def cached_traverse(path_string: str, modified_ns: int, points: tuple):
    geometry = cached_seismic_geometry(path_string, modified_ns)
    return traverse_indices(list(points), geometry["ilines"], geometry["xlines"], geometry)


@st.cache_data(show_spinner=False, max_entries=12)
def cached_traverse_section(path_string: str, modified_ns: int, points: tuple):
    path = cached_traverse(path_string, modified_ns, points)
    return read_seismic_traces(Path(path_string), path["i"], path["j"])


@st.cache_data(show_spinner=False)
def load_tie(well: str, modified_ns: int):
    with np.load(TIE_DIR / f"{well}.npz") as data:
        return {key: data[key] for key in data.files}


def _tie(well: str):
    path = TIE_DIR / f"{well}.npz"
    return load_tie(well, path.stat().st_mtime_ns) if path.is_file() else None


def basemap(geometry, positions, lines=(), fill_slice=False, title="Where the section is"):
    ox, oy = survey_outline(geometry)
    fig = go.Figure(
        go.Scatter(
            x=ox, y=oy, mode="lines", fill="toself",
            fillcolor="rgba(245,183,0,0.35)" if fill_slice else "rgba(245,215,110,0.10)",
            line={"color": "#b58b16", "width": 1.5}, hoverinfo="skip", name="seismic",
        )
    )
    for x, y, label in lines:
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line={"color": "#ff4b4b", "width": 4},
                                 name=label, hoverinfo="name"))
    if field_wells is not None:
        fig.add_trace(
            go.Scatter(
                x=field_wells["X_M"], y=field_wells["Y_M"], mode="markers+text", text=field_wells["WELL_ID"],
                textposition="top center", textfont={"size": 10},
                marker={"size": 9, "color": "#f7d56d",
                        "line": {"color": ["#ff4b4b" if w == well_id else "#172332" for w in field_wells["WELL_ID"]],
                                 "width": [3 if w == well_id else 1 for w in field_wells["WELL_ID"]]}},
                hoverinfo="text", name="wells",
            )
        )
    pad = 2500
    fig.update_layout(
        title={"text": title, "font": {"size": 13}}, template="plotly_dark", height=430, showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)", margin={"l": 10, "r": 10, "t": 40, "b": 10},
        xaxis={"range": [ox.min() - pad, ox.max() + pad], "tickformat": ",.0f", "title": "Easting (m)"},
        yaxis={"range": [oy.min() - pad, oy.max() + pad], "tickformat": ",.0f", "scaleanchor": "x",
               "title": "Northing (m)"},
    )
    return fig


def _heatmap(section_2d, x_axis, y_axis, clip, showscale, hover):
    return go.Heatmap(x=x_axis, y=y_axis, z=np.round(section_2d), zmin=-clip, zmax=clip, zmid=0,
                      colorscale="RdBu_r", showscale=showscale, colorbar={"title": "Amplitude"}, hovertemplate=hover)


@st.fragment
def render_seismic_viewer():
    ready = {name: path for name, path in seismic_stacks.items() if seismic_cache_ready(path)}
    missing = [name for name in seismic_stacks if name not in ready]
    st.subheader("3D seismic: near, mid and far angle stacks")
    if missing:
        st.info(
            "The fast display cache is missing for: " + ", ".join(missing) + ". Build it once with "
            "`python scripts/prepare_poseidon_dashboard.py` (about 1 minute per stack)."
        )
    if not ready:
        return
    first = next(iter(ready.values()))
    mtime = first.stat().st_mtime_ns
    geometry = cached_seismic_geometry(str(first), mtime)
    ilines, xlines, samples = geometry["ilines"], geometry["xlines"], geometry["samples"]
    positions = (
        cached_wells_on_grid(str(first), mtime, field_wells[["WELL_ID", "X_M", "Y_M"]])
        if field_wells is not None and {"X_M", "Y_M"}.issubset(field_wells.columns)
        else pd.DataFrame(columns=["WELL_ID", "INLINE", "CROSSLINE"])
    )
    selected = positions.loc[positions["WELL_ID"] == well_id]
    default_il = int(selected["INLINE"].iloc[0]) if len(selected) else int(ilines[len(ilines) // 2])
    default_xl = int(selected["CROSSLINE"].iloc[0]) if len(selected) else int(xlines[len(xlines) // 2])

    controls = st.columns([1.6, 1.4, 1.0, 1.0])
    view = controls[0].radio("View", ["Well to well", "Inline", "Crossline", "Time slice", "3D"], horizontal=True,
                             key="seis_view")
    stack_options = [*ready] + (["near | mid | far"] if len(ready) > 1 and view != "3D" else [])
    stack_choice = controls[1].radio("Stack", stack_options, horizontal=True, key=f"seis_stack_{view == '3D'}")
    clip_pct = controls[2].slider("Colour clip (percentile)", 90.0, 99.9, 98.0, 0.1, key="seis_clip")
    projection = controls[3].slider("Show wells within (lines)", 0, 200, 25, 5, key="seis_proj",
                                    disabled=view not in ("Inline", "Crossline"))
    names = list(ready) if stack_choice == "near | mid | far" else [stack_choice]

    if view == "Well to well":
        render_traverse(ready, names, geometry, positions, samples, clip_pct, mtime)
        return
    if view == "3D":
        render_3d(ready[names[0]], geometry, positions, samples, clip_pct, default_il, default_xl)
        return

    if view == "Inline":
        kind, value = "inline", st.slider("Inline", int(ilines[0]), int(ilines[-1]), default_il, key=f"seis_il_{well_id}")
    elif view == "Crossline":
        kind, value = "crossline", st.slider("Crossline", int(xlines[0]), int(xlines[-1]), default_xl,
                                             key=f"seis_xl_{well_id}")
    else:
        kind, value = "time", float(st.select_slider(
            "Time slice (ms TWT)", options=[float(t) for t in samples],
            value=float(samples[np.argmin(np.abs(samples - 3150.0))]), key="seis_time"))
    if kind != "time":
        t_window = st.slider("Time window (ms TWT)", float(samples[0]), float(samples[-1]),
                             (float(samples[0]), float(samples[-1])), float(samples[1] - samples[0]), key="seis_twin")

    sections = {n: cached_seismic_section(str(ready[n]), kind, value, ready[n].stat().st_mtime_ns) for n in names}
    clip = max(float(np.percentile(np.abs(sections[names[0]]), clip_pct)), 1e-6)
    figure = make_subplots(rows=1, cols=len(names), shared_yaxes=True, horizontal_spacing=0.02,
                           subplot_titles=[f"{n} stack" for n in names])
    for col, name in enumerate(names, start=1):
        section = sections[name]
        if kind == "time":
            sx, sy = _step(xlines, 700 // len(names) + 250), _step(ilines, 700)
            figure.add_trace(_heatmap(section[::sy, ::sx], xlines[::sx], ilines[::sy], clip, col == len(names),
                                      "Crossline %{x}<br>Inline %{y}<br>Amplitude %{z:.0f}<extra></extra>"), 1, col)
            figure.add_trace(go.Scatter(x=positions["CROSSLINE"], y=positions["INLINE"], mode="markers+text",
                                        text=positions["WELL_ID"], textposition="top center",
                                        marker={"size": 9, "color": "#f7d56d", "line": {"color": "black", "width": 1}},
                                        textfont={"color": "black"}, showlegend=False, hoverinfo="text"), 1, col)
        else:
            lateral = xlines if kind == "inline" else ilines
            t_mask = (samples >= t_window[0]) & (samples <= t_window[1])
            sx = _step(lateral, 900 // len(names) + 150)
            label = "Crossline" if kind == "inline" else "Inline"
            figure.add_trace(_heatmap(section[::sx][:, t_mask].T, lateral[::sx], samples[t_mask], clip,
                                      col == len(names),
                                      f"{label} %{{x}}<br>TWT %{{y:.0f}} ms<br>Amplitude %{{z:.0f}}<extra></extra>"),
                             1, col)
            along, across = ("CROSSLINE", "INLINE") if kind == "inline" else ("INLINE", "CROSSLINE")
            for _, well in positions.loc[(positions[across] - value).abs() <= projection].iterrows():
                figure.add_vline(x=float(well[along]), line={"color": "black", "width": 2 if well["WELL_ID"] == well_id else 1},
                                 row=1, col=col)
                figure.add_annotation(x=float(well[along]), y=float(t_window[0]), text=escape(str(well["WELL_ID"])),
                                      showarrow=False, yanchor="bottom", font={"size": 10, "color": "white"},
                                      bgcolor="rgba(0,0,0,.6)", row=1, col=col)
    if kind == "time":
        figure.update_yaxes(title_text="Inline", row=1, col=1)
        figure.update_xaxes(title_text="Crossline")
    else:
        figure.update_yaxes(autorange="reversed", title_text="TWT (ms)", row=1, col=1)
        figure.update_xaxes(title_text="Crossline" if kind == "inline" else "Inline")
    figure.update_layout(template="plotly_dark", height=620, paper_bgcolor="rgba(0,0,0,0)",
                         margin={"l": 55, "r": 20, "t": 45, "b": 50})

    i = int(np.searchsorted(ilines, value)) if kind == "inline" else None
    j = int(np.searchsorted(xlines, value)) if kind == "crossline" else None
    if kind == "inline":
        lines = [(geometry["cdp_x"][i, :], geometry["cdp_y"][i, :], f"inline {value}")]
    elif kind == "crossline":
        lines = [(geometry["cdp_x"][:, j], geometry["cdp_y"][:, j], f"crossline {value}")]
    else:
        lines = []
    left, right = st.columns([3, 1.15])
    left.plotly_chart(figure, width="stretch")
    right.plotly_chart(basemap(geometry, positions, lines, fill_slice=kind == "time",
                               title=f"Time slice {value:.0f} ms" if kind == "time" else f"{view} {value}"),
                       width="stretch")
    st.caption(
        "All panels use the same colour scale (set from the first panel), so brightness differences between near, "
        "mid and far are real amplitude-versus-angle changes. The map shows where the section lies (red)."
    )


def render_traverse(ready, names, geometry, positions, samples, clip_pct, mtime):
    choices = list(positions["WELL_ID"])
    chosen = st.multiselect("Wells along the line, in order (add, remove or reorder)", choices,
                            default=[w for w in DEFAULT_TRAVERSE if w in choices], key="seis_traverse")
    if len(chosen) < 2:
        st.info("Choose at least two wells.")
        return
    opts = st.columns([2, 1, 1])
    t_window = opts[0].slider("Time window (ms TWT)", float(samples[0]), float(samples[-1]), (2900.0, 3500.0),
                              float(samples[1] - samples[0]), key="trav_twin")
    show_syn = opts[1].checkbox("Synthetic at wells (automated tie)", value=True, key="trav_syn")
    show_tops = opts[2].checkbox("Tops (tied time)", value=True, key="trav_tops")
    rows = positions.set_index("WELL_ID").loc[chosen]
    points = tuple((float(r.INLINE), float(r.CROSSLINE)) for r in rows.itertuples())
    path = cached_traverse(str(next(iter(ready.values()))), mtime, points)
    dist_km = path["distance_m"] / 1000
    well_x = dist_km[path["vertex"]]
    t_mask = (samples >= t_window[0]) & (samples <= t_window[1])
    sections = {n: cached_traverse_section(str(ready[n]), ready[n].stat().st_mtime_ns, points) for n in names}
    clip = max(float(np.percentile(np.abs(sections[names[0]][:, t_mask]), clip_pct)), 1e-6)
    figure = make_subplots(rows=len(names), cols=1, shared_xaxes=True, vertical_spacing=0.04,
                           subplot_titles=[f"{n} stack" for n in names])
    ties = {w: _tie(w) for w in chosen}
    palette = px.colors.qualitative.Set3
    for row, name in enumerate(names, start=1):
        sx = _step(dist_km, 1100)
        figure.add_trace(_heatmap(sections[name][::sx][:, t_mask].T, dist_km[::sx], samples[t_mask], clip, row == 1,
                                  "Distance %{x:.2f} km<br>TWT %{y:.0f} ms<br>Amplitude %{z:.0f}<extra></extra>"),
                         row, 1)
        k = list(STACK_COLORS).index(name) if name in STACK_COLORS else 0
        width = 0.04 * dist_km[-1]
        surfaces = {}
        for w, x in zip(chosen, well_x):
            figure.add_vline(x=float(x), line={"color": "black", "width": 1}, row=row, col=1)
            if row == 1:
                figure.add_annotation(x=float(x), y=float(t_window[0]), text=escape(w), showarrow=False, yanchor="bottom",
                                      font={"size": 10, "color": "white"}, bgcolor="rgba(0,0,0,.65)", row=row, col=1)
            tie = ties.get(w)
            if tie is None:
                continue
            if show_syn:
                win = tie["win_after"] & (tie["t_ms"] >= t_window[0]) & (tie["t_ms"] <= t_window[1])
                syn = tie["syn_after"][k] * tie["scales"][k]
                figure.add_trace(go.Scatter(x=x + width * np.clip(syn[win] / clip, -1.5, 1.5), y=tie["t_ms"][win],
                                            mode="lines", line={"color": "#f7d56d", "width": 2},
                                            name=f"{w} synthetic", showlegend=False,
                                            hovertemplate=f"{w} synthetic ({name})<extra></extra>"), row, 1)
            if show_tops:
                for s, t in zip(tie["tops_names"], tie["tops_twt_ms"]):
                    if t_window[0] <= t <= t_window[1]:
                        surfaces.setdefault(str(s), []).append((float(x), float(t)))
        for n_s, (surface, pts) in enumerate(surfaces.items()):
            pts = sorted(pts)
            figure.add_trace(go.Scatter(x=[p[0] for p in pts], y=[p[1] for p in pts], mode="lines+markers",
                                        line={"color": palette[n_s % len(palette)], "width": 1.5, "dash": "dot"},
                                        marker={"size": 8}, name=surface, showlegend=row == 1,
                                        hovertemplate=f"{surface}<br>%{{y:.0f}} ms<extra></extra>"), row, 1)
        figure.update_yaxes(autorange="reversed", title_text="TWT (ms)", row=row, col=1)
    figure.update_xaxes(title_text="Distance along line (km)", row=len(names), col=1)
    figure.update_layout(template="plotly_dark", height=420 * len(names) + 80, paper_bgcolor="rgba(0,0,0,0)",
                         margin={"l": 55, "r": 20, "t": 45, "b": 50}, legend={"orientation": "h", "y": -0.08})
    left, right = st.columns([3, 1.15])
    left.plotly_chart(figure, width="stretch")
    right.plotly_chart(basemap(geometry, positions, [(path["x"], path["y"], "line")], title="Well-to-well line"),
                       width="stretch")
    untied = [w for w in chosen if ties.get(w) is None]
    st.caption(
        "Yellow wiggles: synthetic from the automated well tie at each well (same amplitude scale as the seismic via "
        "the global per-stack scale). Dotted lines connect formation tops in tied time. "
        + (f"No tie available for: {', '.join(untied)}." if untied else "")
    )


def render_3d(segy_path, geometry, positions, samples, clip_pct, default_il, default_xl):
    ilines, xlines = geometry["ilines"], geometry["xlines"]
    c = st.columns(4)
    il = c[0].slider("Inline", int(ilines[0]), int(ilines[-1]), default_il, key=f"d3_il_{well_id}")
    xl = c[1].slider("Crossline", int(xlines[0]), int(xlines[-1]), default_xl, key=f"d3_xl_{well_id}")
    t_slice = float(c[2].select_slider("Time slice (ms)", options=[float(t) for t in samples],
                                       value=float(samples[np.argmin(np.abs(samples - 3400.0))]), key="d3_t"))
    vscale = c[3].slider("Vertical scale", 0.1, 2.0, 0.6, 0.05, key="d3_vscale")
    mt = segy_path.stat().st_mtime_ns
    inl = cached_seismic_section(str(segy_path), "inline", il, mt)
    xln = cached_seismic_section(str(segy_path), "crossline", xl, mt)
    tsl = cached_seismic_section(str(segy_path), "time", t_slice, mt)
    clip = max(float(np.percentile(np.abs(inl), clip_pct)), 1e-6)
    cx, cy = float(np.nanmean(geometry["cdp_x"])), float(np.nanmean(geometry["cdp_y"]))
    i, j = int(np.searchsorted(ilines, il)), int(np.searchsorted(xlines, xl))
    st_t = _step(samples, 160)
    tt = samples[::st_t]
    surfaces = []
    sj = _step(xlines, 220)
    x_line, y_line = geometry["cdp_x"][i, ::sj] - cx, geometry["cdp_y"][i, ::sj] - cy
    surfaces.append((np.tile(x_line, (len(tt), 1)), np.tile(y_line, (len(tt), 1)), -np.tile(tt[:, None], (1, len(x_line))),
                     inl[::sj, ::st_t].T))
    si = _step(ilines, 220)
    x_line, y_line = geometry["cdp_x"][::si, j] - cx, geometry["cdp_y"][::si, j] - cy
    surfaces.append((np.tile(x_line, (len(tt), 1)), np.tile(y_line, (len(tt), 1)), -np.tile(tt[:, None], (1, len(x_line))),
                     xln[::si, ::st_t].T))
    si2, sj2 = _step(ilines, 200), _step(xlines, 200)
    gx, gy = geometry["cdp_x"][::si2, ::sj2] - cx, geometry["cdp_y"][::si2, ::sj2] - cy
    surfaces.append((gx, gy, np.full(gx.shape, -t_slice), tsl[::si2, ::sj2]))
    fig = go.Figure()
    for k, (X, Y, Z, C) in enumerate(surfaces):
        fig.add_trace(go.Surface(x=X, y=Y, z=Z, surfacecolor=C, colorscale="RdBu_r", cmin=-clip, cmax=clip,
                                 showscale=k == 0, colorbar={"title": "Amplitude"}, hoverinfo="skip"))
    for _, w in positions.iterrows():
        wi, wj = int(np.searchsorted(ilines, w["INLINE"])), int(np.searchsorted(xlines, w["CROSSLINE"]))
        x0, y0 = geometry["cdp_x"][wi, wj] - cx, geometry["cdp_y"][wi, wj] - cy
        fig.add_trace(go.Scatter3d(x=[x0, x0], y=[y0, y0], z=[-samples[0], -samples[-1]], mode="lines",
                                   line={"color": "#f7d56d" if w["WELL_ID"] != well_id else "#ff4b4b", "width": 6},
                                   name=w["WELL_ID"], hoverinfo="name"))
        fig.add_trace(go.Scatter3d(x=[x0], y=[y0], z=[-samples[0] + 40], mode="text", text=[w["WELL_ID"]],
                                   textfont={"color": "white", "size": 11}, showlegend=False, hoverinfo="skip"))
        tie = _tie(w["WELL_ID"])
        if tie is not None:
            ok = (tie["tops_twt_ms"] >= samples[0]) & (tie["tops_twt_ms"] <= samples[-1])
            fig.add_trace(go.Scatter3d(x=np.full(ok.sum(), x0), y=np.full(ok.sum(), y0), z=-tie["tops_twt_ms"][ok],
                                       mode="markers", marker={"size": 4, "color": "white"},
                                       text=tie["tops_names"][ok], hoverinfo="text", showlegend=False))
    dx = float(np.nanmax(geometry["cdp_x"]) - np.nanmin(geometry["cdp_x"]))
    dy = float(np.nanmax(geometry["cdp_y"]) - np.nanmin(geometry["cdp_y"]))
    m = max(dx, dy)
    fig.update_layout(
        template="plotly_dark", height=720, paper_bgcolor="rgba(0,0,0,0)", margin={"l": 0, "r": 0, "t": 30, "b": 0},
        scene={"aspectmode": "manual", "aspectratio": {"x": dx / m, "y": dy / m, "z": vscale},
               "xaxis_title": "East (m, from centre)", "yaxis_title": "North (m, from centre)",
               "zaxis_title": "-TWT (ms)", "camera": {"eye": {"x": -1.3, "y": -1.5, "z": 0.9}}},
        legend={"orientation": "h"},
    )
    left, right = st.columns([3, 1.15])
    left.plotly_chart(fig, width="stretch")
    right.plotly_chart(basemap(geometry, positions,
                               [(geometry["cdp_x"][i, :], geometry["cdp_y"][i, :], f"inline {il}"),
                                (geometry["cdp_x"][:, j], geometry["cdp_y"][:, j], f"crossline {xl}")],
                               title=f"Inline {il}, crossline {xl}, slice {t_slice:.0f} ms"), width="stretch")
    st.caption("Drag to rotate, scroll to zoom. Coordinates are metres relative to the cube centre (avoids float "
               "jitter); vertical scale only changes the display. White dots on wells are formation tops in tied time.")



def render_well_qc_tie():
    import json

    import yaml

    st.subheader(f"Well QC and automated well tie: {well_id}")
    tie_summary_path, qc_summary_path = TIE_DIR / "summary.csv", QC_DIR / "summary.csv"
    if not tie_summary_path.is_file() or not qc_summary_path.is_file():
        st.info("Run `python scripts/run_well_qc.py` and then `python scripts/run_well_tie.py` to create the QC and tie.")
        return
    tie_summary, qc_summary = pd.read_csv(tie_summary_path), pd.read_csv(qc_summary_path)
    sub = st.tabs(["Well tie: what changed", "Well QC", "All wells and gate G2"])

    with sub[0]:
        row = tie_summary.loc[tie_summary["well"] == well_id]
        if row.empty or row["status"].iloc[0] != "tied":
            st.warning(f"No tie for {well_id}: " + (row["status"].iloc[0] if len(row) else "not processed"))
        else:
            r = row.iloc[0]
            meta = json.loads((TIE_DIR / f"{well_id}.json").read_text(encoding="utf-8"))
            tie = _tie(well_id)
            stacks = [str(s) for s in tie["stacks"]]
            if str(r.get("qc_flags", "ok")) != "ok":
                st.warning(f"Review this tie: {r['qc_flags']}")
            else:
                st.success("Automated tie passed all QC checks.")
            m = st.columns(5)
            src = f"{r['td_source']}" + (f" ({r['td_distance_km']} km away)" if r["td_transferred"] else " (own)")
            m[0].metric("Time-depth from", src)
            m[1].metric("Bulk shift", f"{r['shift_ms']:+.0f} ms", help=f"Allowed range +/-{r['max_shift_ms']:.0f} ms")
            m[2].metric("Stretch", f"{r['stretch_pct']:+.1f} %")
            m[3].metric("Mean correlation", f"{r['corr_after_mean']:.2f}", f"{r['corr_after_mean'] - r['corr_before_mean']:+.2f}")
            m[4].metric("Role", "HELD OUT (blind)" if str(r["held_out"]) == "True" else "training")

            t = tie["t_ms"]
            win = tie["win_after"]
            lo = max(float(tie["log_twt_ms"].min()) - 80, float(t[0]))
            hi = min(float(tie["log_twt_ms"].max()) + 40, float(t[-1]))
            fig = make_subplots(rows=1, cols=2 + len(stacks), shared_yaxes=True, horizontal_spacing=0.02,
                                subplot_titles=["AI", "Vp/Vs"] + [
                                    f"{s}: r {r[f'corr_before_{s}']:.2f} -> {r[f'corr_after_{s}']:.2f}" for s in stacks])
            fig.add_trace(go.Scatter(x=tie["ai"], y=tie["log_twt_ms"], line={"color": "#ef6c00", "width": 1},
                                     showlegend=False), 1, 1)
            fig.add_trace(go.Scatter(x=tie["vpvs"], y=tie["log_twt_ms"], line={"color": "#00acc1", "width": 1},
                                     showlegend=False), 1, 2)
            for k, s in enumerate(stacks):
                obs = tie["obs"][k] / tie["scales"][k]
                norm = float(np.abs(obs[win]).max()) or 1.0
                fig.add_trace(go.Scatter(x=obs / norm, y=t, line={"color": "white", "width": 1.5}, name="seismic",
                                         showlegend=k == 0), 1, 3 + k)
                fig.add_trace(go.Scatter(x=np.where(tie["win_before"], tie["syn_before"][k], np.nan) / norm, y=t,
                                         line={"color": "#9e9e9e", "width": 1, "dash": "dash"},
                                         name="synthetic before (transferred/own TD only)", showlegend=k == 0), 1, 3 + k)
                fig.add_trace(go.Scatter(x=np.where(win, tie["syn_after"][k], np.nan) / norm, y=t,
                                         line={"color": STACK_COLORS.get(s, "#ff4b4b"), "width": 2.5},
                                         name="synthetic after automated tie", showlegend=k == 0), 1, 3 + k)
            for name, tt in zip(tie["tops_names"], tie["tops_twt_ms"]):
                if lo <= tt <= hi:
                    fig.add_hline(y=float(tt), line={"color": "#f7d56d", "width": 1, "dash": "dot"})
                    fig.add_annotation(x=1.0, xref="paper", y=float(tt), text=escape(str(name)), showarrow=False,
                                       xanchor="left", font={"size": 9, "color": "#f7d56d"})
            fig.update_yaxes(range=[hi, lo], title_text="TWT (ms)", row=1, col=1)
            fig.update_layout(template="plotly_dark", height=700, paper_bgcolor="rgba(0,0,0,0)",
                              margin={"l": 55, "r": 160, "t": 50, "b": 40}, legend={"orientation": "h", "y": -0.06})
            st.plotly_chart(fig, width="stretch")

            left, right = st.columns([1.2, 1])
            td = pd.read_csv(PROJECT_ROOT / "data" / "interim" / "poseidon_td" / f"{well_id}.csv").dropna()
            td = td[td["twt_final_ms"].between(lo - 200, hi + 200)]
            td_fig = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=["Time-depth", "Change (ms)"])
            td_fig.add_trace(go.Scatter(x=td["twt_initial_ms"], y=td["tvdss"], name="before (source curve)",
                                        line={"color": "#9e9e9e", "dash": "dash"}), 1, 1)
            td_fig.add_trace(go.Scatter(x=td["twt_final_ms"], y=td["tvdss"], name="after automated tie",
                                        line={"color": "#ff4b4b"}), 1, 1)
            td_fig.add_trace(go.Scatter(x=td["twt_final_ms"] - td["twt_initial_ms"], y=td["tvdss"], showlegend=False,
                                        line={"color": "#ff4b4b"}), 1, 2)
            td_fig.update_yaxes(autorange="reversed", title_text="TVDSS (m)", row=1, col=1)
            td_fig.update_xaxes(title_text="TWT (ms)", row=1, col=1)
            td_fig.update_layout(template="plotly_dark", height=420, paper_bgcolor="rgba(0,0,0,0)",
                                 legend={"orientation": "h", "y": -0.2}, margin={"l": 55, "r": 10, "t": 40, "b": 40})
            left.plotly_chart(td_fig, width="stretch")
            changes = pd.DataFrame({
                "stack": stacks,
                "corr before": [r[f"corr_before_{s}"] for s in stacks],
                "corr after": [r[f"corr_after_{s}"] for s in stacks],
                "wavelet (Hz / phase)": [f"{meta['wavelets'][s]['freq_hz']:.0f} / {meta['wavelets'][s]['phase_deg']:+.0f}"
                                         for s in stacks],
            })
            right.markdown("**What the automated tie changed**")
            right.dataframe(changes, hide_index=True, width="stretch")
            tdm = meta["time_depth"]
            right.markdown(
                f"- Time-depth source: **{tdm['donor']}** "
                + (f"(transferred, {tdm['distance_km']} km)" if tdm["transferred"] else "(own checkshot-calibrated curve)")
                + f"\n- Shift {r['shift_ms']:+.0f} ms (limit +/-{r['max_shift_ms']:.0f}), stretch {r['stretch_pct']:+.1f} %"
                + f"\n- Trace: {meta['trace_selection']}"
                + f"\n- Logs: {meta['log_conditioning']['interpolated_samples']} flagged/gap samples interpolated, "
                  f"Vs estimated for {meta['log_conditioning']['vs_estimated_fraction']:.0%} of the interval"
                + f"\n- Polarity: {meta['polarity']}"
            )
            with right.expander("Rejected time-depth donors and raw metadata"):
                st.json(meta)

    with sub[1]:
        q = qc_summary.loc[qc_summary["well"] == well_id]
        image = QC_DIR / f"{well_id}.png"
        if image.is_file():
            st.image(str(image), width="stretch")
        if len(q):
            st.dataframe(q.T.rename(columns={q.index[0]: well_id}).astype(str), width="stretch")
        meta_path = PROJECT_ROOT / "data" / "interim" / "poseidon_logs" / f"{well_id}.meta.json"
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            st.markdown("**Curve provenance** (normalised curve -> source mnemonic, unit, conversion)")
            st.dataframe(pd.DataFrame(meta["provenance"]["curves"]).T, width="stretch")
            st.caption(f"Source file: {meta['provenance']['file']}. Depth: {meta['depth']['method']}, max error at TD "
                       f"{meta['depth']['max_error_at_td_m']} m. Fluid is not labelled; only a hydrocarbon indicator "
                       "(Archie Sw + neutron-density crossover) is reported.")

    with sub[2]:
        st.markdown("**Automated tie, all wells**")
        cols = [c for c in ["well", "status", "held_out", "td_source", "td_distance_km", "shift_ms", "stretch_pct",
                            "corr_before_mean", "corr_after_mean", "corr_after_near", "corr_after_mid",
                            "corr_after_far", "qc_flags"] if c in tie_summary]
        st.dataframe(tie_summary[cols], hide_index=True, width="stretch")
        cal_path = PROJECT_ROOT / "configs" / "calibration.yaml"
        if cal_path.is_file():
            cal = yaml.safe_load(cal_path.read_text())
            st.markdown("**Global calibration from training wells** "
                        f"({', '.join(cal['training_wells'])}; held out: {', '.join(cal['held_out_wells'])})")
            st.dataframe(pd.DataFrame(cal["stacks"]).T, width="stretch")
            st.caption(f"Polarity check (mean r): {cal['polarity_check_mean_corr']}")
        st.markdown("**Well QC summary**")
        st.dataframe(qc_summary, hide_index=True, width="stretch")
        log = PROJECT_ROOT / "outputs" / "decision_log.md"
        if log.is_file():
            with st.expander("Decision log (gate G2)", expanded=False):
                st.markdown(log.read_text(encoding="utf-8"))


seismic_stacks = find_seismic_stacks(data_directory)

tabs = st.tabs(
    [
        "Well logs",
        "Field overview" if show_field_overview else "Reservoir model",
        "Rock-property crossplots",
        "Seismic" if seismic_stacks else "Seismic gather",
        "Well QC & tie",
        "Data preview",
    ]
)

with tabs[1]:
    if show_field_overview:
        st.subheader("Well locations and formation tops")
        st.caption("Coordinates in GDA94 / MGA zone 51. Depths are TVDSS (m below sea level).")
        top_options = (
            sorted(field_tops["SURFACE"].dropna().unique())
            if field_tops is not None and "SURFACE" in field_tops
            else []
        )
        default_top = next(
            (name for name in top_options if "Top Reservoir" in name),
            next((name for name in top_options if "Plover" in name), None),
        )
        map_top = (
            st.selectbox(
                "Surface shown on the map",
                top_options,
                index=top_options.index(default_top) if default_top in top_options else 0,
            )
            if top_options
            else None
        )
        map_wells = field_wells.copy()
        if map_top is not None and "TVDSS_M" in field_tops:
            top_depths = (
                field_tops.loc[field_tops["SURFACE"] == map_top]
                .groupby("WELL_ID")["TVDSS_M"]
                .min()
            )
            map_wells["TOP_TVDSS_M"] = map_wells["WELL_ID"].map(top_depths)
        else:
            map_wells["TOP_TVDSS_M"] = np.nan
        map_wells["LABEL"] = [
            f"{well}<br>{depth:,.0f} m" if pd.notna(depth) else f"{well}<br>no pick"
            for well, depth in zip(map_wells["WELL_ID"], map_wells["TOP_TVDSS_M"])
        ]
        well_map = go.Figure(
            go.Scatter(
                x=map_wells["X_M"],
                y=map_wells["Y_M"],
                mode="markers+text",
                text=map_wells["LABEL"],
                textposition="top center",
                marker={
                    "size": 14,
                    "color": map_wells["TOP_TVDSS_M"],
                    "colorscale": "Viridis_r",
                    "showscale": bool(map_wells["TOP_TVDSS_M"].notna().any()),
                    "colorbar": {"title": "TVDSS (m)"},
                    "line": {
                        "color": [
                            "#ff776e" if well == well_id else "#172332"
                            for well in map_wells["WELL_ID"]
                        ],
                        "width": [3 if well == well_id else 1 for well in map_wells["WELL_ID"]],
                    },
                },
                hovertemplate="%{text}<br>X %{x:,.0f} m<br>Y %{y:,.0f} m<extra></extra>",
            )
        )
        well_map.update_layout(
            title=f"{map_top or 'Wells'} (selected well outlined in red)",
            xaxis_title="Easting (m)",
            yaxis_title="Northing (m)",
            yaxis={"scaleanchor": "x", "scaleratio": 1},
            template="plotly_dark",
            height=560,
            margin={"l": 30, "r": 20, "t": 55, "b": 45},
        )

        correlation = go.Figure()
        if field_tops is not None and "TVDSS_M" in field_tops:
            well_order = (
                map_wells.sort_values("X_M")["WELL_ID"].tolist()
                if "X_M" in map_wells
                else sorted(field_tops["WELL_ID"].unique())
            )
            palette = px.colors.qualitative.Set3 + px.colors.qualitative.Pastel
            for index, (surface, picks) in enumerate(
                field_tops.sort_values("TVDSS_M").groupby("SURFACE", sort=False)
            ):
                picks = picks.groupby("WELL_ID")["TVDSS_M"].min().reindex(well_order)
                correlation.add_trace(
                    go.Scatter(
                        x=well_order,
                        y=picks,
                        mode="lines+markers",
                        name=surface,
                        connectgaps=False,
                        line={"color": palette[index % len(palette)], "width": 2},
                        hovertemplate=f"{surface}<br>%{{x}}<br>%{{y:,.1f}} m TVDSS<extra></extra>",
                    )
                )
        correlation.update_layout(
            title="Formation tops by well (west → east)",
            yaxis={"autorange": "reversed", "title": "TVDSS (m)"},
            template="plotly_dark",
            height=560,
            legend={"font": {"size": 10}},
            margin={"l": 45, "r": 20, "t": 55, "b": 45},
        )
        left_column, right_column = st.columns(2)
        with left_column:
            st.plotly_chart(well_map, width="stretch")
        with right_column:
            st.plotly_chart(correlation, width="stretch")
        wells_without_tops = sorted(
            set(field_wells["WELL_ID"]) - set(field_tops["WELL_ID"] if field_tops is not None else [])
        )
        if wells_without_tops:
            st.info("No formation tops available yet for: " + ", ".join(wells_without_tops) + ".")
        st.dataframe(field_wells, width="stretch", hide_index=True)
    else:
        st.subheader("Faulted reservoir conceptualization")
        st.caption(
            "Synthetic 3D structural model. Depth increases downward; local coordinates are metres."
        )
        if not reservoir_model_path.is_file():
            st.info(
                "No gridded synthetic reservoir model was found in the selected folder. "
                "Choose the generated data/synthetic_reservoir directory to view maps and sections."
            )
        else:
            try:
                reservoir_model = cached_reservoir_model(
                    str(reservoir_model_path), reservoir_model_path.stat().st_mtime_ns
                )
                wells_path = data_directory / "model" / "wells_synthetic.csv"
                faults_path = data_directory / "model" / "faults_synthetic.csv"
                if not wells_path.is_file() or not faults_path.is_file():
                    raise ValueError("The model folder is missing well locations or fault traces.")
                wells_model = cached_local_csv(str(wells_path), wells_path.stat().st_mtime_ns)
                faults_model = cached_local_csv(str(faults_path), faults_path.stat().st_mtime_ns)
                x_coords = reservoir_model["x_m"]
                y_coords = reservoir_model["y_m"]
                z_coords = reservoir_model["z_m"]
                top_reservoir = reservoir_model["horizon_Top_Reservoir_m"]

                metric_columns = st.columns(4)
                metric_columns[0].metric("Wells", f"{len(wells_model)}")
                metric_columns[1].metric("Faults", f"{faults_model['FAULT_ID'].nunique()}")
                metric_columns[2].metric(
                    "Formations", f"{len(np.unique(reservoir_model['formation']))}"
                )
                metric_columns[3].metric("Grid", f"{len(x_coords)} × {len(y_coords)} × {len(z_coords)}")

                map_figure = go.Figure(
                    go.Contour(
                        x=x_coords,
                        y=y_coords,
                        z=top_reservoir,
                        colorscale="Viridis_r",
                        contours={"showlabels": True, "labelfont": {"color": "white"}},
                        colorbar={"title": "Depth (m)"},
                        name="Top reservoir",
                        hovertemplate="X %{x:.0f} m<br>Y %{y:.0f} m<br>Depth %{z:.1f} m<extra></extra>",
                    )
                )
                for fault_id, fault_points in faults_model.groupby("FAULT_ID"):
                    fault_points = fault_points.sort_values("Y_M")
                    map_figure.add_trace(
                        go.Scatter(
                            x=fault_points["X_M"],
                            y=fault_points["Y_M"],
                            mode="lines",
                            name=f"Fault {fault_id}",
                            line={"color": "#ff776e", "width": 3, "dash": "dash"},
                            hovertemplate=(
                                f"{fault_id}<br>X %{{x:.0f}} m<br>Y %{{y:.0f}} m<extra></extra>"
                            ),
                        )
                    )
                map_figure.add_trace(
                    go.Scatter(
                        x=wells_model["x_m"],
                        y=wells_model["y_m"],
                        mode="markers+text",
                        text=wells_model["well_id"],
                        textposition="top center",
                        name="Wells",
                        marker={
                            "size": 11,
                            "color": "#f7d56d",
                            "line": {"color": "#172332", "width": 1},
                        },
                        hovertemplate="%{text}<br>X %{x:.0f} m<br>Y %{y:.0f} m<extra></extra>",
                    )
                )
                map_figure.update_layout(
                    title="Top-reservoir structure map",
                    xaxis_title="Local X (m)",
                    yaxis_title="Local Y (m)",
                    yaxis={"scaleanchor": "x", "scaleratio": 1},
                    template="plotly_dark",
                    height=520,
                    margin={"l": 30, "r": 20, "t": 55, "b": 45},
                )

                section_y = float(
                    st.slider(
                        "Cross-section Y (m)",
                        min_value=float(y_coords.min()),
                        max_value=float(y_coords.max()),
                        value=float(y_coords[len(y_coords) // 2]),
                        step=float(np.diff(y_coords).mean()),
                    )
                )
                iy = int(np.argmin(np.abs(y_coords - section_y)))
                section_formations = reservoir_model["formation"][:, iy, :]
                formation_order = [
                    "Upper_Shale",
                    "Top_Seal",
                    "Reservoir_Sand",
                    "Basal_Shale",
                    "Tight_Carbonate",
                ]
                formation_colors = ["#586c86", "#8495a8", "#d8a34c", "#74889b", "#b67a5a"]
                formation_codes = np.zeros(section_formations.shape, dtype=np.int8)
                for code, formation_name in enumerate(formation_order):
                    formation_codes[section_formations == formation_name] = code
                categorical_colors = []
                for index, color in enumerate(formation_colors):
                    categorical_colors.extend([[index / 5.0, color], [(index + 1) / 5.0, color]])
                section_figure = go.Figure(
                    go.Heatmap(
                        x=x_coords,
                        y=z_coords,
                        z=formation_codes,
                        zmin=-0.5,
                        zmax=4.5,
                        colorscale=categorical_colors,
                        colorbar={
                            "title": "Formation",
                            "tickvals": list(range(len(formation_order))),
                            "ticktext": formation_order,
                        },
                        hovertemplate="X %{x:.0f} m<br>Depth %{y:.1f} m<br>"
                        "Formation code %{z}<extra></extra>",
                    )
                )
                horizon_styles = (
                    ("horizon_Upper_Shale_Base_m", "Upper shale base", "#c5d5e0"),
                    ("horizon_Top_Reservoir_m", "Top reservoir", "#fff0a6"),
                    ("horizon_Base_Reservoir_m", "Base reservoir", "#ffc271"),
                    ("horizon_Base_Basal_Shale_m", "Base basal shale", "#aadbd3"),
                )
                for horizon_key, horizon_name, color in horizon_styles:
                    section_figure.add_trace(
                        go.Scatter(
                            x=x_coords,
                            y=reservoir_model[horizon_key][iy, :],
                            mode="lines",
                            name=horizon_name,
                            line={"color": color, "width": 2},
                        )
                    )
                for fault_id, fault_points in faults_model.groupby("FAULT_ID"):
                    fault_points = fault_points.sort_values("Y_M")
                    fault_point = fault_points.iloc[
                        int(np.argmin(np.abs(fault_points["Y_M"].to_numpy() - y_coords[iy])))
                    ]
                    section_figure.add_trace(
                        go.Scatter(
                            x=[fault_point["X_M"], fault_point["X_M"]],
                            y=[float(z_coords.min()), float(z_coords.max())],
                            mode="lines",
                            name=f"Fault {fault_id}",
                            line={"color": "#ff776e", "width": 2, "dash": "dash"},
                        )
                    )
                y_tolerance = float(np.diff(y_coords).mean()) / 2.0 + 1e-6
                nearby_wells = wells_model.loc[(wells_model["y_m"] - y_coords[iy]).abs() <= y_tolerance]
                for _, well in nearby_wells.iterrows():
                    section_figure.add_trace(
                        go.Scatter(
                            x=[well["x_m"], well["x_m"]],
                            y=[float(z_coords.min()), float(z_coords.max())],
                            mode="lines+text",
                            text=[well["well_id"], ""],
                            textposition="top center",
                            name=well["well_id"],
                            line={"color": "#f7d56d", "width": 2},
                        )
                    )
                section_figure.update_layout(
                    title=f"Formation cross-section at Y = {y_coords[iy]:.0f} m",
                    xaxis_title="Local X (m)",
                    yaxis_title="Depth (m)",
                    yaxis={"autorange": "reversed"},
                    template="plotly_dark",
                    height=520,
                    margin={"l": 45, "r": 20, "t": 55, "b": 45},
                )

                left_column, right_column = st.columns(2)
                with left_column:
                    st.plotly_chart(map_figure, width="stretch")
                with right_column:
                    st.plotly_chart(section_figure, width="stretch")

                model_3d = go.Figure()
                surface_specs = (
                    ("horizon_Top_Reservoir_m", "Top reservoir", "#d8a34c", 0.85),
                    ("horizon_Base_Reservoir_m", "Base reservoir", "#df755d", 0.70),
                )
                x_mesh, y_mesh = np.meshgrid(x_coords, y_coords)
                for horizon_key, horizon_name, color, opacity in surface_specs:
                    model_3d.add_trace(
                        go.Surface(
                            x=x_mesh,
                            y=y_mesh,
                            z=reservoir_model[horizon_key],
                            name=horizon_name,
                            colorscale=[[0, color], [1, color]],
                            showscale=False,
                            opacity=opacity,
                            hovertemplate=f"{horizon_name}<br>X %{{x:.0f}} m<br>"
                            "Y %{y:.0f} m<br>Depth %{z:.1f} m<extra></extra>",
                        )
                    )
                for fault_id, fault_points in faults_model.groupby("FAULT_ID"):
                    fault_points = fault_points.sort_values("Y_M")
                    fault_y = fault_points["Y_M"].to_numpy()
                    fault_x = fault_points["X_M"].to_numpy()
                    fault_z = np.linspace(float(z_coords.min()), float(z_coords.max()), 12)
                    plane_x = np.tile(fault_x, (len(fault_z), 1))
                    plane_y = np.tile(fault_y, (len(fault_z), 1))
                    plane_z = np.tile(fault_z[:, None], (1, len(fault_y)))
                    model_3d.add_trace(
                        go.Surface(
                            x=plane_x,
                            y=plane_y,
                            z=plane_z,
                            name=f"Fault {fault_id}",
                            colorscale=[[0, "#ff776e"], [1, "#ff776e"]],
                            showscale=False,
                            opacity=0.25,
                            hovertemplate=f"Fault {fault_id}<br>X %{{x:.0f}} m<br>"
                            "Y %{y:.0f} m<br>Depth %{z:.1f} m<extra></extra>",
                        )
                    )
                model_3d.add_trace(
                    go.Scatter3d(
                        x=wells_model["x_m"],
                        y=wells_model["y_m"],
                        z=np.full(len(wells_model), float(z_coords.min())),
                        mode="markers+text",
                        text=wells_model["well_id"],
                        textposition="top center",
                        name="Wells",
                        marker={"size": 5, "color": "#f7d56d"},
                    )
                )
                for _, well in wells_model.iterrows():
                    model_3d.add_trace(
                        go.Scatter3d(
                            x=[well["x_m"], well["x_m"]],
                            y=[well["y_m"], well["y_m"]],
                            z=[float(z_coords.min()), float(z_coords.max())],
                            mode="lines",
                            showlegend=False,
                            line={"color": "#f7d56d", "width": 4},
                        )
                    )
                model_3d.update_layout(
                    title="3D reservoir horizons, faults, and well trajectories",
                    template="plotly_dark",
                    height=680,
                    scene={
                        "xaxis_title": "Local X (m)",
                        "yaxis_title": "Local Y (m)",
                        "zaxis_title": "Depth (m)",
                        "zaxis": {"autorange": "reversed"},
                        "aspectmode": "manual",
                        "aspectratio": {"x": 1, "y": 1, "z": 0.55},
                    },
                    margin={"l": 0, "r": 0, "t": 55, "b": 0},
                )
                st.plotly_chart(model_3d, width="stretch")
                st.caption(
                    "The 3D view shows the reservoir top/base, schematic fault planes, and "
                    "vertical synthetic wells. The cross-section carries the formation labels."
                )
            except Exception as exc:
                st.error(f"Could not render the local reservoir model ({type(exc).__name__}).")

with tabs[0]:
    st.subheader("Well-log tracks")
    available_curves = [
        column
        for column in (
            "GR_API",
            "DTC_US_FT",
            "DTS_US_FT",
            "RHOB_G_CC",
            "PHIT_VV",
            "NPHI_VV",
            "RT_OHM_M",
            "VP_M_S",
            "VS_M_S",
            "AI",
            "VPVS",
            "CALI_IN",
            "DRHO_G_CC",
            "PEF_B_E",
        )
        if column in filtered
    ]
    default_curves = [
        column
        for column in (
            "GR_API", "DTC_US_FT", "DTS_US_FT", "RHOB_G_CC", "PHIT_VV", "NPHI_VV", "RT_OHM_M"
        )
        if column in available_curves
    ]
    selected_curves = st.multiselect(
        "Curves to display",
        available_curves,
        default=default_curves,
        format_func=_label,
        key="log_curves",
    )
    if selected_curves:
        figure = make_subplots(
            rows=1,
            cols=len(selected_curves),
            shared_yaxes=True,
            horizontal_spacing=min(0.06, 0.3 / len(selected_curves)),
            subplot_titles=[_label(column) for column in selected_curves],
        )
        for index, column in enumerate(selected_curves, start=1):
            figure.add_trace(
                go.Scattergl(
                    x=filtered[column],
                    y=filtered[depth_column],
                    mode="lines",
                    line={"width": 1.3, "color": px.colors.qualitative.Set2[(index - 1) % 8]},
                    name=_label(column),
                    connectgaps=False,
                    showlegend=False,
                ),
                row=1,
                col=index,
            )
            figure.update_xaxes(
                title_text=_label(column), row=1, col=index, showgrid=True, zeroline=False
            )
            if column == "RT_OHM_M":
                figure.update_xaxes(type="log", row=1, col=index)
        for _, top in well_tops_in_view.iterrows():
            figure.add_hline(
                y=float(top[depth_column]),
                line={"color": "#f7d56d", "width": 1, "dash": "dot"},
                row="all",
                col="all",
            )
            figure.add_annotation(
                x=0,
                xref="paper",
                y=float(top[depth_column]),
                yref="y",
                text=escape(str(top["SURFACE"])),
                showarrow=False,
                xanchor="left",
                yanchor="bottom",
                font={"size": 10, "color": "#f7d56d"},
            )
        figure.update_yaxes(
            title_text="Depth (m)",
            autorange="reversed",
            row=1,
            col=1,
            gridcolor="rgba(180,205,220,.12)",
        )
        figure.update_layout(
            template="plotly_dark",
            height=650,
            margin={"l": 45, "r": 20, "t": 75, "b": 60},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(8,18,28,.5)",
        )
        st.plotly_chart(figure, width="stretch")
    else:
        st.info("Choose one or more curves above.")

    label_column = _color_column(filtered)
    if label_column:
        st.subheader("Synthetic facies / fluid labels" if synthetic else "Source classification")
        fig_labels = px.scatter(
            filtered,
            x=label_column,
            y=depth_column,
            color=label_column,
            hover_data=[
                column
                for column in ("MD_M", "TWT_MS", "SYNTHETIC_FLUID", "QC_BAD_HOLE")
                if column in filtered
            ],
            color_discrete_sequence=px.colors.qualitative.Set2,
            template="plotly_dark",
        )
        fig_labels.update_traces(marker={"size": 5, "opacity": 0.75})
        fig_labels.update_yaxes(autorange="reversed", title="Depth (m)")
        fig_labels.update_layout(height=280, showlegend=True, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_labels, width="stretch")

with tabs[2]:
    st.subheader("Elastic and petrophysical relationships")
    left, right = st.columns(2)
    color_column = _color_column(filtered)
    with left:
        if {"DTC_US_FT", "RHOB_G_CC"}.issubset(filtered.columns):
            crossplot = px.scatter(
                filtered,
                x="DTC_US_FT",
                y="RHOB_G_CC",
                color=color_column,
                hover_data=[
                    column
                    for column in ("MD_M", "TWT_MS", "SYNTHETIC_FLUID", "QC_BAD_HOLE")
                    if column in filtered
                ],
                color_discrete_sequence=px.colors.qualitative.Set2,
                template="plotly_dark",
                title="Compressional slowness vs density",
            )
            crossplot.update_layout(
                xaxis_title="DTC (µs/ft)",
                yaxis_title="RHOB (g/cc)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(crossplot, width="stretch")
        else:
            st.info("DTC_US_FT and RHOB_G_CC are required for this crossplot.")
    with right:
        if {"GR_API", "RT_OHM_M"}.issubset(filtered.columns):
            petrophysics = px.scatter(
                filtered,
                x="RT_OHM_M",
                y="GR_API",
                color=color_column,
                log_x=True,
                hover_data=[
                    column for column in ("MD_M", "TWT_MS", "SYNTHETIC_FLUID") if column in filtered
                ],
                color_discrete_sequence=px.colors.qualitative.Set2,
                template="plotly_dark",
                title="Gamma ray vs deep resistivity",
            )
            petrophysics.update_layout(
                xaxis_title="RT (Ω·m, log scale)",
                yaxis_title="GR (API)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(petrophysics, width="stretch")
        else:
            st.info("GR_API and RT_OHM_M are required for this crossplot.")

    if {"AI", "VPVS"}.issubset(filtered.columns):
        st.markdown("#### Acoustic impedance vs Vp/Vs")
        multi_well = "WELL_ID" in logs and logs["WELL_ID"].nunique(dropna=True) > 1
        scope = (
            st.radio(
                "Samples",
                ["Selected well, depth window", "All wells"],
                horizontal=True,
                key="ai_vpvs_scope",
            )
            if multi_well
            else "Selected well, depth window"
        )
        elastic = filtered if scope.startswith("Selected") else logs
        if "FORMATION" in elastic:
            formations = sorted(elastic["FORMATION"].dropna().astype(str).unique())
            reservoir = [name for name in formations if "Plover" in name]
            chosen = st.multiselect(
                "Formations",
                formations,
                default=reservoir if scope == "All wells" and reservoir else formations,
                key=f"ai_vpvs_formations_{scope}_{well_id}",
            )
            elastic = elastic.loc[elastic["FORMATION"].astype(str).isin(chosen)]
        elastic = elastic.dropna(subset=["AI", "VPVS"])
        if elastic.empty:
            st.info("No samples with both acoustic impedance and Vp/Vs in this selection.")
        else:
            elastic_color = "WELL_ID" if scope == "All wells" else color_column
            ai_vpvs = px.scatter(
                elastic,
                x="AI",
                y="VPVS",
                color=elastic_color,
                hover_data=[
                    column
                    for column in ("WELL_ID", "MD_M", "TVDSS_M", "FORMATION", "GR_API", "NPHI_VV")
                    if column in elastic
                ],
                color_discrete_sequence=px.colors.qualitative.Set2,
                template="plotly_dark",
                title=f"{len(elastic):,} samples",
            )
            ai_vpvs.update_traces(marker={"size": 4, "opacity": 0.6})
            ai_vpvs.update_layout(
                xaxis_title=_label("AI"),
                yaxis_title=_label("VPVS"),
                height=520,
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(ai_vpvs, width="stretch")
            st.caption(
                "Low impedance with low Vp/Vs typically points to hydrocarbon-bearing porous sand; "
                "shales plot at higher Vp/Vs. Interpret with the GR and neutron-density response."
            )

    if checkshot_path:
        checkshot = _cached_file(checkshot_path)
        if selected_well_id is not None and "WELL_ID" in checkshot:
            checkshot = checkshot.loc[checkshot["WELL_ID"].astype(str) == selected_well_id]
        if {"TWT_MS", "TVD_M"}.issubset(checkshot.columns):
            tie_figure = px.scatter(
                checkshot,
                x="TWT_MS",
                y="TVD_M",
                template="plotly_dark",
                title="Checkshot time-depth tie",
                hover_data=[column for column in ("MD_M", "WELL_ID") if column in checkshot],
            )
            tie_figure.update_yaxes(autorange="reversed", title="TVD (m)")
            tie_figure.update_xaxes(title="Two-way travel time (ms)")
            tie_figure.update_layout(height=360, paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(tie_figure, width="stretch")

with tabs[3]:
    if seismic_stacks:
        render_seismic_viewer()
    if not seismic_stacks or gather_path.is_file():
        st.subheader("Local angle gather")
        st.caption(
            "Amplitude is synthetic and dimensionless for the example gather; it is not field seismic."
        )
        if gather_path.is_file():
            try:
                gather = cached_gather(str(gather_path), gather_path.stat().st_mtime_ns)
                trace_ids = [str(value) for value in gather["lateral_trace_id"]]
                selected_trace = st.selectbox(
                    "Lateral trace", range(len(trace_ids)), format_func=lambda value: trace_ids[value]
                )
                amplitude = gather["amplitude"][:, :, selected_trace]
                clip = max(float(np.nanpercentile(np.abs(amplitude), 99.0)), 1e-8)
                hover_template = "TWT %{y:.1f} ms<br>Angle %{x:.0f}°<br>"
                hover_template += "Amplitude %{z:.4f}<extra></extra>"
                seismic_figure = go.Figure(
                    go.Heatmap(
                        x=gather["incident_angle_deg"],
                        y=gather["twt_ms"],
                        z=amplitude,
                        zmin=-clip,
                        zmax=clip,
                        zmid=0,
                        colorscale="RdBu",
                        colorbar={"title": "Amplitude"},
                        hovertemplate=hover_template,
                    )
                )
                seismic_figure.update_yaxes(autorange="reversed", title="Two-way travel time (ms)")
                seismic_figure.update_xaxes(title="Incident angle (°)")
                seismic_figure.update_layout(
                    template="plotly_dark",
                    height=560,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(8,18,28,.5)",
                    margin={"l": 55, "r": 25, "t": 30, "b": 55},
                )
                st.plotly_chart(seismic_figure, width="stretch")
                st.caption(
                    f"Array: {gather['amplitude'].shape[0]} time samples × "
                    f"{gather['amplitude'].shape[1]} angles × {gather['amplitude'].shape[2]} traces."
                )
            except Exception as exc:
                st.error(f"Could not read the local angle gather ({type(exc).__name__}).")
        else:
            st.info("No supported local angle-gather NPZ file was found in the selected folder.")

with tabs[4]:
    render_well_qc_tie()

with tabs[5]:
    st.subheader("Filtered source table")
    st.caption(f"Showing {len(filtered):,} rows from {well_path.name}.")
    st.dataframe(filtered, width="stretch", height=440)
    csv_bytes = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download current depth window as CSV",
        data=csv_bytes,
        file_name="avo_dashboard_depth_window.csv",
        mime="text/csv",
    )
    st.markdown("#### Provenance")
    if manifest:
        st.json({key: value for key, value in manifest.items() if key not in {"assets"}})
    else:
        st.info(
            "No manifest was found beside the selected table. Verify source and "
            "usage terms before interpretation."
        )
