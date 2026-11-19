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
    is_synthetic,
    read_gather,
    read_manifest,
    read_reservoir_model,
    read_table,
    table_candidates,
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


def _cached_file(path: Path):
    return cached_table(str(path), path.stat().st_mtime_ns)


def _color_column(frame):
    for column in ("SYNTHETIC_LITHOLOGY", "LITHOLOGY", "FACIES", "SYNTHETIC_FLUID"):
        if column in frame:
            return column
    return None


def _depth_options(frame):
    return [column for column in ("MD_M", "TVD_M") if column in frame]


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
    }
    return labels.get(column, column.replace("_", " ").title())


with st.sidebar:
    st.markdown("### Data source")
    reservoir_examples = PROJECT_ROOT / "data" / "synthetic_reservoir"
    legacy_examples = PROJECT_ROOT / "data" / "volve" / "examples"
    default_directory = reservoir_examples if reservoir_examples.is_dir() else legacy_examples
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
        "Use a folder containing a well-log CSV/Parquet file. "
        "Checkshot, gather, and reservoir model views are optional."
    )
    if well_path:
        st.success(f"Well logs: {well_path.name}")
    else:
        st.warning("No supported well-log CSV or Parquet file found in this folder.")

if well_path is None:
    st.markdown(
        '<div class="hero"><div class="hero-kicker">AVO uncertainty workflow</div>'
        "<h1>Local Data Explorer</h1>"
        "<p>Inspect well-log, time-depth, QC, and seismic inputs.</p></div>",
        unsafe_allow_html=True,
    )
    st.info("Select a local folder with a well-log CSV or Parquet file in the sidebar.")
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

badge_class = "real" if not synthetic else ""
badge_text = (
    "LOCAL DATA · VERIFY PROVENANCE" if not synthetic else "SYNTHETIC EXAMPLE · NOT FIELD DATA"
)
st.markdown(
    f'<div class="hero"><div class="hero-kicker">Lithology · Fluids · Uncertainty</div>'
    f"<h1>AVO Data Explorer</h1><p>{escape(well_id)} &nbsp; "
    f'<span class="data-badge {badge_class}">{badge_text}</span></p></div>',
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

tabs = st.tabs(
    [
        "Reservoir model",
        "Well logs",
        "Rock-property crossplots",
        "Seismic gather",
        "Data preview",
    ]
)

with tabs[0]:
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

with tabs[1]:
    st.subheader("Well-log tracks")
    available_curves = [
        column
        for column in (
            "GR_API",
            "DTC_US_FT",
            "DTS_US_FT",
            "RHOB_G_CC",
            "PHIT_VV",
            "RT_OHM_M",
            "CALI_IN",
            "DRHO_G_CC",
            "PEF_B_E",
        )
        if column in filtered
    ]
    default_curves = [
        column
        for column in ("GR_API", "DTC_US_FT", "DTS_US_FT", "RHOB_G_CC", "PHIT_VV", "RT_OHM_M")
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
