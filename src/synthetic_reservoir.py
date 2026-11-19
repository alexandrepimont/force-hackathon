"""Build a reproducible conceptual reservoir and synthetic AVO input package.

All generated data are scenario-based and synthetic. No real Volve measurements are
read, inferred, or reproduced by this module.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import lasio
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import pyarrow as pa
import pyarrow.parquet as pq
import segyio
import yaml

DEFAULT_CONFIG = Path("configs/synthetic_reservoir.yml")
DEFAULT_OUTPUT = Path("data/synthetic_reservoir")
SYNTHETIC_WARNING = (
    "Entirely synthetic conceptual scenario; not derived from, calibrated to, or "
    "representative of real Volve measurements."
)
FACIES_PROPERTIES = {
    "shale": "shale",
    "brine_sand": "brine_sand",
    "oil_sand": "oil_sand",
    "gas_sand": "gas_sand",
    "tight": "tight",
}
FORMATION_CODES = {
    "Upper_Shale": 1,
    "Top_Seal": 2,
    "Reservoir_Sand": 3,
    "Basal_Shale": 4,
    "Tight_Carbonate": 5,
}
LITHOLOGY_CODES = {"shale": 1, "sand": 2, "tight": 3}
FLUID_CODES = {"not_applicable": 0, "brine": 1, "oil": 2, "gas": 3}


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load and validate the YAML scenario configuration."""
    with path.expanduser().open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or config.get("scenario", {}).get("synthetic_data") is not True:
        raise ValueError("The scenario must explicitly declare synthetic_data: true.")
    if len(config.get("wells", [])) != 7:
        raise ValueError("This conceptual package requires exactly seven configured wells.")
    if not config.get("faults") or len(config.get("formations", [])) < 3:
        raise ValueError("At least one fault and three formations are required.")
    return config


def _fault_offset(x: np.ndarray, y: np.ndarray, faults: list[dict[str, Any]]) -> np.ndarray:
    offset = np.zeros_like(x, dtype=np.float64)
    for fault in faults:
        trace_x = fault["x_intercept_m"] + fault["x_slope_per_y"] * (y - fault["reference_y_m"])
        downthrown = x >= trace_x if fault["downthrown_side"] == "east" else x < trace_x
        offset += np.where(downthrown, float(fault["throw_m"]), 0.0)
    return offset


def _surfaces(x: np.ndarray, y: np.ndarray, config: dict[str, Any]) -> dict[str, np.ndarray]:
    structure = config["structure"]
    model = config["model"]
    faults = _fault_offset(x, y, config["faults"])
    dx = (x - structure["anticline_x_m"]) / structure["anticline_radius_m"]
    dy = (y - structure["anticline_y_m"]) / structure["anticline_radius_m"]
    anticline = float(structure["anticline_amplitude_m"]) * np.exp(-(dx**2 + dy**2))
    top = (
        float(structure["reservoir_top_base_m"])
        + float(structure["regional_dip_x"]) * (x - model["x_max_m"] / 2.0)
        + float(structure["regional_dip_y"]) * (y - model["y_max_m"] / 2.0)
        - anticline
        + faults
    )
    return {
        "Upper_Shale_Base": top - float(structure["caprock_thickness_m"]),
        "Top_Reservoir": top,
        "Base_Reservoir": top + float(structure["reservoir_thickness_m"]),
        "Base_Basal_Shale": top
        + float(structure["reservoir_thickness_m"])
        + float(structure["basal_shale_thickness_m"]),
    }


def _classify_depths(
    depth: np.ndarray,
    top_shale: np.ndarray,
    top_reservoir: np.ndarray,
    base_reservoir: np.ndarray,
    base_basal_shale: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = np.broadcast_shapes(
        np.shape(depth), np.shape(top_shale), np.shape(top_reservoir), np.shape(x)
    )
    z = np.broadcast_to(depth, shape)
    shale_base = np.broadcast_to(top_shale, shape)
    reservoir_top = np.broadcast_to(top_reservoir, shape)
    reservoir_base = np.broadcast_to(base_reservoir, shape)
    basal_base = np.broadcast_to(base_basal_shale, shape)
    xx = np.broadcast_to(x, shape)
    yy = np.broadcast_to(y, shape)
    structure = config["structure"]
    thickness_from_top = z - reservoir_top
    relief = float(structure["gas_contact_relief_m"]) * np.sin(xx / 900.0) + float(
        structure["gas_contact_relief_m"]
    ) * np.cos(yy / 1100.0)
    oil_relief = float(structure["oil_contact_relief_m"]) * np.sin(yy / 1200.0) - float(
        structure["oil_contact_relief_m"]
    ) * np.cos(xx / 1300.0)
    gas_contact = float(structure["gas_column_m"]) + relief
    oil_contact = float(structure["oil_column_m"]) + oil_relief

    formation = np.full(shape, "Tight_Carbonate", dtype="<U24")
    lithology = np.full(shape, "tight", dtype="<U16")
    fluid = np.full(shape, "not_applicable", dtype="<U16")

    upper = z < shale_base
    seal = (z >= shale_base) & (z < reservoir_top)
    reservoir = (z >= reservoir_top) & (z < reservoir_base)
    basal = (z >= reservoir_base) & (z < basal_base)
    formation[upper] = "Upper_Shale"
    formation[seal] = "Top_Seal"
    formation[reservoir] = "Reservoir_Sand"
    formation[basal] = "Basal_Shale"
    lithology[upper | seal | basal] = "shale"
    lithology[reservoir] = "sand"
    gas = reservoir & (thickness_from_top < gas_contact)
    oil = reservoir & (thickness_from_top >= gas_contact) & (thickness_from_top < oil_contact)
    brine = reservoir & ~(gas | oil)
    fluid[gas] = "gas"
    fluid[oil] = "oil"
    fluid[brine] = "brine"
    return formation, lithology, fluid


def _build_earth_model(config: dict[str, Any]) -> dict[str, Any]:
    model_config = config["model"]
    x_coords = np.linspace(
        model_config["x_min_m"], model_config["x_max_m"], model_config["lateral_samples"]
    )
    y_coords = np.linspace(
        model_config["y_min_m"], model_config["y_max_m"], model_config["lateral_samples"]
    )
    z_coords = np.arange(
        model_config["z_min_m"],
        model_config["z_max_m"] + model_config["z_step_m"] * 0.5,
        model_config["z_step_m"],
    )
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)
    horizons = _surfaces(x_grid, y_grid, config)
    x3 = x_grid[None, :, :]
    y3 = y_grid[None, :, :]
    z3 = z_coords[:, None, None]
    formation, lithology, fluid = _classify_depths(
        z3,
        horizons["Upper_Shale_Base"][None, :, :],
        horizons["Top_Reservoir"][None, :, :],
        horizons["Base_Reservoir"][None, :, :],
        horizons["Base_Basal_Shale"][None, :, :],
        x3,
        y3,
        config,
    )

    properties = config["rock_physics"]["properties"]
    property_keys = ("shale", "brine_sand", "oil_sand", "gas_sand", "tight")
    vp = np.empty(formation.shape, dtype=np.float32)
    vs = np.empty_like(vp)
    rho = np.empty_like(vp)
    depth_compaction = np.broadcast_to(z3 - float(model_config["z_min_m"]), formation.shape)
    for property_key in property_keys:
        if property_key.endswith("_sand"):
            fluid_name = property_key.removesuffix("_sand")
            mask = (lithology == "sand") & (fluid == fluid_name)
        else:
            mask = lithology == property_key
        values = properties[property_key]
        vp[mask] = float(values["vp_m_s"]) + 0.22 * depth_compaction[mask]
        vs[mask] = float(values["vs_m_s"]) + 0.12 * depth_compaction[mask]
        rho[mask] = float(values["rho_g_cc"]) + 0.00010 * depth_compaction[mask]

    return {
        "x_m": x_coords.astype(np.float32),
        "y_m": y_coords.astype(np.float32),
        "z_m": z_coords.astype(np.float32),
        "x_grid_m": x_grid.astype(np.float32),
        "y_grid_m": y_grid.astype(np.float32),
        "horizons": {key: value.astype(np.float32) for key, value in horizons.items()},
        "formation": formation,
        "lithology": lithology,
        "fluid": fluid,
        "vp_m_s": vp,
        "vs_m_s": vs,
        "rho_g_cc": rho,
    }


def _nearest_index(values: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(values.astype(float) - float(value))))


def _well_logs(
    config: dict[str, Any], earth: dict[str, Any], rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model = config["model"]
    noise = config["rock_physics"]["log_noise"]
    props = config["rock_physics"]["properties"]
    depth = np.arange(
        float(model["z_min_m"]),
        float(model["z_max_m"]),
        float(model["log_step_m"]),
        dtype=float,
    )
    rows: list[dict[str, Any]] = []
    checkshot: list[dict[str, Any]] = []
    for well in config["wells"]:
        ix = _nearest_index(earth["x_m"], well["x_m"])
        iy = _nearest_index(earth["y_m"], well["y_m"])
        _, lithology, fluid = _classify_depths(
            depth,
            earth["horizons"]["Upper_Shale_Base"][iy, ix],
            earth["horizons"]["Top_Reservoir"][iy, ix],
            earth["horizons"]["Base_Reservoir"][iy, ix],
            earth["horizons"]["Base_Basal_Shale"][iy, ix],
            np.full(depth.shape, well["x_m"]),
            np.full(depth.shape, well["y_m"]),
            config,
        )
        formations, _, _ = _classify_depths(
            depth,
            earth["horizons"]["Upper_Shale_Base"][iy, ix],
            earth["horizons"]["Top_Reservoir"][iy, ix],
            earth["horizons"]["Base_Reservoir"][iy, ix],
            earth["horizons"]["Base_Basal_Shale"][iy, ix],
            np.full(depth.shape, well["x_m"]),
            np.full(depth.shape, well["y_m"]),
            config,
        )
        base_keys = np.asarray(
            [
                f"{fluid_name}_sand" if lithology_name == "sand" else lithology_name
                for lithology_name, fluid_name in zip(lithology, fluid)
            ],
            dtype="<U16",
        )
        vp_base = np.array([props[key]["vp_m_s"] for key in base_keys], dtype=float)
        vs_base = np.array([props[key]["vs_m_s"] for key in base_keys], dtype=float)
        rho_base = np.array([props[key]["rho_g_cc"] for key in base_keys], dtype=float)
        gr_base = np.array([props[key]["gr_api"] for key in base_keys], dtype=float)
        phi_base = np.array([props[key]["phi"] for key in base_keys], dtype=float)
        rt_base = np.array([props[key]["rt_ohm_m"] for key in base_keys], dtype=float)
        pef_base = np.array([props[key]["pef"] for key in base_keys], dtype=float)
        vp = vp_base * (1.0 + rng.normal(0.0, noise["vp_fraction"], depth.size))
        vs = vs_base * (1.0 + rng.normal(0.0, noise["vs_fraction"], depth.size))
        rho = rho_base + rng.normal(0.0, noise["density_g_cc"], depth.size)
        gr = np.clip(gr_base + rng.normal(0.0, noise["gr_api"], depth.size), 0.0, 150.0)
        phi = np.clip(phi_base + rng.normal(0.0, noise["porosity"], depth.size), 0.005, 0.38)
        rt = rt_base * np.exp(rng.normal(0.0, noise["log_resistivity_fraction"], depth.size))
        pef = np.clip(rng.normal(pef_base, 0.12), 1.0, 6.0)
        dtc = 304800.0 / vp
        dts = 304800.0 / vs
        twt = np.empty(depth.size, dtype=float)
        twt[0] = float(model["seismic_twt_start_ms"])
        twt[1:] = twt[0] + np.cumsum(1000.0 * np.diff(depth) * (1.0 / vp[:-1] + 1.0 / vp[1:]))
        bad_hole = np.zeros(depth.size, dtype=bool)
        for start, end in noise["bad_hole_intervals_m"]:
            bad_hole |= (depth >= float(start)) & (depth < float(end))
        cali = rng.normal(np.where(bad_hole, 12.0, 8.5), np.where(bad_hole, 0.25, 0.10))
        drho = rng.normal(np.where(bad_hole, 0.16, 0.015), np.where(bad_hole, 0.03, 0.012))
        rho[bad_hole] += 0.06
        sw = np.select(
            [fluid == "brine", fluid == "oil", fluid == "gas"],
            [0.88, 0.32, 0.18],
            default=np.nan,
        )
        lith_codes = np.array([LITHOLOGY_CODES[value] for value in lithology])
        fluid_codes = np.array([FLUID_CODES[value] for value in fluid])
        well_rows: list[dict[str, Any]] = []
        for i, md in enumerate(depth):
            row = {
                "WELL_ID": well["well_id"],
                "X_M": float(well["x_m"]),
                "Y_M": float(well["y_m"]),
                "MD_M": round(float(md), 3),
                "TVD_M": round(float(md), 3),
                "TWT_MS": round(float(twt[i]), 3),
                "DTC_US_FT": round(float(dtc[i]), 3),
                "DTS_US_FT": round(float(dts[i]), 3),
                "RHOB_G_CC": round(float(rho[i]), 4),
                "GR_API": round(float(gr[i]), 2),
                "PHIT_VV": round(float(phi[i]), 4),
                "RT_OHM_M": round(float(rt[i]), 3),
                "SW_VV": None if not np.isfinite(sw[i]) else round(float(sw[i]), 3),
                "CALI_IN": round(float(cali[i]), 3),
                "BS_IN": 8.5,
                "DRHO_G_CC": round(float(drho[i]), 4),
                "PEF_B_E": round(float(pef[i]), 3),
                "FORMATION": str(formations[i]),
                "SYNTHETIC_LITHOLOGY": str(lithology[i]),
                "SYNTHETIC_FLUID": str(fluid[i]),
                "LITHOLOGY_CODE": int(lith_codes[i]),
                "FLUID_CODE": int(fluid_codes[i]),
                "QC_BAD_HOLE": bool(bad_hole[i]),
                "SYNTHETIC_DATA": True,
            }
            for curve in ("DTS_US_FT", "RT_OHM_M", "PHIT_VV"):
                if rng.random() < {"DTS_US_FT": 0.002, "RT_OHM_M": 0.006, "PHIT_VV": 0.004}[curve]:
                    row[curve] = None
            well_rows.append(row)
        rows.extend(well_rows)

        for index in range(0, len(well_rows), 100):
            row = well_rows[index]
            checkshot.append(
                {
                    "WELL_ID": well["well_id"],
                    "X_M": float(well["x_m"]),
                    "Y_M": float(well["y_m"]),
                    "MD_M": row["MD_M"],
                    "TVD_M": row["TVD_M"],
                    "TWT_MS": row["TWT_MS"],
                    "SYNTHETIC_DATA": True,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(checkshot)


def _ricker(frequency_hz: float, dt_ms: float) -> np.ndarray:
    half_length = 0.080
    time_s = np.arange(-half_length, half_length + dt_ms / 2000.0, dt_ms / 1000.0)
    term = (np.pi * frequency_hz * time_s) ** 2
    return ((1.0 - 2.0 * term) * np.exp(-term)).astype(np.float32)


def _generate_seismic(
    config: dict[str, Any], earth: dict[str, Any], rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    model = config["model"]
    angles = np.asarray(model["seismic_angles_deg"], dtype=np.float32)
    dt_ms = float(model["seismic_sample_interval_ms"])
    time_ms = np.arange(
        float(model["seismic_twt_start_ms"]),
        float(model["seismic_twt_end_ms"]) + dt_ms * 0.5,
        dt_ms,
        dtype=np.float32,
    )
    vp = earth["vp_m_s"].astype(np.float64)
    vs = earth["vs_m_s"].astype(np.float64)
    rho = earth["rho_g_cc"].astype(np.float64)
    z = earth["z_m"].astype(np.float64)
    dz = float(np.diff(z).mean())
    twt_by_depth = np.empty_like(vp, dtype=np.float64)
    twt_by_depth[0] = float(time_ms[0])
    twt_by_depth[1:] = float(time_ms[0]) + np.cumsum(
        1000.0 * dz * (1.0 / vp[:-1] + 1.0 / vp[1:]), axis=0
    )

    vp1, vp2 = vp[:-1], vp[1:]
    vs1, vs2 = vs[:-1], vs[1:]
    rho1, rho2 = rho[:-1], rho[1:]
    vp_avg = 0.5 * (vp1 + vp2)
    vs_avg = 0.5 * (vs1 + vs2)
    rho_avg = 0.5 * (rho1 + rho2)
    dvp = (vp2 - vp1) / vp_avg
    dvs = (vs2 - vs1) / vs_avg
    drho = (rho2 - rho1) / rho_avg
    intercept = 0.5 * (dvp + drho)
    gradient = 0.5 * dvp - 2.0 * (vs_avg / vp_avg) ** 2 * (2.0 * dvs + drho)
    curvature = 0.5 * dvp
    wavelet = _ricker(float(model["wavelet_frequency_hz"]), dt_ms)
    amplitudes = np.zeros(
        (len(time_ms), len(angles), len(earth["y_m"]), len(earth["x_m"])), dtype=np.float32
    )
    for angle_index, angle in enumerate(angles):
        theta = np.deg2rad(float(angle))
        sin2 = np.sin(theta) ** 2
        tan2 = np.tan(theta) ** 2
        reflectivity = intercept + gradient * sin2 + curvature * (tan2 - sin2)
        for iy in range(len(earth["y_m"])):
            for ix in range(len(earth["x_m"])):
                event_time = twt_by_depth[1:, iy, ix]
                series = np.interp(
                    time_ms,
                    event_time,
                    reflectivity[:, iy, ix],
                    left=0.0,
                    right=0.0,
                )
                convolved = np.convolve(series, wavelet, mode="same")
                noise = rng.normal(0.0, float(model["seismic_noise_std"]), len(time_ms))
                amplitudes[:, angle_index, iy, ix] = (convolved + noise).astype(np.float32)
    return time_ms, amplitudes


def _write_segy(
    path: Path,
    traces: np.ndarray,
    dt_ms: float,
    delay_ms: float,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    angle_deg: int | None,
    title: str,
    angles_deg: list[int] | None = None,
) -> None:
    """Write a regular 3D angle volume or an angle-coded CMP gather SEG-Y."""
    combined = angle_deg is None
    if combined:
        samples, angle_count, ny, nx = traces.shape
        if angles_deg is None or len(angles_deg) != angle_count:
            raise ValueError("Combined SEG-Y output requires one angle value per angle axis.")
        angles = angles_deg
    else:
        samples, ny, nx = traces.shape
        angles = []
    ntraces = ny * nx * (len(angles) if combined else 1)
    spec = segyio.spec()
    spec.sorting = segyio.TraceSortingFormat.INLINE_SORTING
    spec.format = 5  # IEEE floating point
    spec.samples = np.arange(samples, dtype=np.float32) * dt_ms
    spec.ilines = [1] if combined else list(range(1, ny + 1))
    spec.xlines = list(range(1, ntraces + 1)) if combined else list(range(1, nx + 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    with segyio.create(str(path), spec) as segy:
        segy.text[0] = segyio.tools.create_text_header(
            {
                1: title[:78],
                3: "IEEE FLOATS; METRIC LOCAL XY; ANGLE-GATHER TRACE HEADER DOCUMENTED IN MANIFEST",
                4: "TWT START IN DELAY RECORDING TIME; SAMPLE INTERVAL IN MICROSECONDS",
                5: "ANGLES IN DEGREES X 1000 IN UNASSIGNED TRACE HEADER WORD 233-236",
            }
        )
        segy.bin[segyio.BinField.Interval] = int(round(dt_ms * 1000.0))
        segy.bin[segyio.BinField.Samples] = samples
        segy.bin[segyio.BinField.Format] = 5
        segy.bin[segyio.BinField.SEGYRevision] = 1
        segy.bin[segyio.BinField.Traces] = len(angles) if combined else 1
        if combined:
            angles = (5, 15, 25, 35, 45)
            trace_index = 0
            for iy in range(ny):
                for ix in range(nx):
                    cdp = iy * nx + ix + 1
                    for angle_index, angle in enumerate(angles):
                        segy.trace[trace_index] = np.ascontiguousarray(
                            traces[:, angle_index, iy, ix], dtype=np.float32
                        )
                        segy.header[trace_index] = {
                            segyio.TraceField.TRACE_SEQUENCE_LINE: trace_index + 1,
                            segyio.TraceField.TRACE_SEQUENCE_FILE: trace_index + 1,
                            segyio.TraceField.FieldRecord: cdp,
                            segyio.TraceField.CDP: cdp,
                            segyio.TraceField.INLINE_3D: iy + 1,
                            segyio.TraceField.CROSSLINE_3D: ix + 1,
                            segyio.TraceField.SourceX: int(round(float(x_coords[ix]) * 1000)),
                            segyio.TraceField.SourceY: int(round(float(y_coords[iy]) * 1000)),
                            segyio.TraceField.GroupX: int(round(float(x_coords[ix]) * 1000)),
                            segyio.TraceField.GroupY: int(round(float(y_coords[iy]) * 1000)),
                            segyio.TraceField.SourceGroupScalar: -1000,
                            segyio.TraceField.CoordinateUnits: 1,
                            segyio.TraceField.DelayRecordingTime: int(round(delay_ms)),
                            segyio.TraceField.UnassignedInt1: int(angle * 1000),
                        }
                        trace_index += 1
        else:
            for iy in range(ny):
                for ix in range(nx):
                    trace_index = iy * nx + ix
                    segy.trace[trace_index] = np.ascontiguousarray(
                        traces[:, iy, ix], dtype=np.float32
                    )
                    segy.header[trace_index] = {
                        segyio.TraceField.TRACE_SEQUENCE_LINE: trace_index + 1,
                        segyio.TraceField.TRACE_SEQUENCE_FILE: trace_index + 1,
                        segyio.TraceField.FieldRecord: trace_index + 1,
                        segyio.TraceField.CDP: trace_index + 1,
                        segyio.TraceField.INLINE_3D: iy + 1,
                        segyio.TraceField.CROSSLINE_3D: ix + 1,
                        segyio.TraceField.SourceX: int(round(float(x_coords[ix]) * 1000)),
                        segyio.TraceField.SourceY: int(round(float(y_coords[iy]) * 1000)),
                        segyio.TraceField.GroupX: int(round(float(x_coords[ix]) * 1000)),
                        segyio.TraceField.GroupY: int(round(float(y_coords[iy]) * 1000)),
                        segyio.TraceField.SourceGroupScalar: -1000,
                        segyio.TraceField.CoordinateUnits: 1,
                        segyio.TraceField.DelayRecordingTime: int(round(delay_ms)),
                    }
    os.chmod(path, 0o600)


def _write_las(path: Path, frame: pd.DataFrame, well_id: str) -> None:
    well = frame.loc[frame["WELL_ID"] == well_id]
    las = lasio.LASFile()
    las.version.VERS.value = 2.0
    las.well.WELL.value = well_id
    las.well.STRT.value = float(well["MD_M"].iloc[0])
    las.well.STOP.value = float(well["MD_M"].iloc[-1])
    las.well.STEP.value = float(well["MD_M"].iloc[1] - well["MD_M"].iloc[0])
    las.well.NULL.value = -999.25
    las.well.COMP.value = "SYNTHETIC SCENARIO"
    las.well.LOC.value = f"LOCAL X={well['X_M'].iloc[0]:.0f} M, Y={well['Y_M'].iloc[0]:.0f} M"
    las.well.PROV.value = "SYNTHETIC - NOT FIELD DATA"
    las.curves.clear()
    curves = (
        ("TVD", "M", "True vertical depth"),
        ("TWT", "MS", "Two-way travel time"),
        ("DTC", "US/F", "Compressional slowness"),
        ("DTS", "US/F", "Shear slowness"),
        ("RHOB", "G/CC", "Bulk density"),
        ("GR", "API", "Gamma ray"),
        ("PHIT", "V/V", "Total porosity"),
        ("RT", "OHMM", "Deep resistivity"),
        ("SW", "V/V", "Water saturation"),
        ("CALI", "IN", "Caliper"),
        ("BS", "IN", "Bit size"),
        ("DRHO", "G/CC", "Density correction"),
        ("PEF", "B/E", "Photoelectric factor"),
        ("LITHCODE", "CODE", "Synthetic lithology code"),
        ("FLUIDCODE", "CODE", "Synthetic fluid code"),
    )
    source_columns = {
        "TVD": "TVD_M",
        "TWT": "TWT_MS",
        "DTC": "DTC_US_FT",
        "DTS": "DTS_US_FT",
        "RHOB": "RHOB_G_CC",
        "GR": "GR_API",
        "PHIT": "PHIT_VV",
        "RT": "RT_OHM_M",
        "SW": "SW_VV",
        "CALI": "CALI_IN",
        "BS": "BS_IN",
        "DRHO": "DRHO_G_CC",
        "PEF": "PEF_B_E",
        "LITHCODE": "LITHOLOGY_CODE",
        "FLUIDCODE": "FLUID_CODE",
    }
    las.append_curve("DEPT", well["MD_M"].to_numpy(dtype=float), unit="M", descr="Measured depth")
    for mnemonic, unit, description in curves:
        las.append_curve(
            mnemonic,
            well[source_columns[mnemonic]].to_numpy(dtype=float, na_value=np.nan),
            unit=unit,
            descr=description,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    las.write(str(path), version=2.0, wrap=False)
    os.chmod(path, 0o600)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, na_rep="")
    os.chmod(path, 0o600)


def _make_report(
    output_dir: Path,
    config: dict[str, Any],
    earth: dict[str, Any],
    logs: pd.DataFrame,
    checkshot: pd.DataFrame,
    time_ms: np.ndarray,
    amplitudes: np.ndarray,
) -> None:
    x = earth["x_m"]
    y = earth["y_m"]
    top = earth["horizons"]["Top_Reservoir"]
    figure = go.Figure()
    figure.add_trace(
        go.Contour(
            x=x,
            y=y,
            z=top,
            colorscale="Viridis_r",
            contours={"showlabels": True, "labelfont": {"color": "white"}},
            colorbar={"title": "Depth (m)"},
            name="Top reservoir",
        )
    )
    for fault in config["faults"]:
        fault_y = np.linspace(float(y.min()), float(y.max()), 100)
        fault_x = float(fault["x_intercept_m"]) + float(fault["x_slope_per_y"]) * (
            fault_y - float(fault["reference_y_m"])
        )
        figure.add_trace(
            go.Scatter(
                x=fault_x,
                y=fault_y,
                mode="lines",
                name=fault["id"],
                line={"color": "#ff776e", "width": 3, "dash": "dash"},
            )
        )
    locations = pd.DataFrame(config["wells"])
    figure.add_trace(
        go.Scatter(
            x=locations["x_m"],
            y=locations["y_m"],
            mode="markers+text",
            text=locations["well_id"],
            textposition="top center",
            marker={"size": 11, "color": "#f7d56d", "line": {"color": "#172332", "width": 1}},
            name="Synthetic wells",
        )
    )
    figure.update_layout(
        title="Faulted reservoir top and seven synthetic wells",
        xaxis_title="Local X (m)",
        yaxis_title="Local Y (m)",
        yaxis={"scaleanchor": "x", "scaleratio": 1},
        template="plotly_dark",
        height=650,
    )
    pio.write_html(figure, output_dir / "structure_map.html", include_plotlyjs=True, full_html=True)

    central_well = config["wells"][4]
    ix = _nearest_index(x, central_well["x_m"])
    iy = _nearest_index(y, central_well["y_m"])
    angles = np.asarray(config["model"]["seismic_angles_deg"], dtype=float)
    gather = go.Figure(
        go.Heatmap(
            x=angles,
            y=time_ms,
            z=amplitudes[:, :, iy, ix],
            colorscale="RdBu",
            zmid=0,
            colorbar={"title": "Amplitude"},
            hovertemplate="TWT %{y:.1f} ms<br>Angle %{x:.0f}°<br>Amplitude %{z:.4f}<extra></extra>",
        )
    )
    gather.update_yaxes(autorange="reversed", title="Two-way travel time (ms)")
    gather.update_xaxes(title="Incident angle (degrees)")
    gather.update_layout(
        title=f"Synthetic pre-stack gather at {central_well['well_id']}",
        template="plotly_dark",
        height=650,
    )
    pio.write_html(
        gather, output_dir / "seismic_gather.html", include_plotlyjs=True, full_html=True
    )

    area_x = config["model"]["x_max_m"] - config["model"]["x_min_m"]
    area_y = config["model"]["y_max_m"] - config["model"]["y_min_m"]
    fault_summary = ", ".join(
        f"{fault['id']} ({fault['throw_m']} m throw)" for fault in config["faults"]
    )
    formation_summary = ", ".join(item["name"] for item in config["formations"])
    report_lines = [
        "# Synthetic Reservoir Conceptualization",
        "",
        f"> **Synthetic only.** {SYNTHETIC_WARNING} "
        "Do not use for operational or scientific decisions.",
        "",
        "## Scenario",
        "",
        config["scenario"]["description"],
        "",
        f"- Area: {area_x:.0f} m × {area_y:.0f} m; local coordinates only.",
        f"- Model grid: {len(x)} × {len(y)} lateral cells; {len(earth['z_m'])} depth samples.",
        f"- Wells: {len(config['wells'])}; vertical synthetic trajectories.",
        f"- Faults: {fault_summary}.",
        f"- Formations: {formation_summary}.",
        f"- Well-log rows: {len(logs):,}; checkshot points: {len(checkshot):,}.",
        f"- Pre-stack angle range: {min(angles):.0f}°–{max(angles):.0f}° "
        f"in {len(angles)} angle bins.",
        "",
        "## Geological model",
        "",
        "The model is a shallow anticline with two normal-fault offsets. Five formations "
        "are represented: upper shale, top seal, reservoir sand, basal shale, and tight "
        "carbonate. The sand contains gas, oil, and brine legs with synthetic contacts. "
        "Elastic properties are illustrative scenario values, not measurements.",
        "",
        "## Outputs",
        "",
        "- Aggregate CSV/Parquet logs with elastic, petrophysical, QC, and class labels.",
        "- Seven per-well LAS 2.0 files and sparse per-well checkshot tie points.",
        "- Gridded elastic/facies model, structural horizons, faults, and well locations.",
        "- Combined angle-coded CMP SEG-Y plus a regular 3D SEG-Y volume per angle.",
        "- `angle_gather_synthetic.npz` with near-well gathers for the dashboard.",
        "- Interactive `structure_map.html` and `seismic_gather.html` visualizations.",
        "",
        "## Seismic assumptions",
        "",
        f"The traces use Shuey three-term reflectivity and a "
        f"{config['model']['wavelet_frequency_hz']:.0f} Hz zero-phase Ricker wavelet, "
        "with independent seeded Gaussian noise. SEG-Y uses IEEE float, "
        f"{config['model']['seismic_sample_interval_ms']:.1f} ms sampling, and local "
        "synthetic coordinates. In the combined file, angle × 1000 is stored in "
        "trace-header unassigned integer word 233–236; separate angle volumes use "
        "regular inline/crossline geometry. This is not field acquisition data.",
        "",
        "## Intended workflow use",
        "",
        "The package tests ingestion, QC, time-depth mapping, rock-physics sampling, "
        "AVO forward modeling, and future Monte Carlo classification. Configured "
        "uncertainties are starting assumptions only; this generator performs no "
        "inversion or posterior calibration.",
    ]
    report = "\n".join(report_lines) + "\n"
    (output_dir / "README_synthetic.md").write_text(report, encoding="utf-8")
    os.chmod(output_dir / "README_synthetic.md", 0o600)


def _write_model_files(output_dir: Path, config: dict[str, Any], earth: dict[str, Any]) -> None:
    model_dir = output_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        model_dir / "earth_model_synthetic.npz",
        x_m=earth["x_m"],
        y_m=earth["y_m"],
        z_m=earth["z_m"],
        vp_m_s=earth["vp_m_s"],
        vs_m_s=earth["vs_m_s"],
        rho_g_cc=earth["rho_g_cc"],
        formation=earth["formation"],
        lithology=earth["lithology"],
        fluid=earth["fluid"],
        **{f"horizon_{key}_m": value for key, value in earth["horizons"].items()},
    )
    horizon_rows = []
    for name, grid in earth["horizons"].items():
        for iy, y in enumerate(earth["y_m"]):
            for ix, x in enumerate(earth["x_m"]):
                horizon_rows.append(
                    {
                        "HORIZON": name,
                        "X_M": float(x),
                        "Y_M": float(y),
                        "DEPTH_M": float(grid[iy, ix]),
                        "SYNTHETIC_DATA": True,
                    }
                )
    _write_csv(model_dir / "horizons_synthetic.csv", pd.DataFrame(horizon_rows))
    fault_rows = []
    for fault in config["faults"]:
        for y in earth["y_m"]:
            fault_rows.append(
                {
                    "FAULT_ID": fault["id"],
                    "NAME": fault["name"],
                    "X_M": float(
                        fault["x_intercept_m"]
                        + fault["x_slope_per_y"] * (float(y) - fault["reference_y_m"])
                    ),
                    "Y_M": float(y),
                    "THROW_M": float(fault["throw_m"]),
                    "DOWNTHROWN_SIDE": fault["downthrown_side"],
                    "SYNTHETIC_DATA": True,
                }
            )
    _write_csv(model_dir / "faults_synthetic.csv", pd.DataFrame(fault_rows))
    well_locations = pd.DataFrame(config["wells"])
    well_locations["SYNTHETIC_DATA"] = True
    _write_csv(model_dir / "wells_synthetic.csv", well_locations)
    os.chmod(model_dir / "earth_model_synthetic.npz", 0o600)


def _verify_segy(path: Path, expected_traces: int, expected_samples: int) -> None:
    with segyio.open(str(path), "r", ignore_geometry=True) as segy:
        if segy.tracecount != expected_traces or len(segy.samples) != expected_samples:
            raise ValueError(f"SEG-Y validation failed for {path.name}.")
        if not np.isfinite(np.asarray(segy.trace[0])).all():
            raise ValueError(f"SEG-Y contains non-finite amplitudes: {path.name}.")


def generate_reservoir(
    config_path: Path = DEFAULT_CONFIG,
    output_dir: Path = DEFAULT_OUTPUT,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate seven synthetic wells, faulted model, checkshots, and pre-stack SEG-Y."""
    config_path = config_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    config = load_config(config_path)
    output_dir.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if output_dir.exists():
        if not output_dir.is_dir():
            raise FileExistsError(f"Output path exists and is not a directory: {output_dir}.")
        if any(output_dir.iterdir()) and not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. Move it aside before regeneration."
            )
        if any(output_dir.iterdir()):
            manifest_path = output_dir / "synthetic_manifest.json"
            try:
                previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise FileExistsError(
                    "Refusing to overwrite a non-empty output without a valid synthetic manifest."
                ) from exc
            if previous_manifest.get("synthetic_data") is not True:
                raise FileExistsError("Refusing to overwrite an output not marked as synthetic.")
            shutil.rmtree(output_dir)
        else:
            output_dir.rmdir()
    temp_dir = Path(tempfile.mkdtemp(prefix=".synthetic-reservoir-", dir=output_dir.parent))
    try:
        rng = np.random.default_rng(int(config["scenario"]["random_seed"]))
        earth = _build_earth_model(config)
        logs, checkshot = _well_logs(config, earth, rng)
        time_ms, amplitudes = _generate_seismic(config, earth, rng)

        _write_csv(temp_dir / "well_logs_synthetic.csv", logs)
        pq.write_table(
            pa.Table.from_pandas(logs, preserve_index=False),
            temp_dir / "well_logs_synthetic.parquet",
            compression="zstd",
        )
        _write_csv(temp_dir / "checkshot_synthetic.csv", checkshot)
        pq.write_table(
            pa.Table.from_pandas(checkshot, preserve_index=False),
            temp_dir / "checkshot_synthetic.parquet",
            compression="zstd",
        )
        os.chmod(temp_dir / "well_logs_synthetic.parquet", 0o600)
        os.chmod(temp_dir / "checkshot_synthetic.parquet", 0o600)
        wells_dir = temp_dir / "wells"
        wells_dir.mkdir()
        for well in config["wells"]:
            _write_las(wells_dir / f"{well['well_id']}.las", logs, well["well_id"])
        _write_model_files(temp_dir, config, earth)

        seismic_dir = temp_dir / "seismic"
        seismic_dir.mkdir()
        dt_ms = float(config["model"]["seismic_sample_interval_ms"])
        angles = [int(value) for value in config["model"]["seismic_angles_deg"]]
        combined_path = seismic_dir / "prestack_angle_gather_synthetic.segy"
        _write_segy(
            combined_path,
            amplitudes,
            dt_ms,
            float(time_ms[0]),
            earth["x_m"],
            earth["y_m"],
            None,
            "SYNTHETIC PRESTACK CMP ANGLE GATHERS",
            angles_deg=angles,
        )
        _verify_segy(
            combined_path, len(earth["x_m"]) * len(earth["y_m"]) * len(angles), len(time_ms)
        )
        angle_files = []
        for angle_index, angle in enumerate(angles):
            path = seismic_dir / f"prestack_angle_{angle:02d}deg_synthetic.segy"
            _write_segy(
                path,
                amplitudes[:, angle_index, :, :],
                dt_ms,
                float(time_ms[0]),
                earth["x_m"],
                earth["y_m"],
                angle,
                f"SYNTHETIC {angle:02d} DEGREE PARTIAL STACK",
            )
            _verify_segy(path, len(earth["x_m"]) * len(earth["y_m"]), len(time_ms))
            angle_files.append(path.relative_to(temp_dir).as_posix())

        near_well_amplitudes = np.empty(
            (len(time_ms), len(angles), len(config["wells"])), dtype=np.float32
        )
        for well_index, well in enumerate(config["wells"]):
            ix = _nearest_index(earth["x_m"], well["x_m"])
            iy = _nearest_index(earth["y_m"], well["y_m"])
            near_well_amplitudes[:, :, well_index] = amplitudes[:, :, iy, ix]
        np.savez_compressed(
            temp_dir / "angle_gather_synthetic.npz",
            twt_ms=time_ms,
            incident_angle_deg=np.asarray(angles, dtype=np.float32),
            amplitude=near_well_amplitudes,
            lateral_trace_id=np.asarray([well["well_id"] for well in config["wells"]]),
        )
        os.chmod(temp_dir / "angle_gather_synthetic.npz", 0o600)
        _make_report(temp_dir, config, earth, logs, checkshot, time_ms, amplitudes)
        (temp_dir / "scenario_synthetic.yml").write_text(
            config_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
        os.chmod(temp_dir / "scenario_synthetic.yml", 0o600)

        file_paths = sorted(
            path.relative_to(temp_dir).as_posix()
            for path in temp_dir.rglob("*")
            if path.is_file() and path.name != "synthetic_manifest.json"
        )
        metadata = {
            "synthetic_data": True,
            "warning": SYNTHETIC_WARNING,
            "scenario": config["scenario"]["name"],
            "random_seed": int(config["scenario"]["random_seed"]),
            "generated_by": "src/synthetic_reservoir.py",
            "config_file": "scenario_synthetic.yml",
            "coordinate_system": "local synthetic Cartesian coordinates in metres; not geographic",
            "well_count": len(config["wells"]),
            "well_ids": [well["well_id"] for well in config["wells"]],
            "well_log_rows": int(len(logs)),
            "checkshot_tie_points": int(len(checkshot)),
            "formations": [item["name"] for item in config["formations"]],
            "faults": config["faults"],
            "grid": {
                "x_samples": int(len(earth["x_m"])),
                "y_samples": int(len(earth["y_m"])),
                "z_samples": int(len(earth["z_m"])),
                "x_range_m": [float(earth["x_m"].min()), float(earth["x_m"].max())],
                "y_range_m": [float(earth["y_m"].min()), float(earth["y_m"].max())],
                "z_range_m": [float(earth["z_m"].min()), float(earth["z_m"].max())],
            },
            "well_log_units": {
                "MD_M": "m",
                "TVD_M": "m",
                "TWT_MS": "ms",
                "DTC_US_FT": "us/ft",
                "DTS_US_FT": "us/ft",
                "RHOB_G_CC": "g/cc",
                "GR_API": "API",
                "PHIT_VV": "fraction",
                "RT_OHM_M": "ohm.m",
                "SW_VV": "fraction",
                "CALI_IN": "in",
                "BS_IN": "in",
                "DRHO_G_CC": "g/cc",
                "PEF_B_E": "barn/electron",
            },
            "seismic": {
                "combined_file": "seismic/prestack_angle_gather_synthetic.segy",
                "angle_volume_files": angle_files,
                "format": "SEG-Y revision 1, IEEE float32",
                "sample_interval_ms": dt_ms,
                "twt_start_ms": float(time_ms[0]),
                "twt_end_ms": float(time_ms[-1]),
                "angles_deg": angles,
                "combined_trace_count": int(len(earth["x_m"]) * len(earth["y_m"]) * len(angles)),
                "angle_volume_trace_count": int(len(earth["x_m"]) * len(earth["y_m"])),
                "angle_header": (
                    "Combined file: trace-header unassigned integer word 233-236 "
                    "stores angle in degrees x 1000."
                ),
                "method": (
                    "Shuey three-term reflectivity, 30-Hz zero-phase Ricker, "
                    "seeded Gaussian noise."
                ),
            },
            "files": file_paths,
        }
        (temp_dir / "synthetic_manifest.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        os.chmod(temp_dir / "synthetic_manifest.json", 0o600)
        for directory in sorted(
            (path for path in temp_dir.rglob("*") if path.is_dir()), reverse=True
        ):
            os.chmod(directory, 0o700)
        os.chmod(temp_dir, 0o700)
        os.replace(temp_dir, output_dir)
        return metadata
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
