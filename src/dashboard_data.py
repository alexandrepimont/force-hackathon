"""Local data discovery and validation helpers for the AVO dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TABLE_CANDIDATES = {
    "well_logs": (
        "well_logs_synthetic.parquet",
        "well_logs_synthetic.csv",
        "well_logs.parquet",
        "well_logs.csv",
    ),
    "checkshot": (
        "checkshot_synthetic.parquet",
        "checkshot_synthetic.csv",
        "checkshot.parquet",
        "checkshot.csv",
    ),
}


def table_candidates(directory: Path, table_kind: str) -> list[Path]:
    """List known matching tables and safe supported fallbacks in a folder."""
    if table_kind not in TABLE_CANDIDATES:
        raise ValueError(f"Unsupported table kind: {table_kind}")
    root = directory.expanduser().resolve()
    known = [root / name for name in TABLE_CANDIDATES[table_kind] if (root / name).is_file()]
    if table_kind == "well_logs":
        las_files = sorted(
            {
                *root.glob("*.las"),
                *root.glob("*.LAS"),
                *root.glob("wells/*.las"),
                *root.glob("wells/*.LAS"),
            },
            key=lambda path: path.name.lower(),
        )
        if known:
            return [*known, *las_files]
        excluded = ("checkshot", "seismic", "gather", "manifest")
        fallback = [
            path
            for path in root.glob("*.*")
            if path.suffix.lower() in {".csv", ".parquet"}
            and not any(marker in path.name.lower() for marker in excluded)
        ]
        table_files = sorted(
            fallback,
            key=lambda path: (path.suffix.lower() != ".parquet", path.name.lower()),
        )
        return [*table_files, *las_files]
    if known:
        return known

    if table_kind == "checkshot":
        fallback = [path for path in root.glob("*.*") if "checkshot" in path.name.lower()]
    else:
        excluded = ("checkshot", "seismic", "gather", "manifest")
        fallback = [
            path
            for path in root.glob("*.*")
            if path.suffix.lower() in {".csv", ".parquet"}
            and not any(marker in path.name.lower() for marker in excluded)
        ]
    return sorted(
        (path for path in fallback if path.suffix.lower() in {".csv", ".parquet"}),
        key=lambda path: (path.suffix.lower() != ".parquet", path.name.lower()),
    )


def find_table(directory: Path, table_kind: str) -> Path | None:
    """Return a sole matching table, or a known-name match when several exist."""
    candidates = table_candidates(directory, table_kind)
    if not candidates:
        return None
    if len(candidates) == 1 or candidates[0].name in TABLE_CANDIDATES[table_kind]:
        return candidates[0]
    return None


def read_table(path: Path) -> pd.DataFrame:
    """Read a local CSV, Parquet, or LAS table and normalize common curve names."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif suffix == ".csv":
        frame = pd.read_csv(path)
    elif suffix == ".las":
        import lasio

        las = lasio.read(str(path))
        frame = las.df().reset_index()
        aliases = {
            "DEPT": "MD_M",
            "DEPTH": "MD_M",
            "MD": "MD_M",
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
            "DRHO": "DRHO_G_CC",
            "PEF": "PEF_B_E",
            "LITHCODE": "LITHOLOGY_CODE",
            "FLUIDCODE": "FLUID_CODE",
        }
        frame.columns = [str(column).strip().upper() for column in frame.columns]
        frame = frame.rename(columns=aliases)
        frame["WELL_ID"] = str(las.well.WELL.value or path.stem).strip()
        provenance = " ".join(str(item.value or "") for item in (las.well.COMP, las.well.PROV))
        if "synthetic" in provenance.lower():
            frame["SYNTHETIC_DATA"] = True
        if "LITHOLOGY_CODE" in frame:
            frame["SYNTHETIC_LITHOLOGY"] = frame["LITHOLOGY_CODE"].map(
                {1: "shale", 2: "sand", 3: "tight"}
            )
        if "FLUID_CODE" in frame:
            frame["SYNTHETIC_FLUID"] = frame["FLUID_CODE"].map(
                {0: "not_applicable", 1: "brine", 2: "oil", 3: "gas"}
            )
    else:
        raise ValueError("Dashboard supports CSV, Parquet, and LAS well-log files.")

    for column in ("MD_M", "TVD_M", "TWT_MS"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "QC_BAD_HOLE" in frame:
        if pd.api.types.is_bool_dtype(frame["QC_BAD_HOLE"]):
            frame["QC_BAD_HOLE"] = frame["QC_BAD_HOLE"].fillna(False)
        else:
            frame["QC_BAD_HOLE"] = frame["QC_BAD_HOLE"].map(
                lambda value: str(value).strip().lower() in {"true", "1", "yes", "y"}
            )
    return frame


def read_manifest(directory: Path) -> dict[str, Any]:
    """Read the local synthetic/retrieval manifest when present."""
    candidates = (directory / "synthetic_manifest.json", directory / "manifest.json")
    for path in candidates:
        if path.is_file():
            with path.open(encoding="utf-8") as stream:
                value = json.load(stream)
            if isinstance(value, dict):
                return value
    return {}


def read_gather(path: Path) -> dict[str, np.ndarray]:
    """Read and validate the safe, local NPZ angle-gather array."""
    with np.load(path, allow_pickle=False) as archive:
        required = {"twt_ms", "incident_angle_deg", "amplitude", "lateral_trace_id"}
        if not required.issubset(archive.files):
            raise ValueError("The NPZ file does not contain the expected angle-gather arrays.")
        result = {key: archive[key].copy() for key in required}

    amplitudes = result["amplitude"]
    if amplitudes.ndim != 3:
        raise ValueError("Gather amplitude must have time × angle × lateral-trace dimensions.")
    if amplitudes.shape != (
        len(result["twt_ms"]),
        len(result["incident_angle_deg"]),
        len(result["lateral_trace_id"]),
    ):
        raise ValueError("Gather coordinates do not match the amplitude array dimensions.")
    if not np.isfinite(amplitudes).all():
        raise ValueError("Gather amplitudes contain non-finite values.")
    return result


def read_reservoir_model(path: Path) -> dict[str, np.ndarray]:
    """Read and validate the safe, local synthetic reservoir model arrays."""
    required = {
        "x_m",
        "y_m",
        "z_m",
        "vp_m_s",
        "vs_m_s",
        "rho_g_cc",
        "formation",
        "lithology",
        "fluid",
        "horizon_Upper_Shale_Base_m",
        "horizon_Top_Reservoir_m",
        "horizon_Base_Reservoir_m",
        "horizon_Base_Basal_Shale_m",
    }
    with np.load(path, allow_pickle=False) as archive:
        if not required.issubset(archive.files):
            raise ValueError("The NPZ file does not contain the expected reservoir-model arrays.")
        result = {key: archive[key].copy() for key in archive.files}

    x, y, z = result["x_m"], result["y_m"], result["z_m"]
    shape = (len(z), len(y), len(x))
    for key in ("vp_m_s", "vs_m_s", "rho_g_cc", "formation", "lithology", "fluid"):
        if result[key].shape != shape:
            raise ValueError(f"Reservoir array {key} does not match the depth × Y × X grid.")
    for key in required - {"x_m", "y_m", "z_m", "formation", "lithology", "fluid"}:
        values = result[key]
        expected_shape = shape[1:] if key.startswith("horizon_") else None
        if expected_shape is not None and values.shape != expected_shape:
            raise ValueError(f"Reservoir horizon {key} does not match the Y × X grid.")
        if values.dtype.kind not in "US" and not np.isfinite(values).all():
            raise ValueError(f"Reservoir model array {key} contains non-finite values.")
    for axis, name in ((x, "X"), (y, "Y"), (z, "depth")):
        if axis.ndim != 1 or len(axis) < 2 or not np.all(np.diff(axis) > 0):
            raise ValueError(
                f"Reservoir {name} coordinates must be one-dimensional and increasing."
            )
    return result


def is_synthetic(frame: pd.DataFrame, manifest: dict[str, Any]) -> bool:
    """Return true if either column-level or manifest provenance marks synthetic data."""
    if manifest.get("synthetic_data") is True:
        return True
    if "SYNTHETIC_DATA" in frame:
        values = frame["SYNTHETIC_DATA"]
        if pd.api.types.is_bool_dtype(values):
            return bool(values.fillna(False).any())
        return bool(
            values.map(lambda value: str(value).strip().lower() in {"true", "1", "yes", "y"}).any()
        )
    return False
