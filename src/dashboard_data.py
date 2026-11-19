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
    """List known matching tables and safe CSV/Parquet fallbacks in a folder."""
    if table_kind not in TABLE_CANDIDATES:
        raise ValueError(f"Unsupported table kind: {table_kind}")
    root = directory.expanduser().resolve()
    known = [root / name for name in TABLE_CANDIDATES[table_kind] if (root / name).is_file()]
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
    """Read a local CSV or Parquet table and normalize basic depth/time types."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif suffix == ".csv":
        frame = pd.read_csv(path)
    else:
        raise ValueError("Dashboard supports CSV and Parquet tables.")

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
