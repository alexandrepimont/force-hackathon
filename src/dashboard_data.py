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


SEISMIC_STACKS = ("near", "mid", "far")


def find_seismic_stacks(directory: Path) -> dict[str, Path]:
    """Find near/mid/far SEG-Y stacks in the folder, its seismic/ subfolder or a sibling seismic/ folder."""
    root = directory.expanduser().resolve()
    for folder in (root / "seismic", root.parent / "seismic", root):
        if not folder.is_dir():
            continue
        found = {}
        for stack in SEISMIC_STACKS:
            matches = sorted(
                path
                for path in folder.iterdir()
                if path.is_file()
                and path.suffix.lower() in {".segy", ".sgy"}
                and stack in path.stem.lower()
            )
            if matches:
                found[stack] = matches[0]
        if found:
            return found
    return {}


def _cache_paths(segy_path: Path) -> tuple[Path, Path, Path]:
    cache = segy_path.parent / "cache"
    return (cache / f"{segy_path.stem}.f16.npy", cache / f"{segy_path.stem}.geometry.npz",
            cache / f"{segy_path.stem}.timeslices.f16.npy")


def build_seismic_cache(segy_path: Path, iline: int = 189, xline: int = 193, block: int = 64) -> Path:
    """Convert an inline-sorted regular 3D SEG-Y into a float16 (n_il, n_xl, ns) .npy plus geometry.

    SEG-Y is stored trace by trace, so time slices are very slow to read; the cache makes every view fast.
    Amplitudes are divided by a robust scale (kept in the geometry file) so they fit float16.
    """
    import segyio

    data_path, geometry_path, slices_path = _cache_paths(segy_path)
    with segyio.open(str(segy_path), iline=iline, xline=xline) as f:
        if f.sorting != segyio.TraceSortingFormat.INLINE_SORTING:
            raise ValueError("Only inline-sorted SEG-Y is supported for the dashboard cache.")
        ilines, xlines = np.asarray(f.ilines), np.asarray(f.xlines)
        samples = np.asarray(f.samples, dtype=float)
        n_il, n_xl, ns = len(ilines), len(xlines), len(samples)
        if f.tracecount != n_il * n_xl:
            raise ValueError("SEG-Y grid is not complete (missing traces).")
        fmt = int(f.bin[segyio.BinField.Format])
        ext_headers = int(f.ext_headers)
        sample_trace = f.trace.raw[: min(f.tracecount, 20000)]
        scale = float(np.percentile(np.abs(sample_trace), 99.9)) or 1.0

    offset = 3600 + 3200 * ext_headers
    if fmt == 5:
        trace_dtype = np.dtype([("header", "V240"), ("data", ">f4", (ns,))])
        raw = np.memmap(segy_path, dtype=trace_dtype, mode="r", offset=offset, shape=(n_il * n_xl,))
        header_dtype = np.dtype({"names": ["scalar", "x", "y"], "formats": [">i2", ">i4", ">i4"],
                                 "offsets": [70, 180, 184], "itemsize": 240})
        headers = raw["header"].view(header_dtype)
    else:
        raw, headers = None, None

    data_path.parent.mkdir(parents=True, exist_ok=True)
    out = np.lib.format.open_memmap(data_path, mode="w+", dtype=np.float16, shape=(n_il, n_xl, ns))
    out_t = np.lib.format.open_memmap(slices_path, mode="w+", dtype=np.float16, shape=(ns, n_il, n_xl))
    cdp_x = np.full((n_il, n_xl), np.nan)
    cdp_y = cdp_x.copy()
    with segyio.open(str(segy_path), iline=iline, xline=xline) as f:
        f.mmap()
        for start in range(0, n_il, block):
            stop = min(start + block, n_il)
            traces = slice(start * n_xl, stop * n_xl)
            if raw is not None:
                values = np.asarray(raw["data"][traces], dtype=np.float32)
                h = headers[traces]
                scalar, x, y = h["scalar"].astype(float), h["x"].astype(float), h["y"].astype(float)
            else:
                values = np.asarray(f.trace.raw[traces.start:traces.stop], dtype=np.float32)
                scalar = f.attributes(segyio.TraceField.SourceGroupScalar)[traces.start:traces.stop].astype(float)
                x = f.attributes(segyio.TraceField.CDP_X)[traces.start:traces.stop].astype(float)
                y = f.attributes(segyio.TraceField.CDP_Y)[traces.start:traces.stop].astype(float)
            factor = np.where(scalar < 0, 1.0 / np.abs(np.where(scalar == 0, 1, scalar)),
                              np.where(scalar == 0, 1.0, scalar))
            scaled = (values / scale).reshape(stop - start, n_xl, ns).astype(np.float16)
            out[start:stop] = scaled
            out_t[:, start:stop, :] = scaled.transpose(2, 0, 1)
            cdp_x[start:stop] = (x * factor).reshape(stop - start, n_xl)
            cdp_y[start:stop] = (y * factor).reshape(stop - start, n_xl)
    out.flush()
    out_t.flush()
    del out, out_t
    np.savez(geometry_path, ilines=ilines, xlines=xlines, samples=samples, cdp_x=cdp_x, cdp_y=cdp_y,
             scale=scale, source_mtime_ns=segy_path.stat().st_mtime_ns)
    return data_path


def seismic_cache_ready(segy_path: Path) -> bool:
    data_path, geometry_path, slices_path = _cache_paths(segy_path)
    if not all(path.is_file() for path in (data_path, geometry_path, slices_path)):
        return False
    with np.load(geometry_path) as geometry:
        return int(geometry["source_mtime_ns"]) == segy_path.stat().st_mtime_ns


def read_seismic_geometry(segy_path: Path) -> dict[str, np.ndarray]:
    """Axes, CDP X/Y grid and amplitude scale from the dashboard cache of a SEG-Y stack."""
    with np.load(_cache_paths(segy_path)[1]) as geometry:
        return {key: geometry[key].copy() for key in geometry.files}


def read_seismic_section(segy_path: Path, kind: str, value: float) -> np.ndarray:
    """Inline (n_xl, ns), crossline (n_il, ns) or time slice (n_il, n_xl) from the cache, in SEG-Y units."""
    data_path, _, slices_path = _cache_paths(segy_path)
    geometry = read_seismic_geometry(segy_path)
    if kind == "inline":
        section = np.load(data_path, mmap_mode="r")[int(np.searchsorted(geometry["ilines"], value))]
    elif kind == "crossline":
        section = np.load(data_path, mmap_mode="r")[:, int(np.searchsorted(geometry["xlines"], value))]
    elif kind == "time":
        section = np.load(slices_path, mmap_mode="r")[int(np.argmin(np.abs(geometry["samples"] - value)))]
    else:
        raise ValueError(f"Unsupported section kind: {kind}")
    return np.asarray(section, dtype=np.float32) * float(geometry["scale"])


def read_seismic_traces(segy_path: Path, i_idx: np.ndarray, j_idx: np.ndarray) -> np.ndarray:
    """Traces (n, ns) at grid indices (i_idx, j_idx) from the cache, in SEG-Y units."""
    data_path = _cache_paths(segy_path)[0]
    geometry = read_seismic_geometry(segy_path)
    cube = np.load(data_path, mmap_mode="r")
    order = np.lexsort((j_idx, i_idx))
    traces = np.empty((len(i_idx), cube.shape[2]), dtype=np.float32)
    traces[order] = np.asarray(cube[i_idx[order], j_idx[order]], dtype=np.float32)
    return traces * float(geometry["scale"])


def traverse_indices(points_ilxl: list[tuple[float, float]], ilines: np.ndarray, xlines: np.ndarray,
                     geometry: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Grid indices along straight segments between (inline, crossline) points, one sample per trace step.

    Returns i/j indices, inline/crossline numbers, distance along the path (m) and the index of each vertex.
    """
    pts = [(float(np.interp(il, ilines, np.arange(len(ilines)))), float(np.interp(xl, xlines, np.arange(len(xlines)))))
           for il, xl in points_ilxl]
    ii, jj, vertex = [], [], [0]
    for (i0, j0), (i1, j1) in zip(pts[:-1], pts[1:]):
        n = int(max(abs(i1 - i0), abs(j1 - j0))) + 1
        seg_i, seg_j = np.linspace(i0, i1, n), np.linspace(j0, j1, n)
        if ii:
            seg_i, seg_j = seg_i[1:], seg_j[1:]
        ii.extend(seg_i)
        jj.extend(seg_j)
        vertex.append(len(ii) - 1)
    i_idx = np.clip(np.rint(ii).astype(int), 0, len(ilines) - 1)
    j_idx = np.clip(np.rint(jj).astype(int), 0, len(xlines) - 1)
    x, y = geometry["cdp_x"][i_idx, j_idx], geometry["cdp_y"][i_idx, j_idx]
    distance = np.r_[0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
    return {"i": i_idx, "j": j_idx, "inline": ilines[i_idx], "crossline": xlines[j_idx], "x": x, "y": y,
            "distance_m": distance, "vertex": np.asarray(vertex)}


def survey_outline(geometry: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """X, Y of the cube's four corners (closed polygon)."""
    x, y = geometry["cdp_x"], geometry["cdp_y"]
    corners = [(0, 0), (0, -1), (-1, -1), (-1, 0), (0, 0)]
    return np.array([x[i, j] for i, j in corners]), np.array([y[i, j] for i, j in corners])


def wells_on_grid(geometry: dict[str, np.ndarray], wells: pd.DataFrame) -> pd.DataFrame:
    """Nearest inline/crossline for each well (X_M, Y_M); wells outside the grid get NaN."""
    x, y = geometry["cdp_x"], geometry["cdp_y"]
    valid = np.isfinite(x) & np.isfinite(y)
    rows = []
    spacing = np.nanmedian(np.hypot(np.diff(x, axis=1), np.diff(y, axis=1)))
    for _, well in wells.iterrows():
        distance = np.where(valid, np.hypot(x - well["X_M"], y - well["Y_M"]), np.inf)
        i, j = np.unravel_index(np.argmin(distance), distance.shape)
        inside = distance[i, j] <= 2 * spacing
        rows.append({
            "WELL_ID": well["WELL_ID"],
            "INLINE": int(geometry["ilines"][i]) if inside else np.nan,
            "CROSSLINE": int(geometry["xlines"][j]) if inside else np.nan,
            "DISTANCE_M": float(distance[i, j]),
        })
    return pd.DataFrame(rows)
