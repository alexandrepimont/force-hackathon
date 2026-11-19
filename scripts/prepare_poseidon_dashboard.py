"""Build the dashboard dataset for the real Poseidon 3D wells.

Reads the files fetched by scripts/download_poseidon_wells.py and writes data/poseidon/dashboard/:
  well_logs.parquet   all wells in one table with standard curve names (see CURVE_MAP), elastic
                      properties (VP_M_S, VS_M_S, AI, VPVS), TVDSS_M and FORMATION from the tops
  well_tops.csv       formation tops with MD_M and TVDSS_M
  wells.csv           well locations (GDA94 / MGA zone 51) and rig datum elevation
  manifest.json       provenance

Wells are treated as vertical (header TVDSS + datum matches TD MD within a few metres), so
TVDSS_M = MD_M - datum elevation.

    python scripts/prepare_poseidon_dashboard.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import lasio
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard_data import build_seismic_cache, find_seismic_stacks, seismic_cache_ready  # noqa: E402

WELLS_DIR = PROJECT_ROOT / "data" / "poseidon" / "wells"
OUT_DIR = PROJECT_ROOT / "data" / "poseidon" / "dashboard"

# Source mnemonic -> standard column, first match wins per well.
CURVE_MAP = {
    "GR_API": ("ECGR", "GR", "GRD", "GRARC"),
    "RT_OHM_M": ("RD", "ATRT", "P34H"),
    "RHOB_G_CC": ("RHOB", "RHOZ", "HROM"),
    "NPHI_PU": ("TNPH", "NPHI", "TNP", "HTNP"),
    "DTC_US_FT": ("DTCO", "DTC", "BATC"),
    "DTS_US_FT": ("DTSM", "DTS"),
    "CALI_IN": ("HDAR", "CAL1", "DCAV"),
}

# t20 header names -> well ids used in folders and tops
HEADER_NAMES = {
    "Boreas_1": "Boreas-1", "Kronons_1": "Kronos-1", "Pharos_1": "Pharos-1", "Poseidon_1": "Poseidon-1",
    "Poseidon_2": "Poseidon-2", "Poseidon_North": "Poseidon-North-1", "Proteus_1": "Proteus-1",
    "Torosa_1": "Torosa-1",
}

US_FT_TO_M_S = 304_800.0


def read_well(path: Path, well: str) -> pd.DataFrame:
    las = lasio.read(str(path))
    raw = las.df()
    raw.columns = [str(column).upper() for column in raw.columns]
    frame = pd.DataFrame({"WELL_ID": well, "MD_M": raw.index.to_numpy(dtype=float)})
    for column, mnemonics in CURVE_MAP.items():
        source = next((name for name in mnemonics if name in raw.columns), None)
        frame[column] = raw[source].to_numpy(dtype=float) if source else np.nan
    return frame


def add_elastic(frame: pd.DataFrame) -> pd.DataFrame:
    dtc = frame["DTC_US_FT"].where(frame["DTC_US_FT"].between(40, 240))
    dts = frame["DTS_US_FT"].where(frame["DTS_US_FT"].between(60, 600))
    rhob = frame["RHOB_G_CC"].where(frame["RHOB_G_CC"].between(1.5, 3.2))
    frame["VP_M_S"] = US_FT_TO_M_S / dtc
    frame["VS_M_S"] = US_FT_TO_M_S / dts
    frame["AI"] = frame["VP_M_S"] * rhob
    frame["VPVS"] = frame["VP_M_S"] / frame["VS_M_S"]
    frame["NPHI_VV"] = frame.pop("NPHI_PU") / 100.0
    return frame


def assign_formation(frame: pd.DataFrame, tops: pd.DataFrame) -> pd.Series:
    labels = pd.Series(pd.NA, index=frame.index, dtype="object")
    for well, well_tops in tops.groupby("WELL_ID"):
        mask = frame["WELL_ID"] == well
        ordered = well_tops.sort_values("MD_M")
        positions = np.searchsorted(ordered["MD_M"].to_numpy(), frame.loc[mask, "MD_M"].to_numpy(), "right") - 1
        names = ordered["SURFACE"].to_numpy()
        labels.loc[mask] = np.where(positions >= 0, names[np.clip(positions, 0, None)], "Above first top")
    return labels.fillna("No tops available")


def main() -> int:
    log_files = sorted(WELLS_DIR.glob("logs/*/*.LAS")) + sorted(WELLS_DIR.glob("logs/*/*.las"))
    headers_path = WELLS_DIR / "tops" / "t20" / "Well_Headers.csv"
    tops_path = WELLS_DIR / "tops" / "well_tops.csv"
    missing = [str(p) for p in (headers_path, tops_path) if not p.is_file()]
    if not log_files or missing:
        print("Run scripts/download_poseidon_wells.py first; missing: " + ", ".join(missing or ["logs"]))
        return 1

    headers = pd.read_csv(headers_path)
    wells = pd.DataFrame({
        "WELL_ID": headers["Name"].map(HEADER_NAMES),
        "X_M": headers["X"], "Y_M": headers["Y"], "LAT": headers["Lat"], "LON": headers["Lon"],
        "DATUM_ELEV_M": headers["DFE"], "TD_MD_M": headers["TD_(MD)"],
    }).dropna(subset=["WELL_ID"])
    datum = wells.set_index("WELL_ID")["DATUM_ELEV_M"]

    logs = pd.concat([read_well(path, path.parent.name) for path in log_files], ignore_index=True)
    logs = add_elastic(logs)
    logs["TVDSS_M"] = logs["MD_M"] - logs["WELL_ID"].map(datum)

    tops = pd.read_csv(tops_path).rename(columns={"WELL": "WELL_ID"})
    tops = tops[tops["WELL_ID"].isin(set(logs["WELL_ID"]))].copy()
    tops["TVDSS_M"] = tops["MD_M"] - tops["WELL_ID"].map(datum)
    logs["FORMATION"] = assign_formation(logs, tops)

    curve_columns = [c for c in logs.columns if c not in {"WELL_ID", "MD_M", "TVDSS_M", "FORMATION"}]
    logs = logs.dropna(subset=curve_columns, how="all")
    ordered = ["WELL_ID", "MD_M", "TVDSS_M", "FORMATION", *curve_columns]
    logs = logs[ordered].sort_values(["WELL_ID", "MD_M"]).reset_index(drop=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logs.to_parquet(OUT_DIR / "well_logs.parquet", index=False)
    tops.to_csv(OUT_DIR / "well_tops.csv", index=False)
    wells.to_csv(OUT_DIR / "wells.csv", index=False)
    manifest = {
        "synthetic_data": False,
        "dataset": "Poseidon 3D wells, Browse Basin, NW Shelf Australia",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "license": "Geoscience Australia open-file data, CC BY 3.0 AU",
        "sources": {
            "logs": "Occam Technology spliced composite LAS via github.com/WesleyTheGeolien/t20-Dash_Well_Viz",
            "tops": "Well completion reports (t20 repo) and Geoscience Australia Boreholes WFS",
            "well_headers": "t20 repo Well_Headers.csv (GDA94 / MGA zone 51)",
        },
        "assumptions": ["Wells treated as vertical: TVDSS_M = MD_M - DATUM_ELEV_M."],
        "curve_sources": {column: list(names) for column, names in CURVE_MAP.items()},
        "wells": sorted(logs["WELL_ID"].unique().tolist()),
        "wells_without_tops": sorted(set(logs["WELL_ID"]) - set(tops["WELL_ID"])),
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    coverage = logs.groupby("WELL_ID")[["GR_API", "RHOB_G_CC", "DTC_US_FT", "DTS_US_FT", "AI", "VPVS"]].count()
    print(coverage.to_string())
    print(f"\n{len(logs):,} rows -> {OUT_DIR}")

    for stack, segy_path in find_seismic_stacks(OUT_DIR).items():
        if seismic_cache_ready(segy_path):
            print(f"Seismic display cache for {stack} is up to date")
            continue
        print(f"Building seismic display cache for {stack} (about 1 minute)...")
        build_seismic_cache(segy_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
