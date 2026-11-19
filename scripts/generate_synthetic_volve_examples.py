"""Generate deterministic, clearly labeled synthetic inputs for the AVO workflow.

These examples are scenario-based and are not derived from real Volve data.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SEED = 20261119
SYNTHETIC_FLAG = True
WELL_ID = "SYNTHETIC-WELL-001"
DEFAULT_OUTPUT = Path("data/volve/examples")

FACIES = {
    "shale": {
        "vp": 3050.0,
        "vs": 1600.0,
        "rho": 2.38,
        "gr": 92.0,
        "phi": 0.13,
        "rt": 3.2,
        "pef": 3.4,
    },
    "brine_sand": {
        "vp": 3650.0,
        "vs": 2075.0,
        "rho": 2.18,
        "gr": 24.0,
        "phi": 0.24,
        "rt": 1.7,
        "pef": 2.2,
    },
    "oil_sand": {
        "vp": 3425.0,
        "vs": 1980.0,
        "rho": 2.12,
        "gr": 27.0,
        "phi": 0.25,
        "rt": 18.0,
        "pef": 2.2,
    },
    "gas_sand": {
        "vp": 3100.0,
        "vs": 1880.0,
        "rho": 2.04,
        "gr": 22.0,
        "phi": 0.27,
        "rt": 48.0,
        "pef": 2.1,
    },
    "tight_carbonate": {
        "vp": 5050.0,
        "vs": 2925.0,
        "rho": 2.66,
        "gr": 10.0,
        "phi": 0.055,
        "rt": 65.0,
        "pef": 4.7,
    },
}

LAYERS = (
    (2100.0, 2160.0, "shale", "not_applicable"),
    (2160.0, 2172.0, "brine_sand", "brine"),
    (2172.0, 2210.0, "shale", "not_applicable"),
    (2210.0, 2224.0, "oil_sand", "oil"),
    (2224.0, 2260.0, "shale", "not_applicable"),
    (2260.0, 2269.0, "gas_sand", "gas"),
    (2269.0, 2320.0, "shale", "not_applicable"),
    (2320.0, 2335.0, "tight_carbonate", "not_applicable"),
    (2335.0, 2400.0, "shale", "not_applicable"),
)


def _clip_normal(
    rng: np.random.Generator, mean: float, sigma: float, low: float, high: float
) -> float:
    return float(np.clip(rng.normal(mean, sigma), low, high))


def _facies_at(depth_m: float) -> tuple[str, str]:
    for top, base, lithology, fluid in LAYERS:
        if top <= depth_m < base:
            return lithology, fluid
    return "shale", "not_applicable"


def _generate_logs(rng: np.random.Generator) -> list[dict[str, Any]]:
    """Create a depth-sampled well-log scenario with smooth trends and QC defects."""
    md_values = np.arange(2100.0, 2400.0 + 0.0762, 0.1524)
    rows: list[dict[str, Any]] = []
    previous_twt_ms = 1400.0

    for index, md_m in enumerate(md_values):
        lithology, fluid = _facies_at(float(md_m))
        base = FACIES[lithology]
        trend = math.sin((float(md_m) - 2100.0) / 37.0)
        vp = _clip_normal(rng, base["vp"] + 35.0 * trend, 75.0, 2300.0, 5600.0)
        vs = _clip_normal(rng, base["vs"] + 25.0 * trend, 55.0, 1100.0, 3400.0)
        rho = _clip_normal(rng, base["rho"] + 0.012 * trend, 0.025, 1.85, 2.85)
        gr = _clip_normal(rng, base["gr"] + 3.0 * trend, 5.0, 2.0, 150.0)
        phi = _clip_normal(rng, base["phi"] - 0.006 * trend, 0.012, 0.01, 0.38)
        rt = float(np.exp(rng.normal(math.log(base["rt"]), 0.16)))
        pef = _clip_normal(rng, base["pef"], 0.12, 1.2, 6.0)
        dtc = 304800.0 / vp
        dts = 304800.0 / vs

        # A gentle deviation keeps MD/TVD distinct without introducing a sharp trajectory kink.
        tvd_m = float(md_m - 4.0 * math.sin((float(md_m) - 2100.0) / 250.0))
        if index:
            dz_m = tvd_m - rows[-1]["TVD_M"]
            previous_twt_ms += 2000.0 * dz_m / vp

        bad_hole = 2138.0 <= md_m < 2142.0 or 2305.0 <= md_m < 2308.0
        cali = _clip_normal(rng, 12.25 if bad_hole else 8.5, 0.35 if bad_hole else 0.12, 8.0, 15.5)
        drho = _clip_normal(
            rng, 0.18 if bad_hole else 0.015, 0.035 if bad_hole else 0.012, -0.05, 0.35
        )
        if bad_hole:
            rho += 0.08

        row: dict[str, Any] = {
            "WELL_ID": WELL_ID,
            "MD_M": round(float(md_m), 3),
            "TVD_M": round(tvd_m, 3),
            "TWT_MS": round(previous_twt_ms, 3),
            "DTC_US_FT": round(dtc, 3),
            "DTS_US_FT": round(dts, 3),
            "RHOB_G_CC": round(rho, 4),
            "GR_API": round(gr, 2),
            "PHIT_VV": round(phi, 4),
            "RT_OHM_M": round(rt, 3),
            "CALI_IN": round(cali, 3),
            "BS_IN": 8.5,
            "DRHO_G_CC": round(drho, 4),
            "PEF_B_E": round(pef, 3),
            "SYNTHETIC_LITHOLOGY": lithology,
            "SYNTHETIC_FLUID": fluid,
            "QC_BAD_HOLE": bad_hole,
            "SYNTHETIC_DATA": SYNTHETIC_FLAG,
        }
        # Modest curve gaps mimic routine log coverage; labels/QC flags remain available.
        if rng.random() < 0.004:
            row["DTS_US_FT"] = None
        if rng.random() < 0.008:
            row["RT_OHM_M"] = None
        if rng.random() < 0.006:
            row["PHIT_VV"] = None
        rows.append(row)
    return rows


def _generate_checkshot(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Downsample the synthetic time-depth relationship to sparse tie points."""
    selected = rows[::100]
    if selected[-1] is not rows[-1]:
        selected.append(rows[-1])
    return [
        {
            "WELL_ID": row["WELL_ID"],
            "MD_M": row["MD_M"],
            "TVD_M": row["TVD_M"],
            "TWT_MS": row["TWT_MS"],
            "SYNTHETIC_DATA": True,
        }
        for row in selected
    ]


def _ricker(frequency_hz: float, dt_s: float, half_length_s: float) -> np.ndarray:
    time_s = np.arange(-half_length_s, half_length_s + dt_s / 2.0, dt_s)
    term = (np.pi * frequency_hz * time_s) ** 2
    return (1.0 - 2.0 * term) * np.exp(-term)


def _generate_gather(rows: list[dict[str, Any]], rng: np.random.Generator) -> dict[str, Any]:
    """Build a small Shuey-style synthetic angle gather for calibration examples."""
    dt_ms = 2.0
    angles_deg = np.array([5.0, 15.0, 25.0, 35.0, 45.0], dtype=np.float32)
    lateral_traces = 3
    twt_values = np.array([row["TWT_MS"] for row in rows], dtype=float)
    time_ms = np.arange(
        math.floor(twt_values.min() - 80.0), math.ceil(twt_values.max() + 80.0), dt_ms
    )
    reflectivity = np.zeros((len(time_ms), len(angles_deg)), dtype=np.float64)

    for index in range(1, len(rows)):
        if rows[index]["SYNTHETIC_LITHOLOGY"] == rows[index - 1]["SYNTHETIC_LITHOLOGY"]:
            continue
        upper = rows[max(0, index - 8) : index]
        lower = rows[index : min(len(rows), index + 8)]
        vp1 = float(np.mean([304800.0 / row["DTC_US_FT"] for row in upper]))
        vp2 = float(np.mean([304800.0 / row["DTC_US_FT"] for row in lower]))
        vs1 = float(np.mean([304800.0 / row["DTS_US_FT"] for row in upper if row["DTS_US_FT"]]))
        vs2 = float(np.mean([304800.0 / row["DTS_US_FT"] for row in lower if row["DTS_US_FT"]]))
        rho1 = float(np.mean([row["RHOB_G_CC"] for row in upper]))
        rho2 = float(np.mean([row["RHOB_G_CC"] for row in lower]))
        vp_avg = 0.5 * (vp1 + vp2)
        vs_avg = 0.5 * (vs1 + vs2)
        rho_avg = 0.5 * (rho1 + rho2)
        dvp = (vp2 - vp1) / vp_avg
        dvs = (vs2 - vs1) / vs_avg
        drho = (rho2 - rho1) / rho_avg
        intercept = 0.5 * (dvp + drho)
        gradient = 0.5 * dvp - 2.0 * (vs_avg / vp_avg) ** 2 * (2.0 * dvs + drho)
        curvature = 0.5 * dvp
        event_time = 0.5 * (rows[index - 1]["TWT_MS"] + rows[index]["TWT_MS"])
        sample = int(round((event_time - time_ms[0]) / dt_ms))
        if 0 <= sample < len(time_ms):
            for angle_index, angle_deg in enumerate(angles_deg):
                theta = np.deg2rad(float(angle_deg))
                sin2 = np.sin(theta) ** 2
                tan2 = np.tan(theta) ** 2
                reflectivity[sample, angle_index] += (
                    intercept + gradient * sin2 + curvature * (tan2 - sin2)
                )

    wavelet = _ricker(30.0, dt_ms / 1000.0, 0.060)
    gather = np.empty((len(time_ms), len(angles_deg), lateral_traces), dtype=np.float32)
    for angle_index in range(len(angles_deg)):
        convolved = np.convolve(reflectivity[:, angle_index], wavelet, mode="same")
        for trace_index in range(lateral_traces):
            scale = 1.0 + (trace_index - 1) * 0.06
            noise = rng.normal(0.0, 0.008, size=len(time_ms))
            gather[:, angle_index, trace_index] = (scale * convolved + noise).astype(np.float32)

    return {
        "twt_ms": time_ms.astype(np.float32),
        "incident_angle_deg": angles_deg,
        "amplitude": gather,
        "lateral_trace_id": np.array(["SYN-TRACE-001", "SYN-TRACE-002", "SYN-TRACE-003"]),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.chmod(path, 0o600)


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path, compression="zstd")
    os.chmod(path, 0o600)


def generate_examples(output_dir: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Generate the complete synthetic example package into a new local directory."""
    output_dir = output_dir.expanduser().resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if output_dir.exists():
        if not output_dir.is_dir() or any(output_dir.iterdir()):
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. "
                "Move it aside before regenerating examples."
            )
        output_dir.rmdir()

    temp_dir = Path(tempfile.mkdtemp(prefix=".synthetic-volve-", dir=output_dir.parent))
    try:
        rng = np.random.default_rng(SEED)
        logs = _generate_logs(rng)
        checkshot = _generate_checkshot(logs)
        gather = _generate_gather(logs, rng)

        _write_csv(temp_dir / "well_logs_synthetic.csv", logs)
        _write_parquet(temp_dir / "well_logs_synthetic.parquet", logs)
        _write_csv(temp_dir / "checkshot_synthetic.csv", checkshot)
        _write_parquet(temp_dir / "checkshot_synthetic.parquet", checkshot)
        np.savez_compressed(temp_dir / "angle_gather_synthetic.npz", **gather)
        os.chmod(temp_dir / "angle_gather_synthetic.npz", 0o600)

        metadata = {
            "synthetic_data": True,
            "warning": (
                "Entirely synthetic scenario; not derived from or representative "
                "of real Volve measurements."
            ),
            "well_id": WELL_ID,
            "random_seed": SEED,
            "generated_by": "scripts/generate_synthetic_volve_examples.py",
            "depth_range_m": [logs[0]["MD_M"], logs[-1]["MD_M"]],
            "log_sample_interval_m": 0.1524,
            "log_rows": len(logs),
            "checkshot_tie_points": len(checkshot),
            "checkshot_fields": ["MD_M", "TVD_M", "TWT_MS"],
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
                "CALI_IN": "in",
                "BS_IN": "in",
                "DRHO_G_CC": "g/cc",
                "PEF_B_E": "barn/electron",
            },
            "facies_scenarios": [
                "shale",
                "brine_sand",
                "oil_sand",
                "gas_sand",
                "tight_carbonate",
            ],
            "quality_examples": [
                "sparse curve nulls",
                "two synthetic washout intervals flagged by QC_BAD_HOLE",
            ],
            "seismic": {
                "file": "angle_gather_synthetic.npz",
                "method": (
                    "Shuey three-term reflectivity approximation convolved with a "
                    "30 Hz Ricker wavelet and seeded noise"
                ),
                "sample_interval_ms": 2.0,
                "angles_deg": [float(angle) for angle in gather["incident_angle_deg"]],
                "array_shape": list(gather["amplitude"].shape),
                "amplitude_note": "dimensionless synthetic amplitudes; scenario data only",
            },
            "files": [
                "well_logs_synthetic.csv",
                "well_logs_synthetic.parquet",
                "checkshot_synthetic.csv",
                "checkshot_synthetic.parquet",
                "angle_gather_synthetic.npz",
            ],
        }
        manifest_path = temp_dir / "synthetic_manifest.json"
        manifest_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        os.chmod(manifest_path, 0o600)
        os.chmod(temp_dir, 0o700)
        os.replace(temp_dir, output_dir)
        return metadata
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"New output directory (default: {DEFAULT_OUTPUT}).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metadata = generate_examples(args.output_dir)
    print(
        f"Generated {metadata['log_rows']} synthetic well-log rows, "
        f"{metadata['checkshot_tie_points']} checkshot points, and a synthetic angle gather "
        f"in {args.output_dir}. These are examples, not real Volve data."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
