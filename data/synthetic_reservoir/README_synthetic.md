# Synthetic Reservoir Conceptualization

> **Synthetic only.** Entirely synthetic conceptual scenario; not derived from, calibrated to, or representative of real Volve measurements. Do not use for operational or scientific decisions.

## Scenario

A small faulted clastic reservoir with shale seals, a gas cap, an oil leg, a brine leg, and a tight carbonate basement.

- Area: 4000 m × 4000 m; local coordinates only.
- Model grid: 21 × 21 lateral cells; 601 depth samples.
- Wells: 7; vertical synthetic trajectories.
- Faults: F1 (80.0 m throw), F2 (55.0 m throw).
- Formations: Upper_Shale, Top_Seal, Reservoir_Sand, Basal_Shale, Tight_Carbonate.
- Well-log rows: 21,000; checkshot points: 210.
- Pre-stack angle range: 5°–45° in 5 angle bins.

## Geological model

The model is a shallow anticline with two normal-fault offsets. Five formations are represented: upper shale, top seal, reservoir sand, basal shale, and tight carbonate. The sand contains gas, oil, and brine legs with synthetic contacts. Elastic properties are illustrative scenario values, not measurements.

## Outputs

- Aggregate CSV/Parquet logs with elastic, petrophysical, QC, and class labels.
- Seven per-well LAS 2.0 files and sparse per-well checkshot tie points.
- Gridded elastic/facies model, structural horizons, faults, and well locations.
- Combined angle-coded CMP SEG-Y plus a regular 3D SEG-Y volume per angle.
- `angle_gather_synthetic.npz` with near-well gathers for the dashboard.
- Interactive `structure_map.html` and `seismic_gather.html` visualizations.

## Seismic assumptions

The traces use Shuey three-term reflectivity and a 30 Hz zero-phase Ricker wavelet, with independent seeded Gaussian noise. SEG-Y uses IEEE float, 2.0 ms sampling, and local synthetic coordinates. In the combined file, angle × 1000 is stored in trace-header unassigned integer word 233–236; separate angle volumes use regular inline/crossline geometry. This is not field acquisition data.

## Intended workflow use

The package tests ingestion, QC, time-depth mapping, rock-physics sampling, AVO forward modeling, and future Monte Carlo classification. Configured uncertainties are starting assumptions only; this generator performs no inversion or posterior calibration.
