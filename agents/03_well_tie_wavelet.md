# 03 — Poseidon well tie and seismic calibration

## Goal

Determine whether the supplied Poseidon well time-depth information can be tied to a usable seismic volume. Calibrate only products whose geometry, time reference, and amplitude meaning are documented. Near/far CBVS files and an AGC full-stack MDIO volume are present, but their decodability and calibration suitability must be established in task 01.

## Inputs

- QCed Poseidon logs and well metadata from task 02.
- Verified time-depth/checkshot/VSP information from `data/poseidon/wells/`.
- Decoded and documented seismic products from task 01. Use only stack types actually available and characterized; do not assume mid-angle data.

## Tasks

1. Use the closest suitable verified checkshot data to build each well's MD-to-TWT relation for the seismic-to-well tie. Prefer checkshots acquired in the target well; otherwise compare target-well and checkshot-well locations in a verified common projected CRS and select the nearest eligible checkshot survey by horizontal distance. Record the source well/survey, coordinates, separation distance, units, datum, interpolation, depth coverage, and uncertainty. A nearby well's checkshot is a transferred time-depth constraint, not a measurement in the target well: assess geological/velocity and reference compatibility, document transfer assumptions and added uncertainty, and reject unsuitable transfers rather than using proximity alone. Sonic integration may be used only as a documented alternative, not silently substituted for checkshots. If no eligible checkshot or defensible alternative exists, stop the tie and record the missing requirement. Exclude held-out-well checkshots from calibration inputs.
2. Establish seismic and well coordinate/datum compatibility before extracting traces. Where geometry is unavailable or unverified, stop the tie and record the missing requirement.
3. Extract traces or a documented local trace bundle only from readable seismic products. Record the trace-selection rule and distinguish near/far angle products from full-stack products.
4. Before any near/far comparison, test pair eligibility: decode both products; verify matched trace/bin geometry, time sampling/reference, polarity, processing/mutes/gains, and a documented common amplitude scale; and record the actual offset/angle ranges and stack weighting. Mark the pair ineligible for amplitude differences or ratios if any material requirement is unknown or incompatible. Do not fix a mismatch with unconstrained per-stack scaling.
5. Estimate wavelets only for suitable, non-AGC or otherwise amplitude-appropriate products. Record phase, bandwidth, polarity, and limitations; do not assume an 8-bit CBVS or AGC full stack preserves relative amplitudes.
6. For an eligible pair, tie near and far separately using the same verified well/time alignment and trace-selection rule. Preserve both responses and their uncertainty; calculate paired differences/ratios only if the common scale supports them. Forward-model only stack responses whose angle ranges and conventions are documented. Do not fabricate a mid-stack response or treat the AGC full stack as an angle stack.
7. If multiple training wells support calibration, estimate shared calibration parameters on training wells only. Report individual tie diagnostics and avoid unrestricted per-well scaling/stretching.
8. Estimate residual/noise only when calibrated real and synthetic data are meaningfully comparable; otherwise report that noise cannot be separated from encoding/processing mismatch.

## Outputs

- `data/interim/poseidon_td/{well}.csv` with source and reference metadata.
- `data/processed/wavelets/{volume}.npy` and metadata, only for supported ties.
- `configs/calibration.yaml` with explicit unavailable fields and documented parameters.
- `outputs/well_ties/{well}.png` and `outputs/well_ties/summary.csv` for attempted/eligible ties.
- Gate G2 assessment in `outputs/decision_log.md`.

## Acceptance criteria

- Time-depth and seismic geometry/reference assumptions are documented and checked per well.
- Checkshot selection identifies the closest eligible survey using verified well locations and reports its source, distance, suitability checks, and any cross-well transfer uncertainty; rejected closer surveys have documented reasons.
- Any tie result includes visual and quantitative diagnostics, with its input product and limitations identified.
- If no valid tie is possible, G2 records a clear stop/scope reduction rather than presenting a synthetic tie as Poseidon calibration.

## Pitfalls

- Exclude held-out wells from wavelet, scale, phase, and noise fitting.
- Do not assume all time curves use milliseconds, all coordinates share a CRS, or all seismic products are prestack.
- AGC and 8-bit encoding can invalidate absolute-amplitude comparisons; flag this rather than compensating with unconstrained scaling.
- A good-looking tie obtained through excessive shifting, stretching, or scaling is not evidence of calibration.
