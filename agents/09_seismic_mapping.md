# 09 — Poseidon seismic application and mapping

## Goal

Apply a validated method to a Poseidon seismic volume only if the volume is decoded, its geometry and reference are verified, and the method passed task 08. Mapping is optional; do not turn an AGC full-stack or undecoded CBVS product into unsupported probability maps.

## Inputs

- Method and limitations from task 08, with baseline A included where comparable.
- A decoded Poseidon volume with documented geometry, coordinates, time/depth reference, and usable processing/amplitude characteristics.
- Verified well locations and any supplied tops/markers from tasks 01–02.

## Tasks

1. Verify spatial grid, coordinate system, units, sample axis, and well positioning. Stop if reference compatibility cannot be established.
2. Define a volume/window using supplied Poseidon markers or an explicitly documented interval; do not invent named horizons or fluid contacts.
3. Apply only to features/channels compatible with training and validation. Track missing data and out-of-distribution support; do not extrapolate confidently beyond the calibrated domain.
4. Produce supported elastic/lithology summaries and uncertainty. Produce fluid maps only if G3 passed and independent evidence supports the fluid classes.
5. Overlay eligible wells and supplied structural information only after coordinate compatibility is verified. Mark validation wells distinctly from other wells.

## Outputs

- `data/processed/maps/` in a format that preserves verified grid metadata.
- `outputs/maps/` figures and captions, or a short explanation if mapping is not supported.

## Acceptance criteria

- Every map has verified geometry/reference and a reproducible method/model version.
- Captions state the evidence, uncertainty, limitations, and whether values are predictions or scenario outputs.
- No synthetic-reservoir maps or unvalidated predictions are presented as Poseidon field results.

## Pitfalls

- A visually plausible map is not validation; use held-out wells for evidence.
- Do not assume CBVS/MDIO decoding, coordinate references, or amplitude preservation.
- Show unsupported/out-of-distribution areas as missing or low support, not as confident values.
