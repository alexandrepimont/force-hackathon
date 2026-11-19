# 04 — Poseidon rock physics and fluid-sensitivity scenarios

## Goal

Evaluate how uncertainty in elastic properties and hypothetical pore fluids affects Poseidon well-based seismic responses. Establish whether fluid is a defensible prediction target (gate G3); do not import fluid-property values or field labels from unrelated reservoirs.

## Inputs

- QCed Poseidon logs and interpreted intervals from task 02, with measured/estimated Vs provenance.
- Only Poseidon-specific pressure, temperature, fluid, mineral, and test information verified by task 01. Record unavailable parameters explicitly.
- A documented calibration from task 03 is required for comparisons against field seismic.

## Tasks

1. Summarize which required elastic, mineral, pressure, temperature, salinity, fluid, and saturation inputs are measured, sourced, assumed, or absent.
2. Implement and test Gassmann substitution (including inversion to the dry frame) only where assumptions and input curves are adequate. Use named scenarios and uncertainty ranges for missing parameters; label them hypothetical.
3. For each eligible Poseidon interval, compare plausible fluid scenarios in Vp, Vs, density, impedance, and reflectivity. Do not call scenarios observed field fluids unless independent Poseidon evidence supports the labels.
4. Where task 03 produced a valid calibration, pass scenarios through the same documented forward model and compare differences with calibration uncertainty/noise and petrophysical variability. Otherwise keep results in elastic/reflectivity space and state that seismic detectability is untested.
5. Record assumptions, sensitivities, and the G3 verdict. If fluid input/label evidence is insufficient or responses are non-unique, restrict downstream targets to supported elastic/lithology properties.
6. Document any Vs relation, calibration population, and uncertainty when measured Vs is unavailable.

## Outputs

- `src/rock_physics.py` for tested physics functions.
- `configs/rock_physics.yaml` with sourced values and clearly marked scenario assumptions.
- `outputs/fluid_sensitivity/` with input/provenance tables, scenario comparisons, and a qualified verdict.
- G3 assessment in `outputs/decision_log.md`.

## Acceptance criteria

- A numerical Gassmann round-trip test passes for valid input ranges.
- Results distinguish Poseidon measurements from assumed scenario parameters and state whether a seismic comparison was possible.
- The G3 decision is supported by uncertainty comparisons, or explicitly records why evidence is inadequate.

## Pitfalls

- Do not substitute generic or unrelated reservoir fluid numbers as Poseidon measurements.
- Gassmann assumptions may fail for poor-quality, shaly, tight, or otherwise unsuitable intervals; flag exclusions.
- Do not tune parameters to make fluid scenarios appear separable.
