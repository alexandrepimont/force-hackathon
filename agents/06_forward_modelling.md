# 06 — Poseidon AVO forward modelling

## Goal

Forward-model elastic sequences into synthetic responses for comparison with the verified Poseidon seismic products. This task supports physical sensitivity analysis and optional model training; it does not turn hypothetical samples into Poseidon observations.

## Inputs

- Supported sequences/scenarios from task 05 and elastic properties from task 04.
- Verified time sampling, wavelet/calibration, and stack-angle information from task 03.
- Only seismic products suitable for amplitude comparison. If only CBVS near/far or AGC full-stack data are readable, honor those exact limitations.

## Tasks

1. Convert depth sequences to time with documented velocity and sampling assumptions. Track uncertainty in the depth-time conversion.
2. Compute angle-dependent reflectivity using Aki–Richards within its valid range and check selected cases with Zoeppritz where practical. Integrate over documented angle ranges only; if ranges are unknown, do not claim stack-specific predictions.
3. Convolve with supported wavelets. Apply scale/noise perturbations only when estimated from valid Poseidon calibration; otherwise use clearly labeled scenario bounds.
4. Represent only verified data channels. Near/far may be modelled separately if their metadata support it; integrate over each documented range and weighting. For a pair deemed comparable in task 03, retain stack-specific predictions and compare the predicted far-minus-near response (or another justified paired attribute) across scenarios, with calibration and measurement uncertainty. Do not infer the actual stack angles from the filenames, fabricate a mid stack, or use an AGC full stack as calibrated prestack amplitude.
5. Preserve thin-bed/tuning effects where the sampling and model resolve them. Store source, scenario, angles, wavelet, assumptions, and labels with each sample.
6. Validate implementation with analytic/unit tests and any supported well synthetics. Do not call agreement with an uncalibrated field trace a validation result.

## Outputs

- `src/avo_forward.py`.
- `data/processed/library/` with scenario/prior-specific arrays and metadata, if needed by task 07.
- `outputs/forward/` example responses, sensitivity plots, and verification notes.

## Acceptance criteria

- Reflectivity and convolution tests pass, units and conventions are documented, and all unknown angle/calibration fields remain explicit.
- Generated samples are traceable to Poseidon log statistics or marked hypothetical; they are not represented as measured seismic.
- Model size is chosen based on feasibility and validation, not an arbitrary sample-count target.

## Pitfalls

- Do not introduce perturbation ranges solely to improve classifier scores.
- A synthetic-to-real match is not meaningful when input processing, geometry, angle, or amplitude scales differ.
- Avoid extrapolating beyond measured Poseidon properties without marking the scenario as unsupported by observations.
