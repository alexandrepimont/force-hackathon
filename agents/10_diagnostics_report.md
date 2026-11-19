# 10 — Poseidon diagnostics, what-if analysis, and reporting

## Goal

Make Poseidon results traceable and interpretable without overstating evidence. Run reproducible scenario analyses on request and report which conclusions are measurements, interpretations, or hypothetical simulations.

## Inputs

- Outputs and gate decisions from tasks 01–09, `outputs/decision_log.md`, and the exact configuration/model versions used.
- Poseidon data provenance and license/attribution information from task 01.

## Tasks

1. **What-if analysis:** Accept parameter changes only for defined model inputs, such as a documented porosity range, interval thickness, or hypothetical fluid scenario. Copy the configuration, run the applicable physics/forward/inference steps, and show before/after inputs, outputs, and assumptions. Do not answer a simulation question without running it.
2. **Ambiguity diagnostics:** Identify scenario pairs with similar responses but different labels, using only supported channels. Attribute ambiguity to tested factors such as Vs, porosity, wavelet, angle range, or thickness; do not invent causes.
3. **Value of information (optional):** Compare candidate observations (for example, verified time-depth data, measured Vs, improved stack metadata, or independent labels) with a documented sensitivity experiment. State its model dependence.
4. **Report:** Write `outputs/report.md` with the question, Poseidon files and provenance, inventory/gates, methods, eligible/held-out well counts, calibration/validation results, uncertainty, limitations, and next steps. For near/far analysis, state pair eligibility and evidence for geometry, angle ranges, processing, and common amplitude scale; distinguish matched-event observations from amplitude differences/ratios and scenario predictions. Report paired and single-channel results on the same wells where possible, or explicitly state why no comparison was possible. Separate Poseidon measurements from assumed and synthetic scenarios.
5. **Demo (optional):** Show the inventory/readability gate, a supported well-QC or rock-physics result, validation only if available, and one reproducible what-if scenario. If seismic mapping is unsupported, explain why rather than showing an unrelated synthetic map.

## Outputs

- `src/agent_diagnostics.py` with reproducible scenario execution, if an interactive runner is implemented.
- `outputs/report.md`.
- `outputs/demo/` where appropriate.

## Acceptance criteria

- Every quantitative report claim links to a generated output or a cited Poseidon source file.
- The report distinguishes measurements, interpretations, model-conditional probabilities, and hypothetical results.
- Limitations include the number of eligible/held-out wells, measured versus estimated curves, unknown units/CRS/angles where applicable, calibration status, and prior assumptions.

## Pitfalls

- Do not let a language model interpret beyond the data and executed simulations.
- Do not call model-conditional probabilities calibrated field probabilities without adequate held-out validation.
- Never present the conceptual synthetic dataset as Poseidon field evidence.
