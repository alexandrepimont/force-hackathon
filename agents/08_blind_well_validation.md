# 08 — Held-out Poseidon well validation

## Goal

Evaluate supported methods on Poseidon wells excluded from model development, only where well labels, seismic coverage, and time-depth alignment are independently adequate.

## Inputs

- Eligible methods from task 07.
- QCed Poseidon logs and evidence-backed labels from task 02.
- Usable, tied seismic at candidate wells from task 03.
- A pre-registered well/run split and its spatial/temporal metadata.

## Tasks

1. Decide and record the held-out wells before fitting models, priors, calibration, thresholds, or selecting features. Exclude all alternate runs and duplicate observations of each held-out well from training.
2. Confirm each validation well has adequate labels, curve quality, time-depth relation, and seismic geometry. Exclude ineligible wells with a stated reason; do not lower quality criteria to increase sample count.
3. Compare labels at seismic resolution using a documented upscaling/window rule that respects the actual wavelet and time sampling. Preserve ambiguous intervals and label confidence.
4. Report per-well and per-class metrics, uncertainty/calibration diagnostics where sample counts support them, and baseline comparisons. Separate lithology/elastic results from fluid results. Where both stacks are individually usable and the pair is comparable, report paired-channel results alongside near-only and far-only ablations on the same held-out wells; state the paired-well count. Do not compare scores on different eligibility subsets as though they were a controlled channel comparison.
5. Produce per-well plots showing only verified channels, log labels, tie uncertainty, predictions, and exclusions. State the number of independent wells and spatial/temporal limitations.
6. Record G4 and the scope of any claims in `outputs/decision_log.md`.

## Outputs

- `outputs/validation/scores.csv` with per-well metrics and eligibility details.
- `outputs/validation/{well}.png` for each included well.
- `outputs/validation/reliability.png` only when enough independent observations exist to interpret it.
- G4 assessment in `outputs/decision_log.md`.

## Acceptance criteria

- A reproducible split excludes held-out wells/runs from every fitted component, and the exclusion list is published.
- Scores are traceable to Poseidon labels and verified seismic; unsupported or synthetic-only results are not mixed into field scores.
- Conclusions reflect the number and independence of wells, including a no-validation conclusion when prerequisites are unmet.

## Pitfalls

- A small or spatially clustered well set is exploratory, not statistical proof.
- Estimated Vs, uncertain labels, or poor time-depth ties must be visible in the evaluation.
- Do not infer changes between logging and seismic without documented acquisition dates and evidence.
