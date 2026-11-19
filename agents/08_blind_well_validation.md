# 08 — Blind-well validation

## Goal

Provide the main evidence for the project: how methods A, B, and C perform on **real seismic at wells they have not seen**.

## Inputs

- Methods A, B, and C from task 07.
- Labelled logs from task 02, upscaled to seismic resolution.
- Seismic traces at the well locations.

## Tasks

1. Run leave-one-well-out validation. For each blind well, rerun the wavelet and scaling from task 03, prior C from task 05, and training from task 07 **without that well**. If time is short, define a fixed held-out set at the start and document it.
2. Upscale the log labels to seismic resolution, for example with Backus averaging and majority labels over the wavelet length.
3. Score each method per well and per class: balanced accuracy, F1, and log loss. If fluid is a target, report lithology and fluid separately.
4. **Test whether uncertainty is meaningful:**
   - Plot reliability diagrams.
   - Plot accuracy against confidence bins.
   - Check whether high-entropy predictions are more often wrong.
5. Create one comparison figure per blind well: logs and labels, real near/mid/far traces, synthetic traces, and A/B/C predictions side by side.

## Outputs

- `outputs/validation/scores.csv`.
- `outputs/validation/{well}.png`.
- `outputs/validation/reliability.png`.
- Record the G4 outcome in `outputs/decision_log.md`.

## Acceptance criteria

- No leakage. State explicitly which wells were used for wavelets, scaling, priors, and training in every run.
- The final table compares A, B, and C directly.

## Pitfalls

- A few blind wells give an indication, not statistical proof. State the number of wells in every claim.
- Wells in the same fault block are not independent. Note spatial clustering.
- Fluids may have changed between logging and seismic acquisition. Note mismatches instead of hiding them.
