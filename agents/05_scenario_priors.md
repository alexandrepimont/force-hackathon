# 05 — Poseidon scenario priors

## Goal

Build transparent scenario-based and, only if justified by task 02, Poseidon-well-informed elastic sequence priors. Quantify how prior choice affects downstream inference without claiming unsupported field labels.

## Inputs

- QCed Poseidon logs, run metadata, and evidence-backed interpretations from task 02.
- Rock-physics models and scenario bounds from task 04.
- Eligible training/holdout well split established before sampling; exclude all held-out wells and their alternate runs from fitted priors.

## Prior B — explicit scenario prior

- Sample only parameters that are relevant to the defined Poseidon analysis and declare each range in `configs/prior_b.yaml`.
- Separate measured Poseidon distributions from analyst-selected hypothetical ranges. Do not assign fluid or lithology truth labels to unobserved scenarios without marking their origin.
- Convert to elastic properties only through documented models from task 04.

## Prior C — empirical Poseidon prior

- First assess the number, diversity, and curve quality of eligible Poseidon wells and defensible labels. If evidence or sample size is inadequate, mark prior C unavailable rather than training on planted synthetic labels.
- If justified, sample contiguous Poseidon log intervals from training wells only, preserving correlations and run provenance. Record how intervals and labels were selected.
- Use fluid substitution or alternate label assignments only as explicit hypothetical scenarios, never as measured Poseidon fluid observations.

## Tasks

1. Implement a shared sampling interface that returns elastic sequences, sample metadata, and evidence/source tags.
2. Generate comparable libraries only when both priors are supported; otherwise document why a comparison is not possible.
3. Compare generated properties and sequence statistics with eligible Poseidon training wells. Show measured and estimated Vs separately, and avoid extrapolation beyond observed coverage without labeling it.
4. Record seeds, source wells, selected intervals, exclusions, label basis, and any assumptions in metadata.

## Outputs

- `src/priors.py`.
- `configs/prior_b.yaml` and, when supported, `configs/prior_c.yaml`.
- `data/processed/priors/{B,C}/` with sequences and provenance metadata.
- `outputs/priors/` comparison plots and a feasibility note.

## Acceptance criteria

- All generated values pass physical/range checks and include origin tags; missing and estimated inputs are not silently imputed.
- Any learned prior excludes held-out wells and all their source variants.
- A prior comparison is made only when comparable labels and data support it; otherwise the limitation is explicit.

## Pitfalls

- Do not use the conceptual synthetic reservoir as a Poseidon training population or evidence.
- Do not pool unrelated external wells into prior C without explicit authorization, provenance, geological justification, and a revised scope.
- A small set of wells does not support complex generative models; prefer an honest empirical/scenario analysis.
