# 07 — Poseidon inference and classification

## Goal

Estimate supported lithology/elastic classes from usable Poseidon seismic, with uncertainty and explicit data-quality limits. Predict pore fluid only if task 04 gate G3 and task 02 label evidence both permit it.

## Methods

| ID | Method | Training/evidence |
|----|--------|-------------------|
| A | Conventional AVO or elastic-attribute baseline using verified inputs | Poseidon training wells and documented seismic metadata |
| B | Scenario-prior model, if supported | Hypothetical prior B from task 05 |
| C | Poseidon-well-informed model, if sufficient eligible data exist | Training wells only, excluding held-out wells |

Compare B and C with identical features, model capacity, and tuning budget. If verified data does not support a method or channel, mark it unavailable rather than filling it with synthetic or AGC data without disclosure.

## Tasks

1. Define the feature set from products verified in task 03. Use actual channel/angle metadata; do not describe a full-stack volume as near/mid/far. If the near/far pair passed task 03 comparability checks, retain both channels and compare paired-channel results with matched near-only and far-only baselines on identical eligible cases. Use a difference/ratio feature only when common amplitude scaling is verified. If comparability failed, do not derive cross-stack amplitude features.
2. Implement baseline A only when its required seismic attributes/angles are identifiable. If only two documented angles are available, report the limitations of fitting intercept and gradient from two observations.
3. Implement B/C only when tasks 05–06 provide defensible, comparable training data. Use identical input representation, model settings, and tuning procedures.
4. Return probabilities only for defined classes; fluid outputs require G3 and independently supported Poseidon evidence. Include entropy or another uncertainty score and missing/out-of-distribution indicators.
5. Calibrate on data that is separate from the final held-out wells. Keep synthetic test metrics separate from any Poseidon field evaluation.
6. Document training data, excluded wells/runs, features, class definitions, and unsupported cases with each model.

## Outputs

- `src/litho_fluid_inference.py`.
- `data/processed/models/{A,B,C}/` for methods that can be supported.
- `outputs/inference/` with data eligibility, predictions, calibration, and clearly separated synthetic/unit-test results.

## Acceptance criteria

- Eligible methods share a documented input/output contract and can be compared on the same cases.
- Every reported probability corresponds to an explicit class/evidence basis; unavailable classes are not silently predicted.
- Synthetic performance is labeled as a software/scenario check, not Poseidon field accuracy.

## Pitfalls

- Do not assume a particular class distribution or reservoir geology before the Poseidon labels support it.
- Avoid probability calibration on final held-out wells.
- Unsupported stack decoding, AGC scaling, and missing angles can invalidate the features; surface these limitations rather than hiding them in preprocessing.
