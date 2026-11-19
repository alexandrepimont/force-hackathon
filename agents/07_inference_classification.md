# 07 — Inference and classification

## Goal

Predict lithology and fluid probabilities from observed near, mid, and far data using three comparable methods.

## Methods

| ID | Method | Trained on |
|----|--------|------------|
| A | Classical intercept–gradient (I–G) crossplot classification, with background trend and class boundaries | Training wells only |
| B | Classifier on synthetic windows | Library B: hand-picked prior |
| C | Same classifier, same settings | Library C: learned prior |

B and C must use **identical** model architecture, features, and training budget. Only the prior differs.

## Tasks

1. Implement A:
   - Fit I and G from near, mid, and far using the angle ranges.
   - Derive class boundaries from training wells and rock physics.
2. Implement the B/C classifier:
   - Start with gradient boosting on windowed features.
   - Optionally add a small 1D CNN that sees the full trace window, which is useful for tuning.
3. Output probabilities for `litho`. Output `fluid` only if gate G3 passed.
4. Calibrate probabilities on held-out **synthetic** data and check them later on training wells. Do not calibrate on blind wells.
5. Provide a uniform interface: `predict(near, mid, far, twt) -> DataFrame[p_shale, p_sand, p_tight, p_brine, p_oil, entropy]`.

## Outputs

- `src/litho_fluid_inference.py`.
- `data/processed/models/{A,B,C}/`.
- `outputs/inference/`: confusion matrices on synthetic test data and predictions at training wells.

## Acceptance criteria

- A, B, and C run on the same input and return the same output format.
- Performance on synthetic test data is reported, but **not** treated as the final result.

## Pitfalls

- High accuracy on synthetic data proves little. The real test is task 08.
- Classes are imbalanced: Hugin is mostly sand. Use per-class metrics.
