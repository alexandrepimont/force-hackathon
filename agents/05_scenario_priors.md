# 05 — Scenario priors: hand-picked (B) and learned (C)

## Goal

Generate realistic 1D elastic layer sequences with known lithology and fluid labels. Build two priors so we can test whether **geology learned from many wells** beats hand-picked parameter ranges. This comparison is the main scientific question.

## Inputs

- FORCE 2020 wells (118 LAS files; check Vs coverage).
- NPD formation tops for selecting relevant intervals.
- Volve training wells from 02. **Exclude blind wells.**
- Rock physics and Gassmann from 04.

## Prior B — hand-picked Monte Carlo, from the constitution

- Sample porosity, Vsh, layer thickness, fluid state, and Sw from explicit ranges in `configs/prior_b.yaml`.
- Convert samples to Vp, Vs, and density with the rock-physics model from 04.
- Build simple layered sequences, for example caprock above a reservoir or interbedded sand and shale.

## Prior C — learned from wells

- **Start simple:** block-resample real log segments of Vp, Vs, density, and labels from FORCE and training wells in Jurassic and Hugin-equivalent intervals. Use formation tops to choose intervals.
- Use Gassmann to create brine and oil versions of each sand segment, which yields fluid labels the wells alone do not cover.
- Condition on depth and burial trend so segments are physically plausible for about 2700–3100 m.
- **Optional, if time allows:** train a small generative model, such as a VAE or Markov chain over facies plus elastic properties, and compare it with block resampling.

## Tasks

1. Implement both generators behind the same interface: `sample(n, seed) -> list of (vp, vs, rho, litho, fluid, dz)`.
2. Generate equal-size libraries for B and C.
3. Compare B, C, and the real training wells with Vp/Vs-versus-AI crossplots, thickness distributions, and transition statistics.

## Outputs

- `src/priors.py`.
- `configs/prior_b.yaml` and `configs/prior_c.yaml`.
- `data/processed/priors/{B,C}/`: sequences and metadata.
- `outputs/priors/`: comparison plots against real wells.

## Acceptance criteria

- Both priors produce physically valid values: Vp > Vs, plausible densities, and no NaNs.
- Plots show how B and C differ from real geology.

## Pitfalls

- **Leakage:** Blind wells must not contribute segments to prior C.
- Many FORCE wells lie far from Volve and represent other geology. Use filtering and report which wells were used.
- Estimated Vs in FORCE wells can bias prior C. Keep and report `vs_source`.
