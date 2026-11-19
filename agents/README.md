# Agent task board

These files split the project in [constitution.md](../constitution.md) into tasks for separate agents or team members. Each task file defines its goal, inputs, outputs, acceptance criteria, and pitfalls.

## Project question

> Can well-anchored rock-physics simulations predict Hugin reservoir lithology and fluid from the Volve near, mid, and far angle stacks? Does a geology prior learned from many wells beat both hand-picked Monte Carlo ranges and a classical intercept–gradient interpretation at blind wells?

Key decisions that refine the constitution:

1. **Use real data throughout.** Seismic calibration is required, not optional. A purely synthetic study is not an acceptable final result.
2. **Use blind wells as ground truth.** Every prediction method is scored at wells held out from training and calibration.
3. **Always compare against baselines.** The methods are:
   - **A:** classical intercept–gradient classification.
   - **B:** a model trained on a hand-picked Monte Carlo prior, as described in the constitution.
   - **C:** a model trained on a geology prior learned from the FORCE wells.
4. **Test fluid sensitivity first.** Volve Hugin oil is relatively heavy, with low GOR, in deep sandstone, so the fluid effect may be weak. Gassmann substitution decides whether fluid prediction is a realistic goal; see [04](04_rock_physics_fluid_substitution.md).
5. **Make uncertainty testable.** Uncertainty is useful only if it is checked against blind wells, for example by testing whether confident predictions are more often correct.
6. **Give agents concrete tools.** Agents run QC, simulations, and validation. They must not replace physics or claim certainty that the data cannot support.

## Agents and dependencies

| # | Task | Depends on |
|---|------|------------|
| 01 | [Data inventory and verification](01_data_acquisition.md). The team downloads Volve data manually. | — |
| 02 | [Well QC and petrophysics](02_well_qc_petrophysics.md) | 01 |
| 03 | [Well tie, wavelet, and stack calibration](03_well_tie_wavelet.md) | 01, 02 |
| 04 | [Rock physics and fluid substitution](04_rock_physics_fluid_substitution.md) | 02 |
| 05 | [Scenario priors: hand-picked and learned](05_scenario_priors.md) | 02, 04 |
| 06 | [AVO forward modelling and synthetic library](06_forward_modelling.md) | 03, 04, 05 |
| 07 | [Inference and classification](07_inference_classification.md) | 03, 06 |
| 08 | [Blind-well validation](08_blind_well_validation.md) | 07 |
| 09 | [Seismic application and mapping](09_seismic_mapping.md) | 07, 08 |
| 10 | [Diagnostics, what-if analysis, and report](10_diagnostics_report.md) | 08, 09 |

Tasks 02 and 03 can start in parallel once the team has downloaded the first well and seismic files and 01 has verified them. Task 05 can start with the FORCE wells before the Volve data arrives.

## Decision gates

| Gate | When | Question | If no |
|------|------|----------|-------|
| G1 | About 4 h | Do we have the near, mid, and far stacks, plus at least 3 Hugin wells with Vp, density, and Vs or a defensible Vs estimate, trajectories, and time–depth data? | Reduce scope to fewer wells. Flag any estimated Vs as a key weakness. |
| G2 | After the first tie | Does the synthetic match the real seismic at the tie well in timing, frequency, and character? | Fix the tie, wavelet, or scaling before training models. |
| G3 | After Gassmann | Is the brine-to-oil change in synthetic near, mid, and far amplitudes large compared with noise and other scenario variability? | Drop fluid as a target. Predict lithology or reservoir quality only, and report fluid non-uniqueness as a result. |
| G4 | After blind wells | Does B or C beat baseline A? | Report this honestly. A negative result with evidence is still a valid deliverable. |

Record the outcome of every gate in `outputs/decision_log.md`.

## Shared conventions

- **Language:** Python 3. Prefer `numpy`, `scipy`, `pandas`, `lasio`, `segyio`, `matplotlib`, and `scikit-learn`. Use `lightgbm` or `torch` only where needed.
- **Layout:**
  - `data/raw/`: original files downloaded by the team. Agents treat them as read-only and do not commit them.
  - `data/interim/`: cleaned or derived data.
  - `data/processed/`: model-ready tables and libraries.
  - `src/`: code. `configs/`: YAML configs. `outputs/`: figures, tables, and reports.
- **Large data:** `.gitignore` excludes `data/`, `*.segy`, and `*.sgy`. Keep all data under `data/` so it is never committed.
- **Well IDs:** normalize names to the form `15_9-19A` or `15_9-F-1C`. Keep one mapping table in `data/interim/well_index.csv`.
- **Units:** MD and TVDSS in metres; TWT in milliseconds; Vp and Vs in m/s; density in g/cm³; porosity, Vsh, and Sw as fractions.
- **Log tables:** Parquet files with columns `well, md, tvdss, twt, vp, vs, rho, gr, nphi, rt, cali, phi, vsh, sw, litho, fluid, flag_bad`, and `vs_source` (`measured` or `estimated`).
- **Configuration:** put all parameters in `configs/*.yaml`. Do not hard-code assumptions.
- **Reproducibility:** fix random seeds and write a short `*_meta.json` next to each generated dataset.
- **Data leakage:** blind wells must not be used to fit wavelets, scaling, priors, or models.
- **Attribution:**
  - Volve data © Equinor ASA and the Volve licence partners. Check and follow the licence terms on the [Volve data page](https://www.equinor.com/energy/volve-data-sharing).
  - RealPore data © GeoProvider, https://geoprovider.no/, CC BY 4.0.
  - FORCE 2020 data: [Zenodo record 4351156](https://zenodo.org/records/4351156).
