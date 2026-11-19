# Poseidon AVO agent task board

These files define a staged workflow for investigating the Poseidon well and seismic data already present in `data/poseidon/`. Each task lists its inputs, outputs, acceptance criteria, and limitations. The repository's broader motivation is in [main_direction.md](../main_direction.md).

## Project question

> What lithology and, only where supported, pore-fluid information can be inferred from the Poseidon well data and available seismic, and how robust are those inferences to log quality, time-depth uncertainty, rock physics, and seismic calibration?

The repository currently contains Poseidon well logs, markers/tops, selected time-depth/checkshot files, near and far CBVS stacks, and an AGC full-stack MDIO volume. Their metadata, units, coverage, geometry, decoding, and suitability must be verified before interpretation. Do not assume a mid stack, calibrated angle gathers, common CRS, complete well ties, or known fluid labels.

1. **Inventory first.** Determine what the local files actually contain before committing to an AVO scope. The conceptual package under `data/synthetic_reservoir/` is synthetic test data, not Poseidon evidence.
2. **Preserve provenance and evidence.** Keep original inputs read-only; record source file, curve mnemonic, units, depth/time reference, processing, and label confidence.
3. **Use held-out wells only when justified.** A blind-well score is meaningful only when independent wells have usable seismic ties and defensible labels. Report the sample count and spatial limitations.
4. **Compare baselines fairly.** Where data supports them, compare a conventional AVO/elastic-attribute baseline with scenario-based and Poseidon-well-informed models using the same inputs and holdout rules.
5. **Test identifiability before predicting fluid.** Run documented fluid-substitution scenarios and compare their response differences with measured noise and uncertainty. If field fluid properties or labels are absent, do not claim measured Poseidon fluid predictions.
6. **Keep uncertainty explicit.** Report confidence, missingness, estimated curves, and out-of-distribution limitations; do not make unsupported geological claims.

## Agents and dependencies
## Near/far stack comparison

Use the supplied near and far stacks as a potential paired observation of offset/angle-dependent response, not as calibrated AVO channels by default. First verify that both files decode and share trace/bin geometry, sampling and time reference, polarity, processing, and a documented common amplitude scale; establish their actual offset/angle ranges and stack weighting from metadata. The `8bit` filenames alone do not establish amplitude comparability. If any required check fails, record the pair as ineligible for amplitude differences or ratios and do not compensate with unconstrained scaling.

For eligible pairs, compare matched events and traces, preserve near and far as separate channels, and evaluate their difference or ratio only where the common scale supports it. Carry the pair through tie/calibration (03), stack-specific forward modelling (06), paired-versus-single-channel inference (07), and validation (08). Compare predicted near-to-far response changes across scenarios against measurement and calibration uncertainty. Report this as a limited two-range sensitivity test, not a fully sampled AVO curve; no mid stack or exact angle may be inferred. Task 10 should report pair eligibility, the comparisons actually run, and the limitations. If the stacks are not decodable or comparable, retain inventory/readability findings and proceed only with independently supported work.

## Agents and dependencies

| # | Task | Depends on |
|---|------|------------|
| 01 | [Poseidon data inventory and verification](01_data_acquisition.md) | — |
| 02 | [Poseidon well QC and petrophysics](02_well_qc_petrophysics.md) | 01 |
| 03 | [Well tie, wavelet, and seismic calibration](03_well_tie_wavelet.md) | 01, 02 |
| 04 | [Rock physics and fluid-sensitivity scenarios](04_rock_physics_fluid_substitution.md) | 02 |
| 05 | [Poseidon scenario priors](05_scenario_priors.md) | 02, 04 |
| 06 | [AVO forward modelling](06_forward_modelling.md) | 03, 04, 05 |
| 07 | [Inference and classification](07_inference_classification.md) | 03, 06 |
| 08 | [Held-out-well validation](08_blind_well_validation.md) | 07 |
| 09 | [Seismic application and mapping](09_seismic_mapping.md) | 07, 08 |
| 10 | [Diagnostics, what-if analysis, and reporting](10_diagnostics_report.md) | 08, 09 |

Tasks 02 and 03 may proceed in parallel after task 01 verifies their respective inputs. Later tasks must narrow or stop when a required Poseidon data prerequisite is unavailable. Use synthetic data only for software checks and clearly labeled hypothetical scenarios.

## Decision gates

| Gate | Question | If no |
|------|----------|-------|
| G1 — inventory | Are Poseidon well curves, labels/tops, time-depth information, and seismic files sufficiently documented and readable for a defined analysis? | Report the verified inventory and missing prerequisites; limit work to QC or well-log analysis. |
| G2 — tie/calibration | Can any usable Poseidon seismic volume be matched to eligible wells without unsupported geometry, datum, or amplitude assumptions? | Do not claim seismic-calibrated predictions; provide only well-log or explicitly synthetic results. |
| G3 — fluid sensitivity | Are fluid scenarios based on documented inputs distinguishable beyond uncertainty/noise, and are any comparison labels defensible? | Exclude fluid as a field prediction target; report the non-identifiability or missing evidence. |
| G4 — validation | Are enough independent, labeled Poseidon wells available for a leakage-controlled evaluation? | Describe the limitation and avoid performance claims; keep results exploratory. |

Record each gate and its evidence in `outputs/decision_log.md`.

## Shared conventions

- **Language and tools:** Python 3; prefer existing project dependencies and `numpy`, `scipy`, `pandas`, `lasio`, `segyio`, `matplotlib`, and `scikit-learn` where appropriate. Confirm native CBVS/MDIO compatibility before selecting a reader.
- **Inputs:** the Poseidon source tree is `data/poseidon/`. Treat original files as read-only and do not commit or duplicate large source data. Keep synthetic-reservoir data clearly segregated.
- **Derived layout:** use `data/interim/` for normalized/QC data, `data/processed/` for model-ready products, `src/` for code, `configs/` for assumptions, and `outputs/` for reports and figures.
- **Well IDs:** retain source well names (for example `Poseidon-1`, `Kronos-1`, and `Torosa-3`) and maintain a mapping from every derived row to its original file. Do not merge duplicate/alternate log runs without a documented rule.
- **Units and references:** read units, depth datum, time convention, polarity, sample interval, coordinates, and CRS from source metadata; record unknowns as unknown rather than assuming a standard.
- **Log tables:** preserve original mnemonics and values alongside normalized columns. Add a normalized field only when its source, units, conversion, and QC are documented; mark absent/estimated values explicitly.
- **Configuration and reproducibility:** put adjustable parameters in `configs/*.yaml`; fix random seeds and write metadata with every generated dataset.
- **Data leakage:** held-out wells and all alternate runs for those wells must be excluded from fitted wavelets, scaling, priors, thresholds, and models.
- **Attribution:** record the supplied Poseidon dataset's source, license/terms, and any accompanying documentation in the inventory. Do not infer permission or provenance from filenames alone.
