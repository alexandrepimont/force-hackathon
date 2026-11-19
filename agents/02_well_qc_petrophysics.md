# 02 — Poseidon well QC and petrophysics

## Goal

Normalize and quality-check the Poseidon well logs, retaining original data and source provenance. Produce interpreted petrophysical properties or labels only where the supplied measurements and documentation support them; do not treat interpretations as ground truth by default.

## Inputs

- Well-file inventory from task 01, including the Poseidon LAS logs, alternate/extra log runs, supplied time-depth files, tops/markers, and applicable well metadata.
- The well index, source documentation, and project configuration.

## Tasks

1. Read each LAS run independently. Map mnemonics to normalized names only with a documented mnemonic/unit mapping; keep the original mnemonic, raw values, and source file. Verify units and null sentinels before conversion.
2. Compare alternate depth/time runs for matching well names. Do not concatenate or shift runs unless the reference datum, depth basis, and reason for the transformation are verified and recorded.
3. Compute TVD or other depth coordinates only when a valid deviation survey and datum are available. Otherwise retain measured depth and mark derived coordinates unavailable.
4. Inspect curve distributions and coverage. Flag washout or other bad-hole intervals only when supporting curves and thresholds exist; retain samples and record which rules produced each flag.
5. Derive Vsh, porosity, or Sw only when required input curves and units are present. Put cutoffs, transforms, and model parameters in `configs/petrophysics.yaml`; report sensitivity and missing inputs. Do not infer fluid from resistivity alone.
6. Create lithology/fluid interpretations only from supplied markers, documented labels, core/test information, or explicitly stated petrophysical scenarios. Record evidence, confidence, and unknown states separately; do not convert unknown to a class.
7. If Vs is absent, any estimate must use a documented, applicable relation, calibration data, and uncertainty. Mark each sample as measured, estimated, or unavailable.

## Outputs

- `data/interim/poseidon_logs/{well}.parquet` with normalized measurements, original-mnemonic/source provenance, depth/time fields where verified, nulls, and QC flags.
- `outputs/well_qc/{well}.png` with available curves, depth coverage, nulls, and QC/interpretation annotations.
- `outputs/well_qc/summary.csv` with per-well curve/run coverage, QC counts, marker coverage, and any supported derived-property summaries.

## Acceptance criteria

- Every normalized curve can be traced to a source mnemonic, file, unit, and conversion (or explicitly marked unknown).
- Missing/estimated measurements, alternate runs, and uncertain labels remain distinguishable in tables and plots.
- No Poseidon field label or petrophysical property is asserted without documented evidence and a stated method.

## Pitfalls

- Never silently replace, resample, depth-shift, or delete source values.
- Log curves may have different sampling, units, or depth references even within one well.
- A top/marker is not automatically a lithology or fluid label; distinguish observed source metadata from interpretation.
- Keep the conceptual synthetic package out of Poseidon training and validation labels.
