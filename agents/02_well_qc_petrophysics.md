# 02 — Well QC and petrophysics

## Goal

Turn raw well logs into clean, labelled elastic logs. These labels are the ground truth for training and for blind-well validation.

## Inputs

- Volve wells from 01.
- FORCE 2020 wells, which have lithology labels, for prior learning in task 05.
- NPD formation tops.
- RealPore core porosity, permeability, and lithology descriptions, where depth matching is reliable.

## Tasks

1. Load the logs. Standardize mnemonics, for example `DTC/DT/AC → vp` and `DTS/DTSM → vs`, convert units, and apply depth shifts if needed.
2. Compute TVDSS from the deviation surveys.
3. Flag bad hole using `CALI` versus bit size and `DRHO`. Store the result in `flag_bad`. Do not silently delete flagged samples.
4. Compute Vsh from GR and total or effective porosity from density and neutron. Compute Sw from resistivity using Archie or another documented model. Put all parameters in `configs/petrophysics.yaml`.
5. Assign labels:
   - `litho`: `shale`, `sand`, `tight` (cemented or carbonate-cemented sand), or another class if well supported.
   - `fluid` in sands: `brine` or `oil`, using Sw, known contacts, and DST and Factpage information.
6. If Vs is missing, estimate it with a documented relation, such as Greenberg–Castagna, calibrated on wells with measured Vs. Set `vs_source = estimated` and record its uncertainty.
7. Compare petrophysical porosity with RealPore core porosity where wells overlap. Report the bias.

## Outputs

- `data/interim/logs/{well}.parquet`, following the shared column contract.
- `outputs/well_qc/{well}.png`: tracks for GR, Vp, Vs, density, porosity, Sw, labels, and bad-hole flags.
- `outputs/well_qc/summary.csv`: Hugin thickness, net sand, average porosity, and fluid per well.

## Acceptance criteria

- Labels in Hugin are consistent with the published facts, for example oil in Hugin in 15/9-19 SR and 15/9-19 A.
- The `vs_source` flag is present for every sample.

## Pitfalls

- Oil-based mud invasion affects the shallow resistivity. Use deep resistivity for Sw.
- Labels are interpretations, not absolute truth. Note the confidence of each label.
- A RealPore core plug samples a much smaller volume than a log. Use the core data for checks, not as one-to-one labels.
