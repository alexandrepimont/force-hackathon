# Decision log (Poseidon)

## G2 - tie/calibration (2026-11-19T16:10:39+00:00)

**PASS** - mean correlation at training wells after the automated tie: near 0.67, mid 0.66, far 0.57 (threshold 0.5).

- Training wells: Kronos-1, Pharos-1, Poseidon-1, Poseidon-North-1, Proteus-1. Held out: Boreas-1, Poseidon-2.
- Seismic: near/mid/far 6-18/18-30/30-42 deg, reverse polarity (both from the CGGVeritas processing report). Polarity check (mean r): {'reverse': 0.634, 'normal': -0.634}.
- Time-depth: own VSP-calibrated curves for Kronos-1 and Pharos-1; other wells use the nearest eligible donor (see outputs/well_ties/summary.csv). Transferred curves carry extra uncertainty, absorbed by a bounded bulk shift and stretch.
- Limitations: TVDSS assumes vertical wells; common amplitude scale between stacks is undocumented (time variant scaling); wavelets are Ricker estimates; Vs is partly estimated where not measured.

| well | held out | TD source | shift ms | stretch % | r before | r after |
|---|---|---|---|---|---|---|
| Proteus-1 | False | Pharos-1 (transferred) | -21 | -0.5 | -0.21 | 0.80 |
| Kronos-1 | False | Kronos-1 | +11 | +0.0 | 0.18 | 0.82 |
| Boreas-1 | True | Pharos-1 (transferred) | +21 | -0.5 | -0.57 | 0.81 |
| Poseidon-1 | False | Pharos-1 (transferred) | +17 | -2.0 | -0.43 | 0.74 |
| Pharos-1 | False | Pharos-1 | +1 | +0.5 | 0.45 | 0.48 |
| Poseidon-2 | True | Kronos-1 (transferred) | -17 | +0.5 | -0.26 | 0.64 |
| Poseidon-North-1 | False | Pharos-1 (transferred) | +53 | +0.0 | -0.05 | 0.32 |
| Torosa-1 | - | Torosa-3 | - | - | - | stopped: well outside the seismic sub-volume |
