# force-hackathon

## Dashboard

Run the local explorer on the real Poseidon wells:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\download_poseidon_wells.py
.\.venv\Scripts\python.exe scripts\prepare_poseidon_dashboard.py
.\.venv\Scripts\streamlit.exe run dashboard\app.py
```

`prepare_poseidon_dashboard.py` combines the 8 wells into `data\poseidon\dashboard\` (log table with
standard curve names, Vp, Vs, acoustic impedance, Vp/Vs, TVDSS and formation per sample, plus tops, well
locations and a provenance manifest). If the near/mid/far SEG-Y stacks are downloaded, it also builds a fast
display cache in `data\poseidon\seismic\cache\` (~3 GB per stack, about 1 minute each). The dashboard opens
the Poseidon folder by default and shows log tracks with formation tops, a well map with a tops correlation,
AI vs Vp/Vs crossplots, and a **Seismic** tab with inlines, crosslines and time slices of the near, mid and far
stacks (side by side on one colour scale), wells overlaid, and near/mid/far traces at the selected well.

To view the synthetic test data instead, enter `data\synthetic_reservoir` as the data folder in the sidebar.
The synthetic data is for software testing and is not representative of field measurements.

## Poseidon well data

Download the Poseidon 3D well data into `data\poseidon\wells\` (git-ignored):

```powershell
.\.venv\Scripts\python.exe scripts\download_poseidon_wells.py          # logs, headers, tops
.\.venv\Scripts\python.exe scripts\download_poseidon_wells.py --drive  # + checkshot reports, surveys, well reports
```

| Folder | Content |
|---|---|
| `logs\<well>\` | Spliced composite LAS for Boreas-1, Kronos-1, Pharos-1, Poseidon-1, Poseidon-2, Poseidon-North-1, Proteus-1, Torosa-1 |
| `logs_extra\<well>\` | VSP-calibrated depth logs (Kronos-1, Pharos-1, Proteus-1 ST2), Poseidon-1/-2 CSV logs, Torosa-3 logs, deviation, VSP and markers |
| `ga\` | Geoscience Australia borehole headers and lithostratigraphy intervals |
| `tops\well_tops.csv` | Combined formation tops (MD below rig datum) with source per pick |

All data is Geoscience Australia open-file data (CC BY 3.0 AU; authoritative source:
[NOPIMS](https://public.neats.nopta.gov.au/nopims)). The script pulls it from public GitHub mirrors and the
GA Boreholes WFS; `--drive` adds files from ConocoPhillips' Google Drive share, which is often blocked by
Google's "Quota exceeded" limit. The script is resumable, so rerun it to fill gaps.

## Poseidon seismic (near, mid, far stacks)

The full angle stacks are ~104 GB each, so this downloads a sub-volume from the public TGS MDIO copy on AWS
(`s3://tgs-opendata-poseidon`, CC BY 4.0) and writes SEG-Y to `data\poseidon\seismic\` (git-ignored):

```powershell
.\.venv\Scripts\python.exe scripts\download_poseidon_seismic.py
```

Default window: inlines 2310–3625, crosslines 1292–2769 (the 7 Poseidon-area wells plus ~100 traces) and
2560–4092 ms TWT (top Plover reservoir is ~3.1 s), about 3 GB per stack. Torosa-1 lies outside it; use
`--il`, `--xl`, `--t` and `--name` for other windows. Files use inline byte 189, crossline byte 193 and CDP X/Y
in GDA94 / MGA zone 51, so they load with `avo.segy_io.read_cube`.

## Well QC and automated well tie (agents 02 and 03)

```powershell
.\.venv\Scripts\python.exe scripts\run_well_qc.py     # task 02 -> data\interim\poseidon_logs\, outputs\well_qc\
.\.venv\Scripts\python.exe scripts\run_well_tie.py    # task 03 -> data\interim\poseidon_td\, data\processed\wavelets\,
                                                      #            configs\calibration.yaml, outputs\well_ties\, outputs\decision_log.md
```

- **QC** (`configs/petrophysics.yaml`): raw curves and provenance kept; bad-hole flags with the triggering rule;
  Vsh, porosity, Archie Sw and a lithotype scenario (shale, tight, low/high-porosity sand, volcanic, unknown).
  Fluid is not labelled from logs; only a gas indicator (low Sw + neutron-density crossover) is stored. TVDSS
  assumes vertical wells (no deviation surveys); the error estimate is in the summary.
- **Tie** (`configs/well_tie.yaml`, seismic facts in `configs/poseidon_seismic.yaml`): each well uses its own
  VSP-calibrated time-depth (Kronos-1, Pharos-1) or the nearest eligible well's; a bounded bulk shift and stretch is
  found automatically; Ricker wavelets and one global scale per stack come from training wells only (Poseidon-2
  and Boreas-1 held out). Polarity (reverse) and angle ranges (6-18, 18-30, 30-42 deg) are from the CGGVeritas
  processing report.
- **Dashboard:** step-by-step pages (1 Data, 2 Well logs, 3 Well QC, 4 Seismic, 5 Well tie, 6 Seismic through
  wells, 7 Rock physics, 8 AVO classification, 9 Prospect simulation) with back/next links. Step 5 shows per well
  what the automated tie changed and a line through the well with the tie on/off; every seismic view has a 2D map
  showing where the line is. The previous tabbed explorer is kept in `dashboard/legacy_explorer.py`.

## Prospect simulation: 1D pseudo-well at a location (step 9 in the app)

```powershell
.\.venv\Scripts\python.exe scripts\run_prospect_sim.py --well Poseidon-2                          # blind test
.\.venv\Scripts\python.exe scripts\run_prospect_sim.py --il 3180 --xl 2337 --t0 3044 --t1 3292    # any location
```

Pick a location (map) and a TWT window (top and base on the section) in step 9 of the app, or use the script.
Thousands of pseudo-wells are simulated from the training wells (facies Markov chain and layer thicknesses,
correlated porosity / Vsh / Vp / Vs / density / GR per facies with depth trends, gas or brine via Gassmann),
forward-modelled to near/mid/far with the tie wavelets and scales, and weighted against the observed traces at
the location. Output: facies probabilities, P10/P50/P90 logs, P(gas | sand) and the expected vs observed AVO
response (`outputs/simulation/<name>.npz/.json`; `<name>_blind.csv` with scores for wells). Settings and all
assumptions are in `configs/prospect_sim.yaml`; held-out wells are never used.
