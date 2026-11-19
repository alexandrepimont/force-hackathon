# Handover: Poseidon AVO workflow (FORCE Hackathon 2026)

This note explains how the project works in practice: what the data is, how it gets onto disk, what every
script and module does, which assumptions are baked in, what the current results are, and the traps we hit.
It is written for an assistant picking the work up cold. Paths are relative to the repository root.

---

## 1. Goal and current state in one page

**End goal.** Classify the Plover Formation reservoir of the Poseidon field (Browse Basin, NW Australia) in 3D
by **lithotype** (shale, tight, low-porosity sand, high-porosity sand, volcanic) and, only if the data support
it, **fluid** (brine vs gas), with honest uncertainty. The link between rocks and seismic is learned at ~7 wells
via rock physics and synthetic near/mid/far angle-stack responses ("synthetic AVO"), then applied to the real
near/mid/far stacks. 1-2 wells are held out as blind tests. Many scenarios (Monte Carlo) are run to quantify
uncertainty. Everything is inspected in a Streamlit dashboard.

**Done so far**
1. Real well data and a 3D seismic sub-volume (near/mid/far) downloaded and verified.
2. Dashboard on real data: logs with tops, field map, crossplots, a fast 3D seismic viewer (inline, crossline,
   time slice, well-to-well line, 3D), each with a 2D location map, and a *Well QC & tie* tab.
3. Agent task 02 (well QC and petrophysics) run on all 8 wells.
4. Agent task 03 (automated well tie, wavelets, calibration) run; gate G2 passed.
5. A synthetic end-to-end pipeline (`src/avo/`) exists and runs on a synthetic Hugin-like earth (written for the
   earlier Volve plan) - this is what tasks 04-09 will reuse for Poseidon.
6. A junior-level textbook with figures: `docs/textbook/AVO_workflow_textbook.md` (+ self-contained `.html`).

**Not done yet (next steps, see section 12):** gas fluid model, Poseidon rock-physics/priors (04-05), forward
library and classifiers on real data (06-07), blind-well validation (08), 3D probability cubes (09).

---

## 2. Environment

- Windows, PowerShell 5.1. Repo path contains spaces (`C:\FORCE Hackathon 2026\force-hackathon`) - always quote.
- Python 3.12 in `.venv`. Always call `.\.venv\Scripts\python.exe` (activation does not persist between
  commands in agent shells).
- Install: `.\.venv\Scripts\python.exe -m pip install -r requirements.txt` then
  `.\.venv\Scripts\python.exe -m pip install -e .` (installs package `avo` from `src/`).
- `requirements.txt`: numpy, pandas, scikit-learn, matplotlib, pyyaml, segyio, pytest, plotly, streamlit,
  lasio, numcodecs. Also installed ad hoc (not required by the main flow): openpyxl, xlrd, markdown, pypdf.
- `pyproject.toml` sets pytest `pythonpath = ["src"]`. Tests: `.\.venv\Scripts\python.exe -m pytest -q`
  (cover the synthetic `avo` pipeline only).
- `data/` holds ~25 GB (mostly seismic + caches) and must never be added to version control.

---

## 3. Data: what we have and where it came from

All data are public: Geoscience Australia open-file data (CC BY 3.0 AU) and TGS Open Data (CC BY 4.0).
The authoritative portal, NOPIMS, has no scriptable download, so mirrors are used.

### 3.1 Wells (`data/poseidon/wells/`, ~32 MB) - `scripts/download_poseidon_wells.py`

| Folder | Content | Source |
|---|---|---|
| `logs/<well>/*Decim.LAS` | **Primary log run**: Occam Technology spliced composite LAS, 0.5 m, 8 wells (Boreas-1, Kronos-1, Pharos-1, Poseidon-1, Poseidon-2, Poseidon-North-1, Proteus-1 ST2, Torosa-1) | GitHub `WesleyTheGeolien/t20-Dash_Well_Viz` (Apache-2.0) |
| `logs_extra/` | VSP-calibrated depth logs (Kronos-1, Pharos-1, Proteus-1 ST2), Poseidon-1/-2 CSV logs, Torosa-3 package (logs, deviation, VSP checkshot levels, markers) | GitHub `aldimhd/Shearwave_Prediction`, `Firzalds/Gabor-Deconvolution-and-DTW-...` |
| `tops/t20/` | Well headers (X/Y GDA94 MGA51, DFE, TD MD/TVDSS), Poseidon-1 and Kronos-1 tops | t20 repo |
| `ga/` | Borehole headers and lithostratigraphic intervals | Geoscience Australia Boreholes WFS (`services.ga.gov.au/gis/boreholes/wfs`) |
| `tops/well_tops.csv` | Combined tops table (WELL, SURFACE, MD_M, SOURCE) built by the script | derived |
| `checkshots/`, `deviation/`, `reports/` | Only with `--drive`: ConocoPhillips Google Drive share; usually blocked by Google "Quota exceeded" | Drive |

Key facts:
- Tops exist for Kronos-1, Poseidon-1, Poseidon-2, Torosa-1 (full), Boreas-1 and Proteus-1 (Plover top only).
  **Pharos-1 and Poseidon-North-1 have no tops.**
- No deviation surveys for the Poseidon-area wells. Header TD(MD) vs TD(TVDSS) shows they are near-vertical
  (error at TD 1-15 m; Poseidon-1 worst at 15 m).
- Curve mnemonics differ per well (density = RHOB / RHOZ / HROM; P-sonic = DTCO / DTC / BATC (Torosa-1);
  neutron = TNPH / NPHI / TNP / HTNP in percent). Mapping is in `src/avo/wellqc.py::CURVE_MAP`.
- Time-depth sources: Kronos-1 (`ADJS` = VSP-adjusted one-way sonic time, depth index TVD below MSL), Pharos-1
  (VSP-calibrated DTCO from 0 m, integrated), Torosa-3 (VSP levels, TVD/OWT). Others have none.

### 3.2 Seismic (`data/poseidon/seismic/`, 3 x 3.2 GB SEG-Y) - `scripts/download_poseidon_seismic.py`

- Source: TGS MDIO (Zarr v2) copy of the GA/ConocoPhillips Poseidon 3D PSTM angle stacks
  (`https://tgs-opendata-poseidon.s3.amazonaws.com/{near,mid,far}_stack.mdio`), anonymous HTTP. The script reads
  the Zarr chunks (128^3, float32, blosc/zstd decoded with `numcodecs`) directly - no zarr library needed.
- Full stacks are ~104 GB each (3437 IL x 5053 XL x 1501 samples at 4 ms), so only a **sub-volume** is fetched:
  inlines 2310-3625, crosslines 1292-2769, TWT 2560-4092 ms (7 Poseidon-area wells + ~100 traces margin; top
  Plover is ~3.1 s). Torosa-1 (IL ~1194) is outside. Other windows: `--il --xl --t --name`.
- Output: SEG-Y rev1, IEEE float, inline-sorted, IL byte 189, XL byte 193, CDP X/Y bytes 181/185 with scalar
  -100, GDA94 / MGA zone 51. Loads with `avo.segy_io.read_cube`.
- Resumable: raw chunks are cached in `.chunk_cache/` during download (deleted when a stack completes),
  12 retries with backoff, and the PowerShell wrapper we used re-ran the script on failure. Download ~6-7 min
  per stack on a good connection; it once failed on a DNS blip before the retry logic was added.
- Documented facts (CGGVeritas processing report, in `configs/poseidon_seismic.yaml`):
  **near 6-18 deg, mid 18-30 deg, far 30-42 deg; reverse polarity (trough = increase in impedance)**; post-stack
  "time variant scaling" with undocumented parameters, so a common amplitude scale across stacks is **unknown**
  (treated as "conditional" eligibility; calibrate one global scale per stack, never per well).
- Bin size 18.75 m (IL) x 12.5 m (XL); survey is rotated relative to north.

### 3.3 Dashboard dataset (`data/poseidon/dashboard/`) - `scripts/prepare_poseidon_dashboard.py`

- `well_logs.parquet`: all wells, standard columns `WELL_ID, MD_M, TVDSS_M, FORMATION, GR_API, RT_OHM_M,
  RHOB_G_CC, DTC_US_FT, DTS_US_FT, CALI_IN, VP_M_S, VS_M_S, AI, VPVS, NPHI_VV` (TVDSS = MD - DFE, vertical).
- `well_tops.csv` (+TVDSS), `wells.csv` (X_M, Y_M, LAT, LON, DATUM_ELEV_M, TD_MD_M), `manifest.json`
  (`synthetic_data: false`, sources, assumptions).
- Also builds the **seismic display cache** (section 5.2) if missing (~40 s per stack).

---

## 4. End-to-end command sequence

```powershell
.\.venv\Scripts\python.exe scripts\download_poseidon_wells.py        # ~1 min
.\.venv\Scripts\python.exe scripts\download_poseidon_seismic.py      # ~20 min, ~10 GB
.\.venv\Scripts\python.exe scripts\prepare_poseidon_dashboard.py     # ~3 min incl. seismic cache
.\.venv\Scripts\python.exe scripts\run_well_qc.py                    # task 02, ~20 s
.\.venv\Scripts\python.exe scripts\run_well_tie.py                   # task 03, ~1-2 min
.\.venv\Scripts\streamlit.exe run dashboard\app.py                   # http://localhost:8501
.\.venv\Scripts\python.exe scripts\make_textbook_figures.py          # optional, docs figures
```

Order matters: QC writes `data/interim/poseidon_logs/`, the tie reads it and writes `twt` back into it.

---

## 5. Dashboard (`dashboard/app.py` + `src/dashboard_data.py`)

### 5.1 Structure

- Streamlit, Plotly. Top-level script; the sidebar picks the **data folder** (default
  `data/poseidon/dashboard`, falls back to `data/synthetic_reservoir`), the **well**, depth reference
  (MD / TVDSS) and depth window. `well_id` is the global "selected well" used by all tabs.
- Tabs: **Well logs** (tracks with tops), **Field overview** (well map coloured by a surface depth, tops
  correlation), **Rock-property crossplots** (incl. AI vs Vp/Vs, per well or all wells, formation filter),
  **Seismic** (`render_seismic_viewer`), **Well QC & tie** (`render_well_qc_tie`), **Data preview**.
  With the synthetic folder the second tab becomes the synthetic reservoir model and the seismic tab shows the
  synthetic angle gather.
- `render_seismic_viewer` is decorated with `@st.fragment`, so moving sliders only reruns that function (whole
  script reruns took ~5 s). Views:
  - **Well to well** (`render_traverse`): ordered multiselect of wells -> straight segments between their
    IL/XL positions (`traverse_indices`, one sample per trace step) -> traces read by fancy indexing
    (`read_seismic_traces`). Overlays: vertical well lines, yellow synthetic wiggles from the tie
    (`syn_after * scale`, same colour clip as the seismic), tops in tied time joined between wells.
    near / mid / far can be stacked vertically.
  - **Inline / Crossline / Time slice**: Plotly heatmaps, decimated to <= ~900 columns, `RdBu_r`, symmetric
    clip from a percentile of the first panel (same clip for all panels so near/mid/far brightness is comparable).
    Wells within N lines are drawn.
  - **3D** (`render_3d`): Plotly `go.Surface` for one inline, one crossline and one time slice (decimated to
    ~200 x 160), coordinates in metres **relative to the cube centre** (avoids float jitter), z = -TWT, a
    vertical-scale slider (aspect ratio), wells as vertical lines, tops as white dots. This is the "quick
    fallback" from the team's 3D notes; a three.js component (textured quads, click-to-move) is the upgrade path.
  - Every view has a **2D basemap** (`basemap`): survey outline (4 cube corners), wells, and the displayed line(s)
    in red; time slices shade the whole outline.
- `render_well_qc_tie` sub-tabs: **Well tie: what changed** (metrics: TD source + distance, shift, stretch,
  mean r with delta, training/held-out; QC flag banner; interactive tracks AI, Vp/Vs, and per stack seismic vs
  synthetic before (dashed grey) vs after (coloured) with tops; time-depth before/after and difference;
  wavelets; raw JSON with rejected donors), **Well QC** (QC PNG, summary row, curve provenance table),
  **All wells and gate G2** (tie summary, global calibration, QC summary, decision log).

### 5.2 Why the seismic is fast: the display cache

SEG-Y is trace-ordered; a time slice through 1.9 M traces took **~5 min** with segyio and reading all trace
headers for coordinates took **~9 min**. `build_seismic_cache` (in `src/dashboard_data.py`) converts each stack
once (~40 s, numpy memmap over the raw SEG-Y bytes - only valid for IEEE format 5 and a complete inline-sorted
grid; it falls back to segyio otherwise) into `data/poseidon/seismic/cache/`:
- `<stem>.f16.npy` (n_il, n_xl, ns) for inline/crossline reads,
- `<stem>.timeslices.f16.npy` (ns, n_il, n_xl) for time slices,
- `<stem>.geometry.npz` (ilines, xlines, samples, cdp_x, cdp_y, **scale**, source mtime).
Amplitudes reach ~3e5, so they are divided by the 99.9th percentile (`scale`) before float16; readers multiply
back. `seismic_cache_ready` checks the source mtime. Any section now reads in ~0.05 s. ~2.8 GB per stack.

### 5.3 Running it

`.\.venv\Scripts\streamlit.exe run dashboard\app.py` (or `python -m streamlit run ... --server.headless true`).
**Important:** Streamlit reloads `app.py` on change but **not** imported modules such as
`src/dashboard_data.py` - after editing those, stop the server process and start it again, otherwise you get
ImportError / stale behaviour. Test headlessly with `streamlit.testing.v1.AppTest` (all views were tested this
way; radio keys: `seis_view`, `seis_stack_False`, multiselect `seis_traverse`).

---

## 6. Well QC (agent 02) - `src/avo/wellqc.py`, `scripts/run_well_qc.py`, `configs/petrophysics.yaml`

Agent 02 (updated version) demands provenance, no silent edits, unknown stays unknown, and no fluid labels
from resistivity alone. Implementation:

1. **Read** each primary LAS (`read_primary_run`): every original curve kept as `raw_<MNEMONIC>`; normalised
   columns `gr, rt, rho, nphi (v/v), dtc, dts, cali` with a provenance record (source mnemonic, unit,
   conversion) saved in `data/interim/poseidon_logs/{well}.meta.json`.
2. **Depth**: `tvdss = md - DFE`, `tvdss_method = vertical_assumption`, error at TD from headers.
3. **Bad hole** (`flag_bad`, `flag_reason`, never deleted): washout = caliper > inferred nominal bit size +
   1.5 in, where the bit size is the standard size (6, 8.5, 12.25, 17.5 in) nearest the 200 m running median
   caliper (needed because Poseidon-2 has mixed hole sizes - a global median flagged 34 % of the zone); density,
   Vp and Vp/Vs range checks.
4. **Petrophysics**: Vsh = linear GR index between per-well P5/P95 below 3500 m TVDSS; porosity from
   neutron-density (average, or RMS where neutron < density = gas effect), density-only if no neutron;
   Archie Sw (a=1, m=2, n=2, **Rw = 0.04 ohm.m assumed**). Vp = 304800/DTC.
5. **Vs**: measured where DTS exists; else Greenberg-Castagna x bias factor fitted on **training wells only**
   (`vs_source` = measured / estimated / unavailable, `vs_uncertainty_rel`).
6. **Labels (scenario, not truth)**: `lithotype` = shale if Vsh >= 0.5 **or** neutron-minus-density porosity
   >= 0.12 (added because Jamieson claystone has moderate GR and was mislabelled sand); else tight (phi < 0.06),
   sand_low_phi (0.06-0.12), sand_high_phi (>= 0.12); `volcanic` between the "Top Volcanics" and "Top Reservoir"
   tops; `unknown` where inputs are missing. `litho` collapses sand classes. `fluid = unknown` always;
   `hc_indicator` = gas_indicated (Sw < 0.6 AND crossover > 0.02), possible (one of them), none.
7. **Outputs**: `data/interim/poseidon_logs/{well}.parquet` (+ `twt` filled later by the tie, `held_out`),
   `outputs/well_qc/{well}.png` (GR/Vsh, RHOB/NPHI, Vp/Vs measured vs estimated, phi, RT, Sw, lithotype with gas
   dots, bad-hole column, tops), `outputs/well_qc/summary.csv` (coverage per curve, flagged fractions, reservoir
   gross/net/NTG/phi/Sw, gas-indicated metres), `outputs/well_qc/vs_calibration.json`.
   Reservoir interval = first of "Plover Formation (Top Reservoir)" / "Plover Formation" to Nome or TD.

---

## 7. Automated well tie (agent 03) - `src/avo/welltie.py`, `scripts/run_well_tie.py`, `configs/well_tie.yaml`

1. **Time-depth per well** (`load_td_sources`, `select_td`): own checkshot-calibrated curve if the well has one
   (Kronos-1, Pharos-1); otherwise the **nearest eligible donor** by horizontal distance (GDA94 MGA51), rejecting
   donors > 25 km away or whose Plover top differs > 250 m; rejected donors and reasons are recorded. The
   donor's TWT(TVDSS) curve is applied at the target's TVDSS. Torosa-1 gets Torosa-3 but is outside the cube.
2. **Logs for synthetics** (`conditioned_logs`): longest continuous interval with Vp, Vs (measured or estimated),
   rho; flagged samples and gaps <= 5 m interpolated (counted and reported).
3. **Seismic at the well** (`well_traces`): mean of a 3 x 3 trace bundle around the nearest trace, from the cache.
4. **Synthetics** (`synthetics`): `avo.forward.model_reflectivity` (Aki-Richards averaged over each stack's
   documented angles, 1 ms model grid, blocky depth-to-time) convolved with a Ricker, multiplied by
   **polarity = -1** (reverse, documented). Correlation window = log interval minus 32 ms at each end.
5. **Automatic tie** (`auto_tie`): grid search of bulk shift (+/-20 ms with own TD, +/-60 ms with transferred
   TD, 1 ms steps) and linear stretch about the window centre (+/-2 %, 0.5 % steps), maximising the mean
   near/mid/far correlation with a small penalty on |shift| and |stretch| (prefers the smallest good shift,
   limits cycle skips).
6. **Wavelets** (`estimate_wavelets`): per stack, Ricker frequency 8-50 Hz and residual phase +/-60 deg, chosen
   to maximise mean correlation over **training wells only**. Two passes (pass 1 with a 25 Hz reference
   wavelet, pass 2 with estimated wavelets).
7. **Calibration**: one global amplitude scale per stack = RMS(observed)/RMS(synthetic) over training wells;
   noise = std of the residual. Polarity check compares reverse vs normal.
8. **QC flags per well**: shift >= 80 % of the limit (possible cycle skip), tie window < 200 ms, mean r < 0.5,
   any stack r < 0.3 -> `needs_review`.
9. **Outputs**: `data/interim/poseidon_td/{well}.csv` (md, tvdss, twt_initial_ms, twt_final_ms),
   `data/processed/wavelets/{near,mid,far}.npy|.json`, `configs/calibration.yaml` (GENERATED),
   `outputs/well_ties/{well}.npz` (everything the dashboard plots: t_ms, obs, syn_before/after, windows, logs in
   time, tops in tied time, scales), `{well}.json` (all metadata incl. rejected donors), `{well}.png`,
   `summary.csv`, and gate G2 in `outputs/decision_log.md` (pass if mean training correlation >= 0.5).

Held-out wells (**Poseidon-2, Boreas-1**, set in `configs/petrophysics.yaml: held_out_wells`) are tied (shift
only, so they can be evaluated later) but excluded from Vs calibration, wavelets, scales and noise.

---

## 8. Current results

**QC summary (Plover reservoir):** Boreas-1 gross 302 m / net 116 m, Kronos-1 299/41, Poseidon-1 281/132,
Poseidon-2 619/52, Proteus-1 330/80, Torosa-1 191/59; Pharos-1 and Poseidon-North-1 not computed (no tops).
Flagged fraction in the zone <= 2.4 % in all wells.

**Tie (gate G2 PASS; training mean r near 0.67, mid 0.66, far 0.57; polarity reverse +0.63 vs normal -0.63):**

| Well | TD from | Shift | Stretch | r before -> after | Flags |
|---|---|---|---|---|---|
| Kronos-1 | own VSP | +11 ms | 0 % | 0.18 -> 0.82 | short window (172 ms) |
| Boreas-1 (held out) | Pharos-1, 11.3 km | +21 | -0.5 | -0.57 -> 0.81 | ok |
| Proteus-1 | Pharos-1, 9.3 km | -21 | -0.5 | -0.21 -> 0.80 | short window (56 ms) |
| Poseidon-1 | Pharos-1, 11.9 km | +17 | -2.0 | -0.43 -> 0.74 | short window (160 ms) |
| Poseidon-2 (held out) | Kronos-1, 9.6 km | -17 | +0.5 | -0.26 -> 0.64 | ok |
| Pharos-1 | own VSP | +1 | +0.5 | 0.45 -> 0.48 | low r; far poor |
| Poseidon-North-1 | Pharos-1, 13.6 km | +53 | 0 | -0.05 -> 0.32 | shift near limit (cycle skip?) |

Wavelets: near 20 Hz / -10 deg, mid 19 Hz / -10 deg, far 19 Hz / +10 deg.

**Needs human review:** Poseidon-North-1 (likely cycle skip), Proteus-1 (56 ms window makes r unreliable),
Pharos-1 far stack. Short windows come from where Vp, Vs and density overlap in the spliced logs.

---

## 9. The synthetic pipeline (`src/avo/`) that later tasks will reuse

Written for the original Volve/Hugin plan and validated on a synthetic earth (`configs/synthetic_earth.yaml`,
`python -m avo.run_synthetic [--quick]`, report in `outputs/synthetic/report.md`).
- `data.py`: contracts `StackDef(name, angle_min, angle_max)`, `Well(name, il, xl, logs, checkshot, hugin_top,
  hugin_base)` with logs columns `tvdss, vp, vs, rho, phi, vsh, sw, litho, fluid, vs_source`, `Cube(stacks,
  stack_defs, t0, dt, ...)`. Real-data loaders must produce these. `Cube` holds full float64 arrays in memory -
  too big for the full Poseidon sub-volume (~6 GB per stack); crop or adapt to the float16 cache.
- `rock_physics.py`: Batzle-Wang brine and **live oil** (no gas yet), VRH mixing, Gassmann, critical-porosity
  frame, fluid substitution, porosity what-if, Greenberg-Castagna. Units GPa, g/cc, m/s.
- `forward.py`: Ricker, Aki-Richards, exact Zoeppritz, partial-stack reflectivity, depth-to-time, synthetics.
- `calibration.py`: synthetic-world tie (bulk shift, Ricker grid, RMS scale, noise) - the Poseidon tie in
  `welltie.py` is the real-data version with transferred TD, stretch, polarity and QC.
- `priors.py`: prior B (hand-picked Monte Carlo pseudo-wells) and prior C (block-resampled training wells with
  Gassmann relabelling). `library.py` builds the labelled synthetic library with wavelet/scale jitter.
- `features.py`, `inference.py`: windowed near/mid/far features, intercept/gradient, LDA baseline (method A)
  and gradient boosting (methods B, C).
- `pipeline.py`: leave-one-well-out validation, reliability, cube mapping, LOF out-of-distribution map.
- `export.py`: probability cubes as SEG-Y (geometry copied from a template stack) + manifest.
- `gates.py`, `report.py`, `plots.py`, `whatif.py`: gates G2/G3, decision log, figures, what-if tool.
- `avo/__init__.py`: class codes (litho shale/sand/tight; fluid none/brine/oil; joint classes).

`configs/pipeline.yaml` still describes Volve Hugin **oil** (P 30 MPa, T 110 C). Poseidon needs gas.

---

## 10. Agent task files (`agents/`)

`agents/README.md` + `01`-`10`. The current versions are Poseidon-specific and strict: inventory first,
preserve provenance, use held-out wells only when justified, do not assume a mid stack or angle ranges unless
documented (we have both documented), no fluid labels without test/fluid evidence, record gates G1-G4 in
`outputs/decision_log.md`. Note they mention near/far **CBVS** and an AGC full-stack MDIO volume (a
teammate's data); this project uses the 32-bit near/mid/far SEG-Y sub-volume instead - keep the AGC full stack
out of amplitude work.

---

## 11. Gotchas we hit (save time)

- Google Drive shared files often return an HTML "Quota exceeded" page; the downloader detects it and never
  saves HTML as data.
- Network blips (DNS `getaddrinfo failed`) killed a 15-minute download once; the seismic downloader now retries
  with backoff and caches chunks.
- segyio time slices and header scans are extremely slow on Windows; always use the dashboard cache or crop.
- Amplitudes ~1e5: never cast raw to float16; scale first.
- Streamlit does not reload imported modules: restart the server after changing `src/dashboard_data.py`.
- Plotly rejects numpy booleans in some properties (`showscale` needs a Python `bool`).
- `st.dataframe` of a transposed mixed-type row needs `.astype(str)` to avoid Arrow warnings.
- PowerShell: `Stop-Process` must use `-Id`; inline Python via `@' ... '@ | python -`; non-ASCII characters in
  inline Python strings can be mangled (use `\u` escapes or files).
- Duplicate surface names per well in tops (e.g. two "Plover Formation" picks in Poseidon-2 = top volcanics and
  top reservoir in GA naming) - aggregate with min depth before reindexing.
- Matplotlib text placed at data coordinates outside the axis limits silently expands figures; filter labels
  to the visible range.
- `pd.Series(..., dtype=object)` booleans mixed with NaN break `~` (cast with `.astype(bool)`).

---

## 12. Recommended next steps

1. **Review the three flagged ties** in the dashboard (*Well QC & tie*). For Poseidon-North-1 try a smaller
   shift limit or a different donor; consider widening tie windows by using more of the log interval where Vs is
   estimated.
2. **Pick tops for Pharos-1 and Poseidon-North-1** (Plover top, Top Reservoir, Nome) using the tied seismic and
   Kronos-1/Poseidon-1 as type wells; then rerun QC and tie.
3. **Gas fluid model**: add Batzle-Wang gas (or gas-condensate) to `rock_physics.py`; set Plover P/T
   (~4.8 km TVDSS) in a Poseidon pipeline config; run gate G3 (fluid sensitivity vs tie noise from
   `configs/calibration.yaml`).
4. **Real-data loaders**: build `avo.data.Well` objects from `data/interim/poseidon_logs/*.parquet` (+ tied
   twt, il/xl from the cache geometry) and a cropped `Cube` (or a cache-backed variant) so `pipeline.py`,
   `priors.py`, `library.py` and `inference.py` run on Poseidon with the calibrated wavelets/scales/noise.
5. **Horizon**: interpret/auto-track top Plover in the cube (seed at tied tops) to define the prediction window.
6. **Train A/B/C, run leave-one-well-out on training wells, then score once on Poseidon-2 and Boreas-1**
   (gate G4), and export probability cubes; add a probability/entropy overlay to the dashboard seismic views.
7. Optional: three.js 3D component per the team's 3D notes (textured quads, click-to-move sections).
