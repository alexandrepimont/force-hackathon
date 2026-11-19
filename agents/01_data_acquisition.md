# 01 — Data inventory and verification

## Goal

Verify and catalogue the Volve and supporting datasets for a well-tied AVO study of the Hugin Formation. Record exactly what is available and what is missing.

**The team downloads the Volve data manually.** This agent does not download Volve data. It works only with files the team has placed in `data/raw/`.

## Inputs

- Volve files downloaded by the team into `data/raw/{seismic,wells,horizons,docs}/`. The source is the [Equinor Volve page](https://www.equinor.com/energy/volve-data-sharing).
- Local files in `C:\FORCE Hackathon 2026\`:
  - FORCE 2020 LAS zip with 118 wells.
  - NPD lithostratigraphy tables.
  - RealPore core workbook.
- Public SODIR Factpages for the Volve field and wellbores, including formation tops, DST results, and completion reports.

## Tasks

1. List the files in `data/raw/`. Check them against the wish list below, and report anything missing so the team can download it:
   - **ST10010 time-migrated angle stacks:** near, mid, and far, plus the full stack and migration velocity if available.
   - **Volve wells that penetrate Hugin:** LAS or DLIS logs, deviation surveys, checkshots or VSP, and formation tops.
   - **Interpretations, if available:** horizons, especially Hugin top, and fault interpretation.
2. Open each stack with `segyio`. Record the angle range per stack, sample rate, polarity convention, inline/crossline and coordinate headers, and survey vintage. Use the documentation for metadata that is not in the headers.
3. Check each well file's curves, depth coverage through Hugin, units, and coordinate reference system.
4. Build `data/interim/well_index.csv` with these columns:
   `well, source, x, y, has_vp, has_vs, has_rho, has_gr, has_nphi, has_rt, has_checkshot, has_trajectory, hugin_top_md, hugin_base_md, inside_seismic, known_fluid`.
5. Note which FORCE wells, especially in quadrant 15/9, overlap with or lie near Volve.

## Outputs

- `data/interim/well_index.csv`.
- `outputs/data_inventory.md`: what was found, file sizes, and a **missing-data list** for the team to download.

## Acceptance criteria

- The near, mid, and far stacks load with `segyio`, and their geometry is understood.
- At least 3 wells with Hugin coverage are listed, together with the logs available for each.
- Gate **G1** is evaluated and recorded in `outputs/decision_log.md`.

## Pitfalls

- Do not download, move, rename, or modify files in `data/raw/`. Treat them as read-only.
- Do not assume a curve contains valid data in Hugin just because it exists in a LAS header. Check depth coverage.
- Survey vintage matters. Production changed fluids after 2008, so compare well logging dates with the seismic acquisition date.
- Do not commit raw data.
