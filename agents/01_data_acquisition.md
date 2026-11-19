# 01 — Poseidon data inventory and verification

## Goal

Inventory the Poseidon files already in `data/poseidon/` and establish which well-log, marker, time-depth, and seismic inputs are actually usable. Record metadata, gaps, and provenance before selecting a scientific scope. Do not download additional data or assume a missing product exists.

## Inputs

- `data/poseidon/wells/`: supplied logs, extra well files, checkshots, tops, and geological/stratigraphic CSVs.
- `data/poseidon/NearStack_8bit.cbvs` and `data/poseidon/FarStack_8bit.cbvs`.
- `data/poseidon/full_stack_agc.mdio/`.
- Any data dictionaries, project exports, and license/attribution notes supplied with the data.

## Tasks

1. Recursively list files, sizes, formats, and checksums for the Poseidon tree. Identify duplicate or alternate well runs without merging them. Include the synthetic-reservoir directory only as a separate software-test fixture, not as Poseidon evidence.
2. Inventory well files by well and source/run. Record curve mnemonics, units, sample interval, depth range, null/sentinel conventions, and available coordinates. Include the `.LAS` files, time/depth variants, deviation/VSP files, tops/markers, and GA CSVs where present.
3. Inspect the CBVS stacks and MDIO full stack with compatible metadata readers or the source application. Record whether each can be decoded, dimensionality, sample interval, geometry, coordinate/datum references, polarity, processing (including AGC), and any documented angle range. Do not guess missing metadata. Do not treat full stack, near stack, and far stack as interchangeable calibrated angles.
4. Build a well-to-seismic coverage table only after coordinate systems and units are known. Checkshot/time-depth and trajectory coverage must be reported per well and source file.
5. Record known limitations, including unreadable formats, absent metadata, missing curves, uncertain units, unknown CRS, and lack of independently confirmed lithology/fluid labels. Cite the source file for each inventory claim.

## Outputs

- `data/interim/poseidon_well_index.csv` with source/run, available curves, depth ranges, markers, coordinates/reference, and time-depth coverage.
- `outputs/data_inventory.md` with file catalog, metadata, provenance, readability, and a prioritized missing-information list (not an assumed download list).
- G1 assessment in `outputs/decision_log.md`.

## Acceptance criteria

- The inventory can be traced to actual local files and distinguishes all source variants.
- Every seismic volume is either characterized from metadata or explicitly marked undecoded/unverified; no angle or geometry is invented.
- Well availability and seismic/time-depth compatibility are reported per well, without imposing a minimum well count before seeing the inventory.
- Gate G1 is recorded with its evidence and a realistic scope decision.

## Pitfalls

- Treat all supplied source files as read-only. Do not rename, move, rewrite, or commit raw data.
- A mnemonic in a LAS header does not prove valid measurements or documented units; inspect values and coverage.
- Do not infer coordinates, datum, time units, stack angles, or polarity from a filename.
- Keep AGC processing and 8-bit encoding visible as possible amplitude-calibration limitations.
- Do not treat the synthetic reservoir, planted labels, or generated gathers as field observations.
