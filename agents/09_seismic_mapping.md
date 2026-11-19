# 09 — Seismic application and mapping

## Goal

Apply the best validated method to a **small 3D subvolume around Hugin**. Produce maps that geoscientists can judge against known structure, wells, and contacts.

## Inputs

- Best method from task 08, with A always included for comparison.
- Near, mid, and far stacks, plus the Hugin top horizon from task 01, or a horizon picked from well tops if needed.

## Tasks

1. Define a subvolume, for example the field outline plus a margin, and a time window around Hugin top.
2. Apply the method trace by trace, or in windows. Parallelize if needed.
3. Extract horizon maps along Hugin top or as an average over the interval:
   - `p_sand`.
   - `p_oil`, only if gate G3 passed.
   - Entropy.
   - Out-of-distribution score: distance from the observed features to the synthetic library.
4. Overlay well locations, blind-well results, known fluid contacts, and structure contours.
5. Produce an inline and a crossline section through the main wells.

## Outputs

- `data/processed/maps/*.npy` or GeoTIFF.
- `outputs/maps/*.png`.

## Acceptance criteria

- Maps have consistent geometry and coordinates, and wells plot in the correct positions.
- Every map has a short caption that states what it shows and what it does not show.

## Pitfalls

- A plausible map is not validation because we already know the field. Use it as a consistency check, not as proof.
- High out-of-distribution areas should be shown as low support, not presented as confident predictions.
