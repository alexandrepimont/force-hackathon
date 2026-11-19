# 03 — Well tie, wavelet, and stack calibration

## Goal

Connect wells and seismic in time and amplitude, so synthetic and real angle stacks can be compared directly. **This task is the critical bridge from synthetic to real data.**

## Inputs

- Clean logs from 02.
- Near, mid, and far stacks plus stack metadata from 01.
- Checkshots and VSP data.

## Tasks

1. Build a time–depth relation for each well from checkshots, with sonic calibrated to the checkshots.
2. Extract seismic traces along the well path, or a small averaged trace bundle around it, for each stack.
3. Estimate one wavelet per stack from the **training wells only**: statistical first, then well-based if time allows. Record phase, frequency band, and polarity.
4. Compute the synthetic near, mid, and far responses at each training well. Use angles matching the stack definitions; see task 06 for the modelling code.
5. Fit a **global amplitude scale per stack** on the training wells. Do not fit a separate free scale for every well. Report the bulk shift and any stretch or squeeze applied.
6. Measure tie quality: correlation and time shift per stack and well.
7. Estimate the noise level per stack, for example from the residual between the real data and the synthetic.

## Outputs

- `data/interim/td/{well}.csv`: MD to TWT.
- `data/processed/wavelets/{stack}.npy`, plus metadata JSON.
- `configs/calibration.yaml`: scale and noise for each stack.
- `outputs/well_ties/{well}.png`: synthetic versus real for near, mid, and far.
- `outputs/well_ties/summary.csv`.

## Acceptance criteria

- Gate **G2:** At the main tie well, the correlation in a window around Hugin is reasonable and visually convincing. Record the threshold you use.
- Relative near/mid/far amplitudes are preserved. Per-stack scaling must not hide the AVO trend.

## Pitfalls

- **Leakage:** Do not use blind wells for wavelets, scaling, or noise estimation.
- Too much stretch and squeeze can produce a good-looking tie that is not meaningful. Limit it and report what was applied.
- Polarity errors flip the interpretation. Confirm polarity against a known strong reflector.
