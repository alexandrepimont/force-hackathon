# 06 — AVO forward modelling and synthetic library

## Goal

Convert elastic sequences into realistic synthetic near, mid, and far traces with known labels. The result is the training library for task 07.

## Inputs

- Sequences from priors B and C in task 05.
- Wavelets, per-stack scale, noise, and angle ranges from task 03.

## Tasks

1. Convert depth sequences to time using interval Vp.
2. Compute angle-dependent reflectivity:
   - Use Aki–Richards as the default.
   - Use full Zoeppritz as a check.
   - Integrate reflectivity over each stack's angle range to mimic a partial stack.
3. Convolve with the per-stack wavelet. Apply the calibrated scale and add realistic noise, including band-limited and correlated noise rather than only white noise.
4. Add **domain-gap perturbations**: small time shifts, wavelet phase and frequency jitter, and residual scaling errors within the ranges measured in task 03.
5. Store each sample as a short time window around the target. Include `near`, `mid`, `far`, the label sequence, the target label at the window centre, and scenario parameters.
6. Verify the code by forward-modelling the training wells' real logs and reproducing the synthetics from task 03.

## Outputs

- `src/avo_forward.py`.
- `data/processed/library/{B,C}.npz` or Zarr, plus metadata.
- `outputs/forward/`: example gathers and an amplitude-variation-with-angle plot for typical classes.

## Acceptance criteria

- Forward-modelling the training well logs matches task 03 output.
- Library size is large enough for training. Start with about 1e5 windows and scale up if needed.
- Real-versus-synthetic attribute distributions at the training wells overlap; plot them.

## Pitfalls

- Thin beds and tuning are part of the signal. Do not model only isolated interfaces.
- Do not use perturbation ranges wider than measured just to make the classifier robust. That can wash out the fluid signal.
