"""Well tie and stack calibration (task 03): bulk shift, wavelet, global scale and noise per stack.

Only the wells passed in are used, so blind wells can be excluded explicitly.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from avo import forward as fw


@dataclass
class Calibration:
    wavelets: dict          # stack name -> {"freq", "phase"}
    scales: dict            # stack name -> amplitude scale (observed / synthetic)
    noise: dict             # stack name -> noise std in synthetic units
    shifts: dict            # well name -> bulk time shift (s) added to well TWT
    ties: pd.DataFrame      # well, stack, corr
    train_wells: list
    wavelet_length: float
    dt_model: float

    def wavelet_list(self, stack_defs, freq_scale=None, phase_add=None):
        out = []
        for k, s in enumerate(stack_defs):
            p = self.wavelets[s.name]
            f = p["freq"] * (1.0 if freq_scale is None else freq_scale[k])
            ph = p["phase"] + (0.0 if phase_add is None else phase_add[k])
            out.append(fw.ricker(f, self.dt_model, self.wavelet_length, ph))
        return out

    def scale_observed(self, traces, stack_defs):
        return traces / np.array([self.scales[s.name] for s in stack_defs])[:, None]


def _well_arrays(well):
    l = well.logs
    return well.twt(), l["vp"].to_numpy(), l["vs"].to_numpy(), l["rho"].to_numpy()


def _corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def tie_window(twt, t_out, margin):
    return (t_out >= twt[0] + margin) & (t_out <= twt[-1] - margin)


def calibrate(wells, cube, cfg):
    """Estimate shifts, Ricker wavelet (freq, phase), scale and noise per stack from the given wells."""
    c = cfg["calibration"]
    dt_model, length = cfg["dt_model"], c["wavelet_length"]
    stacks, t_out = cube.stack_defs, cube.t
    freqs = np.arange(*c["freqs"])
    phases = np.arange(c["phases"][0], c["phases"][1] + 1e-9, c["phases"][2])
    margin = length / 2

    data = {}
    for w in wells:
        twt, vp, vs, rho = _well_arrays(w)
        obs = cube.trace(w.il, w.xl)
        win = tie_window(twt, t_out, margin)
        ref_w = fw.ricker(c["reference_freq"], dt_model, length)
        best = (-np.inf, 0.0)
        for s_ms in range(-c["max_shift_ms"], c["max_shift_ms"] + 1):
            refl, r = fw.model_reflectivity(twt, vp, vs, rho, stacks[:1], t_out, dt_model, s_ms / 1000.0)
            cc = _corr(fw.convolve_decimate(refl[0], ref_w, r)[win], obs[0][win])
            best = max(best, (cc, s_ms / 1000.0))
        refl, r = fw.model_reflectivity(twt, vp, vs, rho, stacks, t_out, dt_model, best[1])
        data[w.name] = dict(refl=refl, r=r, obs=obs, win=win, shift=best[1])

    wavelets, scales, noise, ties = {}, {}, {}, []
    for k, s in enumerate(stacks):
        best = (-np.inf, None)
        for f in freqs:
            for ph in phases:
                wl = fw.ricker(f, dt_model, length, ph)
                score = np.mean([_corr(fw.convolve_decimate(d["refl"][k], wl, d["r"])[d["win"]], d["obs"][k][d["win"]])
                                 for d in data.values()])
                if score > best[0]:
                    best = (score, (float(f), float(ph)))
        f, ph = best[1]
        wavelets[s.name] = {"freq": f, "phase": ph}
        wl = fw.ricker(f, dt_model, length, ph)
        syn = {n: fw.convolve_decimate(d["refl"][k], wl, d["r"])[d["win"]] for n, d in data.items()}
        obs = {n: d["obs"][k][d["win"]] for n, d in data.items()}
        # RMS ratio rather than least squares: LS is biased low by the tie correlation, which would
        # inflate scaled observed amplitudes relative to the synthetic library.
        scales[s.name] = float(np.sqrt(sum(obs[n] @ obs[n] for n in data) / sum(syn[n] @ syn[n] for n in data)))
        resid = np.concatenate([obs[n] / scales[s.name] - syn[n] for n in data])
        noise[s.name] = float(resid.std())
        ties += [dict(well=n, stack=s.name, corr=_corr(syn[n], obs[n])) for n in data]

    return Calibration(wavelets, scales, noise, {n: d["shift"] for n, d in data.items()}, pd.DataFrame(ties),
                       [w.name for w in wells], length, dt_model)
