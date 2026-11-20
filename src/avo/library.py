"""Synthetic training library (task 06): pseudo-wells -> noisy partial stacks with known labels."""

import numpy as np

from avo import collapse_fluid, joint_class
from avo import forward as fw
from avo.features import window_features


def build_library(pseudo_wells, calib, stack_defs, dt_out, cfg, rng, ig_op, use_fluid):
    lib = cfg["library"]
    half = lib["window_half"]
    n_st = len(stack_defs)
    X, y = [], []
    for pw in pseudo_wells:
        twt = fw.twt_from_vp(pw["vp"], pw["dz"], 0.1 + rng.uniform(0, dt_out))
        t_out = np.arange(0.0, twt[-1] + 0.1, dt_out)
        wl = calib.wavelet_list(stack_defs, freq_scale=1 + lib["freq_jitter"] * rng.standard_normal(n_st),
                                phase_add=lib["phase_jitter"] * rng.standard_normal(n_st))
        traces = fw.synthesize(twt, pw["vp"], pw["vs"], pw["rho"], stack_defs, wl, t_out, calib.dt_model)
        traces *= (1 + lib["scale_jitter"] * rng.standard_normal(n_st))[:, None]
        for k, s in enumerate(stack_defs):
            traces[k] += fw.bandlimited_noise(len(t_out), calib.wavelets[s.name]["freq"], dt_out,
                                              calib.noise[s.name], rng)
        labels = fw.labels_on_grid(twt, joint_class(pw["litho"], pw["fluid"]), t_out, calib.dt_model, 4)
        centers = np.flatnonzero(labels >= 0)
        centers = centers[(centers >= half) & (centers < len(t_out) - half)]
        if centers.size == 0:
            continue
        X.append(window_features(traces, centers, half, ig_op))
        y.append(labels[centers])
    y = np.concatenate(y)
    return np.vstack(X), (y if use_fluid else collapse_fluid(y))
