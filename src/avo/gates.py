"""Decision gates G2 (well tie quality) and G3 (fluid detectability via Gassmann)."""

import numpy as np
import pandas as pd

from avo import FLUID_OIL, LITHO_SAND
from avo import forward as fw
from avo import rock_physics as rp


def gate_g2(calib, cfg):
    mean_corr = calib.ties.groupby("stack")["corr"].mean()
    passed = bool((mean_corr >= cfg["calibration"]["g2_min_corr"]).all())
    return passed, mean_corr.to_dict()


def hugin_fluid_cases(well, fluids, minerals, sw_oil):
    """Logs with every Hugin sand sample substituted to 100% brine and to oil at sw_oil."""
    l = well.logs
    m = ((l["tvdss"] >= well.hugin_top) & (l["tvdss"] < well.hugin_base) & (l["litho"] == LITHO_SAND)).to_numpy()
    vp, vs, rho = (l[c].to_numpy().copy() for c in ("vp", "vs", "rho"))
    phi, vsh, sw = (l[c].to_numpy() for c in ("phi", "vsh", "sw"))
    cases = {}
    for name, target in (("brine", 1.0), ("oil", sw_oil)):
        a, b, c = vp.copy(), vs.copy(), rho.copy()
        a[m], b[m], c[m] = rp.substitute_fluid(vp[m], vs[m], rho[m], phi[m], vsh[m], sw[m], target, fluids, minerals)
        cases[name] = (a, b, c)
    return cases, m


def gate_g3(wells, cube, calib, cfg):
    """Compare the brine-vs-oil synthetic difference in Hugin with the calibrated noise level per stack."""
    fluids = rp.fluid_set(cfg["rock_physics"]["reservoir"])
    minerals = cfg["rock_physics"]["minerals"]
    wl = calib.wavelet_list(cube.stack_defs)
    rows, panels = [], {}
    for w in wells:
        cases, m = hugin_fluid_cases(w, fluids, minerals, cfg["gate_g3"]["sw_oil"])
        if not m.any():
            continue
        twt = w.twt() + calib.shifts.get(w.name, 0.0)
        syn = {k: fw.synthesize(twt, *v, cube.stack_defs, wl, cube.t, calib.dt_model) for k, v in cases.items()}
        t_res = twt[m]
        win = (cube.t >= t_res.min() - 0.01) & (cube.t <= t_res.max() + 0.01)
        diff = syn["oil"] - syn["brine"]
        for k, s in enumerate(cube.stack_defs):
            rms = float(np.sqrt(np.mean(diff[k][win] ** 2)))
            rows.append(dict(well=w.name, stack=s.name, diff_rms=rms, noise=calib.noise[s.name],
                             ratio=rms / calib.noise[s.name]))
        panels[w.name] = dict(syn=syn, twt_range=(t_res.min(), t_res.max()))
    table = pd.DataFrame(rows)
    best_stack_ratio = table.groupby("well")["ratio"].max().median() if len(table) else 0.0
    passed = bool(best_stack_ratio >= cfg["gate_g3"]["min_ratio"])
    return passed, float(best_stack_ratio), table, panels


def has_oil(well):
    return bool((well.logs["fluid"] == FLUID_OIL).any())
