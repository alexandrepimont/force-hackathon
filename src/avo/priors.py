"""Scenario priors (task 05).

Prior B: hand-picked Monte Carlo ranges, as in the constitution.
Prior C: geology learned from training wells by stratigraphic block resampling, with Gassmann
         fluid relabelling so fluid cases the wells do not cover are still represented.
Both return pseudo-wells: dict(vp, vs, rho, litho, fluid, dz).
"""

import numpy as np

from avo import FLUID_BRINE, FLUID_NONE, FLUID_OIL, LITHO_SAND, LITHO_SHALE, LITHO_TIGHT
from avo import rock_physics as rp


def _u(rng, lo_hi):
    return rng.uniform(lo_hi[0], lo_hi[1])


def sample_prior_b(n, cfg, dz, rng):
    p = cfg["prior_b"]
    fluids = rp.fluid_set(cfg["rock_physics"]["reservoir"])
    minerals = cfg["rock_physics"]["minerals"]
    out = []
    for _ in range(n):
        length = _u(rng, p["length_m"])
        z = np.arange(0.0, length, dz)
        m = len(z)
        litho = np.full(m, LITHO_SHALE)
        phi = np.full(m, _u(rng, p["shale_phi"]))
        vsh = np.full(m, _u(rng, p["shale_vsh"]))
        calcite = np.zeros(m)
        sw = np.ones(m)
        fluid = np.full(m, FLUID_NONE)

        top = _u(rng, p["reservoir_top_frac"]) * length
        thick = min(_u(rng, p["reservoir_thickness"]), length - top - 20.0)
        base = top + thick

        for _ in range(rng.poisson(p["stringers_per_100m"] * top / 100.0)):
            s0 = rng.uniform(0, top)
            msk = (z >= s0) & (z < s0 + _u(rng, p["stringer_thickness"]))
            litho[msk], phi[msk], calcite[msk], vsh[msk] = LITHO_TIGHT, _u(rng, p["tight_phi"]), \
                _u(rng, p["tight_calcite"]), 0.05

        n_beds = rng.integers(p["n_beds"][0], p["n_beds"][1] + 1)
        edges = top + thick * np.r_[0, np.cumsum(rng.dirichlet(np.ones(n_beds)))]
        for b in range(n_beds):
            msk = (z >= edges[b]) & (z < edges[b + 1])
            litho[msk], phi[msk], vsh[msk] = LITHO_SAND, _u(rng, p["sand_phi"]), _u(rng, p["sand_vsh"])
            if b > 0 and rng.random() < p["interbed_prob"]:
                ib = (z >= edges[b]) & (z < edges[b] + _u(rng, p["interbed_thickness"]))
                litho[ib], phi[ib], vsh[ib] = LITHO_SHALE, _u(rng, p["shale_phi"]), _u(rng, p["shale_vsh"])

        zc = base
        while zc < length:
            bed = _u(rng, p["underburden_bed_thickness"])
            msk = (z >= zc) & (z < zc + bed)
            if rng.random() < p["underburden_sand_fraction"]:
                litho[msk], phi[msk], vsh[msk] = LITHO_SAND, _u(rng, p["sand_phi"]) * 0.8, _u(rng, p["sand_vsh"])
            zc += bed

        sand = litho == LITHO_SAND
        fluid[sand] = FLUID_BRINE
        if rng.random() < p["p_oil"]:
            owc = top + rng.uniform(-0.2, 1.2) * thick
            oil = sand & (z >= top) & (z < base) & (z < owc)
            fluid[oil], sw[oil] = FLUID_OIL, _u(rng, p["sw_oil"])

        vp, vs, rho = rp.elastic_from_rock(phi, vsh, calcite, sw, fluids, minerals, p["phic"],
                                           _u(rng, p["frame_exponent"]))
        out.append(dict(vp=vp, vs=vs, rho=rho, litho=litho, fluid=fluid, dz=dz))
    return out


def _segments(well):
    l = well.logs
    cols = ["vp", "vs", "rho", "phi", "vsh", "sw", "litho", "fluid"]
    z = l["tvdss"].to_numpy()
    arr = {c: l[c].to_numpy() for c in cols}
    over = z < well.hugin_top
    res = (z >= well.hugin_top) & (z < well.hugin_base)
    under = z >= well.hugin_base
    return {k: {c: v[m] for c, v in arr.items()} for k, m in (("over", over), ("res", res), ("under", under))}


def _take(seg, n, from_end, rng):
    m = len(seg["vp"])
    n = min(n, m)
    s = slice(m - n, m) if from_end else slice(0, n)
    return {c: v[s] for c, v in seg.items()}


def _stretch(seg, factor):
    m = len(seg["vp"])
    k = max(2, int(round(m * factor)))
    idx = np.minimum((np.arange(k) / factor).astype(int), m - 1)
    return {c: v[idx] for c, v in seg.items()}


def sample_prior_c(n, wells, cfg, rng):
    """Pseudo-wells assembled from overburden, Hugin and underburden blocks of training wells."""
    p = cfg["prior_c"]
    fluids = rp.fluid_set(cfg["rock_physics"]["reservoir"])
    minerals = cfg["rock_physics"]["minerals"]
    dz = wells[0].dz
    segs = [_segments(w) for w in wells]
    out = []
    for _ in range(n):
        over = _take(segs[rng.integers(len(segs))]["over"], int(_u(rng, p["overburden_m"]) / dz), True, rng)
        res_a = segs[rng.integers(len(segs))]["res"]
        if rng.random() < p["p_split"]:
            res_b = segs[rng.integers(len(segs))]["res"]
            cut_a = rng.integers(1, len(res_a["vp"]))
            cut_b = rng.integers(0, len(res_b["vp"]) - 1)
            res = {c: np.r_[res_a[c][:cut_a], res_b[c][cut_b:]] for c in res_a}
        else:
            res = res_a
        res = _stretch(res, _u(rng, p["thickness_stretch"]))
        under = _take(segs[rng.integers(len(segs))]["under"], int(_u(rng, p["underburden_m"]) / dz), False, rng)

        res = {c: v.copy() for c, v in res.items()}
        sand = res["litho"] == LITHO_SAND
        target_sw = np.ones(len(sand))
        target_fluid = np.where(sand, FLUID_BRINE, FLUID_NONE)
        if rng.random() < p["p_oil"]:
            contact = rng.uniform(-0.2, 1.2) * len(sand)
            oil = sand & (np.arange(len(sand)) < contact)
            target_sw[oil], target_fluid[oil] = _u(rng, p["sw_oil"]), FLUID_OIL
        if sand.any():
            res["vp"][sand], res["vs"][sand], res["rho"][sand] = rp.substitute_fluid(
                res["vp"][sand], res["vs"][sand], res["rho"][sand], res["phi"][sand], res["vsh"][sand],
                res["sw"][sand], target_sw[sand], fluids, minerals)
        res["fluid"], res["sw"] = target_fluid, target_sw

        pw = {c: np.r_[over[c], res[c], under[c]] for c in ("vp", "vs", "rho", "litho", "fluid")}
        pw["dz"] = dz
        out.append(pw)
    return out
