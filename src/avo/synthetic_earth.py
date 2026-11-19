"""Synthetic Hugin-like truth earth: wells, checkshots, interpreted horizon and noisy partial stacks.

Used to test the full workflow where the answer is known everywhere. Real data replaces this module.
"""

import numpy as np
import pandas as pd

from avo import FLUID_BRINE, FLUID_NONE, FLUID_OIL, LITHO_SAND, LITHO_SHALE, LITHO_TIGHT, joint_class
from avo import forward as fw
from avo import rock_physics as rp
from avo.data import Cube, StackDef, Well


def earth_column(x, y, cfg, z):
    """Rock description along depth z for map position x (west->east) and y (south->north) in [0, 1]."""
    s, h, ob, ub, ti = cfg["structure"], cfg["hugin"], cfg["overburden"], cfg["underburden"], cfg["tight"]
    top = s["crest_depth"] + s["relief"] * ((x - 0.5) ** 2 + (y - 0.5) ** 2) / 0.5
    thick = h["thickness_min"] + (h["thickness_max"] - h["thickness_min"]) * x
    base = top + thick
    n = len(z)
    litho = np.full(n, LITHO_SHALE)
    phi = np.full(n, ob["phi"])
    vsh = np.full(n, ob["vsh"])
    calcite = np.zeros(n)

    def set_tight(mask):
        litho[mask], phi[mask], vsh[mask], calcite[mask] = LITHO_TIGHT, ti["phi"], ti["vsh"], ti["calcite"]

    st_thk = ob["stringer_thickness"] * np.clip((y - 0.3) / 0.2, 0, 1)
    st_top = top - ob["stringer_above_top"]
    set_tight((z >= st_top) & (z < st_top + st_thk))

    rel = (z - top) / thick
    res = (rel >= 0) & (rel < 1)
    litho[res] = LITHO_SAND
    phi[res] = h["phi_top"] + (h["phi_base"] - h["phi_top"]) * rel[res] + h["phi_lateral"] * (y - 0.5)
    vsh[res] = h["vsh"]
    sb = h["shale_break"]
    sb_top = top + sb["rel_depth"] * thick
    sb_mask = res & (z >= sb_top) & (z < sb_top + sb["max_thickness"] * np.clip((x - 0.3) / 0.3, 0, 1))
    litho[sb_mask], phi[sb_mask], vsh[sb_mask] = LITHO_SHALE, ob["phi"], ob["vsh"]
    cs = h["cemented_stringer"]
    cs_top = top + cs["rel_depth"] * thick
    set_tight(res & (z >= cs_top) & (z < cs_top + cs["max_thickness"] * np.clip((0.6 - y) / 0.3, 0, 1)))

    below = z >= base
    ub_sand = below & (((z - base) / ub["period"]) % 1.0 < ub["sand_fraction"])
    ub_shale = below & ~ub_sand
    litho[ub_sand], phi[ub_sand], vsh[ub_sand] = LITHO_SAND, ub["sand_phi"], ub["sand_vsh"]
    phi[ub_shale], vsh[ub_shale] = ub["shale_phi"], ub["shale_vsh"]

    f = cfg["fluids"]
    fluid = np.where(litho == LITHO_SAND, FLUID_BRINE, FLUID_NONE)
    sw = np.ones(n)
    oil = res & (litho == LITHO_SAND) & (z < f["owc"])
    fluid[oil] = FLUID_OIL
    sw[oil] = f["sw_oil"] + (1 - f["sw_oil"]) * np.clip((z[oil] - (f["owc"] - f["transition"])) / f["transition"], 0, 1)
    return dict(litho=litho, fluid=fluid, phi=phi, vsh=vsh, calcite=calcite, sw=sw, top=top, base=base)


def _elastic(col, fluids, cfg):
    return rp.elastic_from_rock(col["phi"], col["vsh"], col["calcite"], col["sw"], fluids, cfg["minerals"],
                                cfg["rock_physics"]["phic"], cfg["rock_physics"]["exponent"])


def build_synthetic_dataset(cfg):
    """Return (wells, cube) for the synthetic test."""
    rng = np.random.default_rng(cfg["seed"])
    d, sc = cfg["depth"], cfg["seismic"]
    z = np.arange(d["z_top"], d["z_base"], d["dz"])
    t_top = 2.0 * d["z_top"] / cfg["overburden_velocity"]
    stacks = [StackDef(s["name"], s["angle_min"], s["angle_max"]) for s in sc["stacks"]]
    wavelets = [fw.ricker(sc["wavelets"][s.name]["freq"], sc["dt_model"], phase_deg=sc["wavelets"][s.name]["phase"])
                for s in stacks]
    t_out = sc["t0"] + np.arange(sc["nt"]) * sc["dt"]
    fluids = rp.fluid_set(cfg["reservoir"])
    n_il, n_xl = cfg["grid"]["n_il"], cfg["grid"]["n_xl"]

    clean = np.zeros((len(stacks), n_il, n_xl, sc["nt"]))
    truth = np.zeros((n_il, n_xl, sc["nt"]), dtype=int)
    horizon = np.zeros((n_il, n_xl))
    columns = {}
    for il in range(n_il):
        for xl in range(n_xl):
            x, y = xl / (n_xl - 1), il / (n_il - 1)
            col = earth_column(x, y, cfg, z)
            vp, vs, rho = _elastic(col, fluids, cfg)
            twt = fw.twt_from_vp(vp, d["dz"], t_top)
            clean[:, il, xl] = fw.synthesize(twt, vp, vs, rho, stacks, wavelets, t_out, sc["dt_model"])
            truth[il, xl] = fw.labels_on_grid(twt, joint_class(col["litho"], col["fluid"]), t_out, sc["dt_model"], 4)
            horizon[il, xl] = np.interp(col["top"], z, twt)
            columns[(il, xl)] = (col, vp, vs, rho, twt)

    cube_stacks = {}
    for k, s in enumerate(stacks):
        sigma = np.sqrt(np.mean(clean[k] ** 2)) / sc["snr"]
        noise = np.stack([fw.bandlimited_noise(sc["nt"], sc["wavelets"][s.name]["freq"], sc["dt"], sigma, rng)
                          for _ in range(n_il * n_xl)]).reshape(n_il, n_xl, sc["nt"])
        cube_stacks[s.name] = (clean[k] + noise) * sc["scale"][s.name]
    horizon += rng.normal(0, sc["horizon_noise_ms"] / 1000.0, horizon.shape)
    g = cfg["geometry"]
    ilines, xlines = g["il0"] + np.arange(n_il), g["xl0"] + np.arange(n_xl)
    cdp_x = np.broadcast_to(g["x0"] + np.arange(n_xl)[None, :] * g["bin"], (n_il, n_xl)).copy()
    cdp_y = np.broadcast_to(g["y0"] + np.arange(n_il)[:, None] * g["bin"], (n_il, n_xl)).copy()
    cube = Cube(cube_stacks, stacks, sc["t0"], sc["dt"], horizon_twt=horizon, truth_class=truth,
                meta={"source": "synthetic", "true_wavelets": sc["wavelets"], "true_scale": sc["scale"],
                      "ilines": ilines, "xlines": xlines, "cdp_x": cdp_x, "cdp_y": cdp_y})

    wells = []
    for w in cfg["wells"]:
        col, vp, vs, rho, twt = columns[(w["il"], w["xl"])]
        noise = 1 + cfg["log_noise"] * rng.standard_normal((3, len(z)))
        logs = pd.DataFrame(dict(tvdss=z, vp=vp * noise[0], vs=vs * noise[1], rho=rho * noise[2],
                                 phi=np.clip(col["phi"] + rng.normal(0, 0.01, len(z)), 0.0, 0.4),
                                 vsh=np.clip(col["vsh"] + rng.normal(0, 0.03, len(z)), 0.0, 1.0),
                                 sw=col["sw"], litho=col["litho"], fluid=col["fluid"]))
        logs["vs_source"] = "measured"
        if w["name"] in cfg.get("vs_missing", []):
            logs["vs"] = rp.vs_greenberg_castagna(logs["vp"], logs["vsh"])
            logs["vs_source"] = "estimated"
        cs_z = np.arange(d["z_top"], d["z_base"], cfg["checkshot"]["spacing"])
        cs_t = np.interp(cs_z, z, twt) + rng.normal(0, cfg["checkshot"]["noise_ms"] / 1000.0, len(cs_z))
        wells.append(Well(w["name"], w["il"], w["xl"], logs, pd.DataFrame(dict(tvdss=cs_z, twt=cs_t)),
                          float(col["top"]), float(col["base"])))
    return wells, cube
