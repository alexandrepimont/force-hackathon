"""1D Monte Carlo pseudo-well simulation at a prospect location, conditioned on the near/mid/far AVO signal.

Prior: facies layers from a Markov chain of vertical transitions and per-facies thickness distributions, with
correlated elastic and petrophysical properties per facies (Gaussian with a linear depth trend), all estimated from
the QC'd logs of the training wells. Held-out wells are never used. Fluid is a scenario (gas column or brine) that
is applied with Gassmann substitution.

Posterior: every realization is forward-modelled (Aki-Richards averaged over each stack's angle range, convolved
with the tie wavelets, documented reverse polarity) and weighted by its Gaussian likelihood against the observed
near/mid/far traces at the location (importance sampling). Results are simulated, not measured, data.

Units: depth m TVDSS, time s (two-way) inside the module, velocity m/s, density g/cm3.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.signal import fftconvolve

from . import forward as fw
from . import rock_physics as rp
from .data import StackDef

PROPS = ["phi", "vsh", "vp", "vs", "rho", "gr"]
CLIP = {"phi": (0.002, 0.35), "vsh": (0.0, 1.0), "vp": (1500.0, 7000.0), "vs": (700.0, 4200.0),
        "rho": (1.9, 3.05), "gr": (0.0, 250.0)}


# ----------------------------------------------------------------------------------------- fluids
def gas_properties(pressure, temperature, gravity):
    """Batzle and Wang (1992) hydrocarbon gas bulk modulus (GPa) and density (g/cm3)."""
    ta = np.asarray(temperature, dtype=float) + 273.15
    p = np.asarray(pressure, dtype=float)
    ppr = p / (4.892 - 0.4048 * gravity)
    tpr = ta / (94.72 + 170.75 * gravity)
    c = 0.45 + 8 * (0.56 - 1 / tpr) ** 2
    e = 0.109 * (3.85 - tpr) ** 2 * np.exp(-c * ppr ** 1.2 / tpr)
    z = (0.03 + 0.00527 * (3.5 - tpr) ** 3) * ppr + (0.642 * tpr - 0.007 * tpr ** 4 - 0.52) + e
    rho = 28.8 * gravity * p / (z * 8.31441 * ta)
    gamma0 = 0.85 + 5.6 / (ppr + 2) + 27.1 / (ppr + 3.5) ** 2 - 8.7 * np.exp(-0.65 * (ppr + 1))
    dz = 0.03 + 0.00527 * (3.5 - tpr) ** 3 + e * (-1.2 * c * ppr ** 0.2 / tpr)
    k = p * gamma0 / (1 - ppr / z * dz) / 1000.0
    return k, rho


def fluids_at(tvdss, fcfg):
    """Brine and gas properties at depth (arrays). Returned in the key layout used by rock_physics.fluid_mix."""
    z = np.asarray(tvdss, dtype=float)
    p = fcfg["pressure_gradient_mpa_per_m"] * z
    t = fcfg["seabed_temperature_c"] + fcfg["geothermal_gradient_c_per_m"] * np.maximum(z - fcfg["water_depth_m"], 0)
    k_b, rho_b = rp.brine_properties(p, t, fcfg["salinity_ppm"])
    k_g, rho_g = gas_properties(p, t, fcfg["gas_gravity"])
    return {"k_brine": k_b, "rho_brine": rho_b, "k_oil": k_g, "rho_oil": rho_g}


def substitute_gas(vp, vs, rho, phi, vsh, sw_to, fluids, fcfg):
    """Brine (Sw = 1) -> partial gas saturation sw_to with Gassmann. The dry bulk modulus is bounded below by
    kdry_floor_frac * K_sat(brine): drawn Vp/Vs/rho combinations that are not mutually consistent would otherwise
    invert to an unrealistically soft frame and exaggerate the fluid effect."""
    k_min, _, _ = rp.mineral_mix(vsh, 0.0, fcfg["minerals"])
    g = rho * (vs / 1000.0) ** 2
    k_sat = rho * (vp / 1000.0) ** 2 - 4.0 / 3.0 * g
    k_fl1, rho_fl1 = rp.fluid_mix(1.0, fluids)
    k_fl2, rho_fl2 = rp.fluid_mix(sw_to, fluids)
    k_dry = rp.gassmann_dry(k_sat, k_min, k_fl1, phi)
    k_dry = np.clip(k_dry, fcfg["kdry_floor_frac"] * k_sat, k_min)
    k_new = rp.gassmann_saturated(k_dry, k_min, k_fl2, phi)
    rho_new = rho + phi * (rho_fl2 - rho_fl1)
    return (*rp.velocities(k_new, g, rho_new), rho_new)


# ----------------------------------------------------------------------------------------- prior
@dataclass
class Prior:
    facies: list[str]
    proportions: np.ndarray          # (F,)
    transitions: np.ndarray          # (F, F) layer-to-layer, zero diagonal
    thickness: list[np.ndarray]      # metres, per facies
    trend_a: np.ndarray              # (F, P) intercept
    trend_b: np.ndarray              # (F, P) slope per metre
    chol: np.ndarray                 # (F, P, P) residual covariance Cholesky
    depth_range: np.ndarray          # (F, 2) training depth range
    sw_gas: np.ndarray               # Archie Sw of gas-indicated training sands
    wells: list[str]
    n_blocks: np.ndarray             # (F,)
    ar1: float = 0.0                 # correlation of (whitened) property residuals between consecutive layers
    ar1_calibration: dict | None = None

    def to_dict(self):
        return {"facies": self.facies, "proportions": self.proportions.round(4).tolist(),
                "transitions": self.transitions.round(4).tolist(), "wells": self.wells,
                "n_blocks": self.n_blocks.tolist(), "ar1_between_layers": round(self.ar1, 3),
                "ar1_calibration": self.ar1_calibration,
                "mean_thickness_m": [round(float(t.mean()), 2) if len(t) else None for t in self.thickness],
                "depth_range_m": self.depth_range.round(0).tolist(), "n_sw_gas": int(len(self.sw_gas))}


def load_training_logs(logs_dir: Path, held_out: list[str]) -> pd.DataFrame:
    frames = []
    for f in sorted(logs_dir.glob("*.parquet")):
        if f.stem in held_out:
            continue
        frames.append(pd.read_parquet(f))
    fr = pd.concat(frames, ignore_index=True)
    fr = fr[fr["held_out"].astype(str) != "True"]
    return fr


def brine_equivalent(fr: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Gassmann-substitute gas-indicated sands to brine (Sw = 1). Other samples are assumed brine already."""
    fcfg = cfg["fluid"]
    out = fr.copy()
    m = (out["hc_indicator"] == "gas_indicated") & out["lithotype"].isin(fcfg["gas_facies"])
    m &= out[["vp", "vs", "rho", "phi", "vsh", "sw"]].notna().all(axis=1)
    out["brine_substituted"] = False
    if m.any():
        d = out.loc[m]
        fl = fluids_at(d["tvdss"].to_numpy(), fcfg)
        vp, vs, rho = rp.substitute_fluid(d["vp"].to_numpy(), d["vs"].to_numpy(), d["rho"].to_numpy(),
                                          d["phi"].to_numpy(), d["vsh"].to_numpy(), d["sw"].to_numpy(), 1.0,
                                          fl, fcfg["minerals"])
        out.loc[m, "vp"], out.loc[m, "vs"], out.loc[m, "rho"] = vp, vs, rho
        out.loc[m, "brine_substituted"] = True
    return out


def _block(fr: pd.DataFrame, block_m: float, facies: list[str]) -> pd.DataFrame:
    """Blocks of block_m metres per well: majority facies and mean properties."""
    rows = []
    for well, g in fr.groupby("well"):
        g = g.sort_values("tvdss")
        b = np.floor(g["tvdss"] / block_m).astype(int)
        agg = g.groupby(b).agg(**{p: (p, "mean") for p in PROPS},
                               lithotype=("lithotype", lambda s: s.mode().iat[0] if len(s.mode()) else "unknown"),
                               bad=("flag_bad", "mean"), tvdss=("tvdss", "mean"))
        agg["well"] = well
        agg["block"] = agg.index
        rows.append(agg.reset_index(drop=True))
    blk = pd.concat(rows, ignore_index=True)
    blk["fid"] = blk["lithotype"].map({f: i for i, f in enumerate(facies)}).fillna(-1).astype(int)
    return blk


def build_prior(logs_dir: Path, held_out: list[str], cfg: dict) -> Prior:
    tcfg = cfg["training"]
    facies = list(tcfg["facies"])
    fr = load_training_logs(logs_dir, held_out)
    fr = fr[fr["tvdss"] >= tcfg["min_tvdss_m"]]
    fr = brine_equivalent(fr, cfg)
    sw_gas = fr.loc[(fr["hc_indicator"] == "gas_indicated") & fr["lithotype"].isin(cfg["fluid"]["gas_facies"]),
                    "sw"].dropna().to_numpy()
    blk = _block(fr, tcfg["block_m"], facies)

    # layers: runs of equal facies in consecutive blocks; unknown blocks and gaps break the sequence
    thickness = [[] for _ in facies]
    trans = np.zeros((len(facies), len(facies)))
    layer_arr = np.full(len(blk), -1)
    seq_arr = np.full(len(blk), -1)   # sequence id: consecutive layers without a break share it
    layer_no, seq_no = 0, 0
    for _, g in blk.groupby("well"):
        g = g.sort_values("block")
        fid, block, gidx = g["fid"].to_numpy(), g["block"].to_numpy(), g.index.to_numpy()
        start = 0
        prev_layer = None
        seq_no += 1
        for i in range(1, len(g) + 1):
            end_run = i == len(g) or fid[i] != fid[start] or block[i] != block[i - 1] + 1
            if not end_run:
                continue
            f = fid[start]
            if f >= 0:
                thickness[f].append((i - start) * tcfg["block_m"])
                if prev_layer is not None:
                    trans[prev_layer, f] += 1
                prev_layer = f
                layer_arr[gidx[start:i]] = layer_no
                seq_arr[gidx[start:i]] = seq_no
                layer_no += 1
            else:
                prev_layer = None
                seq_no += 1
            if i < len(g) and block[i] != block[i - 1] + 1:
                prev_layer = None
                seq_no += 1
            start = i
    blk["layer"], blk["seq"] = layer_arr, seq_arr
    np.fill_diagonal(trans, 0)
    trans = (trans + 0.5) * (1 - np.eye(len(facies)))   # small pseudo-count keeps every transition possible
    trans /= trans.sum(axis=1, keepdims=True)

    ok = blk[(blk["fid"] >= 0) & blk[PROPS].notna().all(axis=1) & (blk["bad"] < 0.5)]
    n_p = len(PROPS)
    a = np.zeros((len(facies), n_p))
    b = np.zeros((len(facies), n_p))
    chol = np.zeros((len(facies), n_p, n_p))
    depth_range = np.zeros((len(facies), 2))
    n_blocks = np.zeros(len(facies), dtype=int)
    # one property vector per layer (mean of its good blocks): the simulation draws one value per layer, so the
    # spread must be the spread of layer means, not of 1 m blocks
    lay = ok[ok["layer"] >= 0].groupby("layer").agg(**{p: (p, "mean") for p in PROPS}, tvdss=("tvdss", "mean"),
                                                     fid=("fid", "first"), seq=("seq", "first"),
                                                     nb=("tvdss", "size"))
    lay_res = np.full((len(lay), n_p), np.nan)
    for f in range(len(facies)):
        d = ok[ok["fid"] == f]
        n_blocks[f] = len(d)
        if len(d) < 5:
            raise ValueError(f"too few training blocks for facies {facies[f]} ({len(d)})")
        z = d["tvdss"].to_numpy()
        x = d[PROPS].to_numpy()
        depth_range[f] = [z.min(), z.max()]
        if len(d) >= tcfg["min_samples_trend"] and z.std() > 50:
            A = np.c_[np.ones_like(z), z - 4500.0]
            coef, *_ = np.linalg.lstsq(A, x, rcond=None)
            a[f], b[f] = coef[0], coef[1]
            res = x - A @ coef
        else:
            a[f] = x.mean(axis=0)
            res = x - a[f]
        m = (lay["fid"] == f).to_numpy()
        lres = lay.loc[m, PROPS].to_numpy() - (a[f] + b[f] * (lay.loc[m, "tvdss"].to_numpy()[:, None] - 4500.0))
        if m.sum() >= 8:
            res = lres
        cov = np.cov(res.T)
        cov = cov + np.eye(n_p) * 1e-9 * np.abs(np.diag(cov)).max()
        chol[f] = np.linalg.cholesky(cov)
        lay_res[m] = np.linalg.solve(chol[f], lres.T).T      # whitened layer residuals
    # vertical correlation of whitened residuals between consecutive layers in the same sequence (AR(1))
    lay = lay.sort_index()
    same = (lay["seq"].to_numpy()[1:] == lay["seq"].to_numpy()[:-1]) & (np.diff(lay.index.to_numpy()) == 1)
    r1, r0 = lay_res[1:][same], lay_res[:-1][same]
    ar1 = float(np.clip(np.nanmean([np.corrcoef(r0[:, k], r1[:, k])[0, 1] for k in range(n_p)]), 0.0, 0.95)) \
        if same.sum() > 10 else 0.0
    props = np.array([sum(t) for t in thickness], dtype=float)
    return Prior(facies, props / props.sum(), trans, [np.array(t) for t in thickness], a, b, chol, depth_range,
                 sw_gas, sorted(fr["well"].unique().tolist()), n_blocks, ar1)


# ----------------------------------------------------------------------------------------- simulation
def calibrate_ar1(prior: Prior, cfg: dict, calib: dict, ties_dir: Path, held_out: list[str]) -> dict:
    """Choose the between-layer correlation so that prior synthetics have the same RMS amplitude as the
    synthetics computed from the TRAINING wells' own logs (task 03 tie outputs; no seismic, no held-out wells).

    The per-facies property distributions are unchanged; only how strongly neighbouring layers differ."""
    target = []
    for f in sorted(ties_dir.glob("*.npz")):
        if f.stem in held_out:
            continue
        d = np.load(f)
        m = d["win_after"]
        target.append(np.sqrt((d["syn_after"][:, m] ** 2).mean(axis=1)))
    target = np.mean(target, axis=0)
    stacks = [StackDef(k, v["angle_min"], v["angle_max"]) for k, v in calib["stacks"].items()]
    wl = [fw.ricker(v["freq_hz"], 0.001, 0.128, v["phase_deg"]) for v in calib["stacks"].values()]
    t_out = np.arange(3.1, 3.4, 0.004)
    tm, r = fw.model_grid(t_out, 0.001)
    z0 = float(np.mean(prior.depth_range[:, 1])) - 400.0
    rows = []
    for a1 in [prior.ar1, 0.5, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97]:
        if a1 < prior.ar1:
            continue
        p = Prior(**{**prior.__dict__, "ar1": a1})
        lay = simulate_layers(p, cfg, 600, z0, tm[0] - 0.002, tm[-1] + 0.002, (z0 + 50, z0 + 500),
                              np.random.default_rng(11))
        g = layers_to_grid(lay, tm, ["vp", "vs", "rho"])
        syn = forward_batch(g, stacks, wl, r, -1.0)[:, :, 10:-10]
        rms = np.sqrt((syn ** 2).mean(axis=(0, 2)))
        rows.append((float(np.abs(np.log(rms / target)).mean()), a1, rms))
    best = min(rows)
    return {"ar1_estimated": round(prior.ar1, 3), "ar1_calibrated": best[1],
            "target_rms_training_synthetics": target.round(4).tolist(), "prior_rms": best[2].round(4).tolist(),
            "mean_abs_log_ratio": round(best[0], 3)}


def _draw_props(prior: Prior, f: np.ndarray, z: np.ndarray, margin: float, eps: np.ndarray) -> np.ndarray:
    lo = prior.depth_range[f, 0] - margin
    hi = prior.depth_range[f, 1] + margin
    zc = np.clip(z, lo, hi) - 4500.0
    mean = prior.trend_a[f] + prior.trend_b[f] * zc[:, None]
    x = mean + np.einsum("nij,nj->ni", prior.chol[f], eps)
    for k, p in enumerate(PROPS):
        x[:, k] = np.clip(x[:, k], *CLIP[p])
    x[:, 3] = np.minimum(x[:, 3], x[:, 2] / 1.4)   # Vp/Vs >= 1.4
    return x


def simulate_layers(prior: Prior, cfg: dict, n: int, z_start: float, t_start: float, t_end: float,
                    z_window: tuple[float, float], rng) -> dict:
    """Layered realizations from t_start to beyond t_end. Arrays are (n, L) with NaN padding."""
    fcfg = cfg["fluid"]
    margin = cfg["training"]["trend_depth_margin_m"]
    gas_f = np.array([f in fcfg["gas_facies"] for f in prior.facies])
    span = z_window[1] - z_window[0]
    has_gas = rng.random(n) < fcfg["p_gas_column"]
    contact = rng.uniform(z_window[0], z_window[1] + fcfg["contact_below_window_frac"] * span, n)
    contact[~has_gas] = -np.inf
    lo, hi = fcfg["sw_gas_clip"]
    sw_pool = np.clip(prior.sw_gas, lo, hi) if len(prior.sw_gas) else np.array([0.3])

    cum_t = np.full(n, t_start)
    cum_z = np.full(n, z_start)
    f = rng.choice(len(prior.facies), size=n, p=prior.proportions)
    eps = rng.standard_normal((n, len(PROPS)))
    a1 = prior.ar1
    cols = {k: [] for k in ["facies", "top_t", "top_z", "h", "phi", "vsh", "vp", "vs", "rho", "gr", "sw", "gas"]}
    cdf = np.cumsum(prior.transitions, axis=1)
    while True:
        active = cum_t < t_end
        if not active.any():
            break
        h = np.array([prior.thickness[fi][rng.integers(len(prior.thickness[fi]))] for fi in f])
        x = _draw_props(prior, f, cum_z + h / 2, margin, eps)
        eps = a1 * eps + np.sqrt(1 - a1 ** 2) * rng.standard_normal(eps.shape)
        phi, vsh, vp, vs, rho, gr = x.T
        gas = gas_f[f] & (cum_z + h / 2 < contact)
        sw = np.ones(n)
        if gas.any():
            sw[gas] = rng.choice(sw_pool, size=int(gas.sum()))
            fl = fluids_at(cum_z[gas] + h[gas] / 2, fcfg)
            vp_g, vs_g, rho_g = substitute_gas(vp[gas], vs[gas], rho[gas], phi[gas], vsh[gas], sw[gas], fl, fcfg)
            vp, vs, rho = vp.copy(), vs.copy(), rho.copy()
            vp[gas], vs[gas], rho[gas] = vp_g, vs_g, rho_g
        for k, v in zip(cols, [f, cum_t, cum_z, h, phi, vsh, vp, vs, rho, gr, sw, gas]):
            vals = np.asarray(v, dtype=float).copy()
            vals[~active] = np.nan
            cols[k].append(vals)
        cum_t = cum_t + np.where(active, 2 * h / vp, 0)
        cum_z = cum_z + np.where(active, h, 0)
        u = rng.random(n)[:, None]
        f = np.minimum((u > cdf[f]).sum(axis=1), len(prior.facies) - 1)
    out = {k: np.stack(v, axis=1) for k, v in cols.items()}
    out["has_gas"] = has_gas
    out["contact_tvdss"] = np.where(has_gas, contact, np.nan)
    return out


def layers_to_grid(lay: dict, t_grid: np.ndarray, keys: list[str]) -> dict:
    """Blocky sampling of layer properties on a regular time grid -> (n, nt) arrays."""
    top = np.nan_to_num(lay["top_t"], nan=np.inf)
    n = top.shape[0]
    idx = np.empty((n, len(t_grid)), dtype=int)
    for i in range(n):
        idx[i] = np.clip(np.searchsorted(top[i], t_grid, side="right") - 1, 0, top.shape[1] - 1)
    rows = np.arange(n)[:, None]
    out = {k: lay[k][rows, idx] for k in keys}
    if "top_z" in keys:   # depth at each time sample (linear inside each layer)
        dt = t_grid[None, :] - lay["top_t"][rows, idx]
        out["tvdss"] = lay["top_z"][rows, idx] + dt * lay["vp"][rows, idx] / 2
    return out


def forward_batch(grid: dict, stacks: list[StackDef], wavelets: list[np.ndarray], r: int, polarity: float,
                  n_angles: int = 5) -> np.ndarray:
    """Synthetic partial stacks (n, n_stacks, nt_out) from properties on the fine time grid."""
    vp, vs, rho = grid["vp"], grid["vs"], grid["rho"]
    vpm, vsm, rhom = 0.5 * (vp[:, 1:] + vp[:, :-1]), 0.5 * (vs[:, 1:] + vs[:, :-1]), 0.5 * (rho[:, 1:] + rho[:, :-1])
    dvp, dvs, drho = np.diff(vp, axis=1), np.diff(vs, axis=1), np.diff(rho, axis=1)
    k = (vsm / vpm) ** 2
    out = []
    for st, wl in zip(stacks, wavelets):
        refl = np.zeros_like(vp)
        for th in fw.stack_angles(st, n_angles):
            s2 = np.sin(th) ** 2
            refl[:, 1:] += (0.5 * (1 - 4 * k * s2) * drho / rhom + dvp / (2 * vpm * np.cos(th) ** 2)
                            - 4 * k * s2 * dvs / vsm)
        refl /= n_angles
        syn = fftconvolve(refl, wl[None, :], mode="same", axes=1)[:, r // 2::r]
        out.append(polarity * syn)
    return np.stack(out, axis=1)


HIST_RANGE = {"phi": (0.0, 0.35), "vsh": (0.0, 1.0), "sw": (0.0, 1.0), "vp": (1500.0, 7000.0),
              "vs": (700.0, 4200.0), "rho": (1.9, 3.05), "gr": (0.0, 250.0), "ai": (3000.0, 21000.0),
              "vpvs": (1.3, 3.3)}
N_BINS = 240


def _hist_add(hist, values, weights, lo, hi):
    """hist (nt, N_BINS) += weighted histogram of values (n, nt) per time sample."""
    nt = values.shape[1]
    b = np.clip(((values - lo) / (hi - lo) * N_BINS).astype(int), 0, N_BINS - 1)
    flat = (np.arange(nt)[None, :] * N_BINS + b).ravel()
    hist += np.bincount(flat, weights=weights.ravel(), minlength=nt * N_BINS).reshape(nt, N_BINS)


def _hist_quantiles(hist, lo, hi, q=(0.1, 0.5, 0.9)):
    cdf = np.cumsum(hist, axis=1)
    cdf /= np.maximum(cdf[:, -1:], 1e-300)
    edges = np.linspace(lo, hi, N_BINS + 1)
    out = np.empty((len(q), hist.shape[0]))
    for k, qq in enumerate(q):
        i = (cdf < qq).sum(axis=1).clip(0, N_BINS - 1)
        c0 = np.where(i > 0, cdf[np.arange(len(i)), i - 1], 0.0)
        c1 = cdf[np.arange(len(i)), i]
        frac = np.clip((qq - c0) / np.maximum(c1 - c0, 1e-12), 0, 1)
        out[k] = edges[i] + frac * (edges[1] - edges[0])
    return out


def run_simulation(*, prior: Prior, cfg: dict, calib: dict, t_seis_s: np.ndarray, obs: np.ndarray,
                   t_top_s: float, t_base_s: float, td: tuple[np.ndarray, np.ndarray], n: int | None = None,
                   seed: int | None = None, progress=None) -> dict:
    """Prior Monte Carlo + localised importance weighting. obs: (3, nt) raw amplitudes on t_seis_s.

    Pass 1 forward-models every realization and stores its squared misfit per output sample. Local weights at
    output sample j use the misfit within +/- the localisation half-window. Pass 2 regenerates the same
    realizations (same seeds) and accumulates weighted histograms / facies counts on the fine time grid.
    """
    rng = np.random.default_rng(cfg["seed"] if seed is None else seed)
    n = n or cfg["n_realizations"]
    lcfg = cfg["likelihood"]
    stacks = [StackDef(k, v["angle_min"], v["angle_max"]) for k, v in calib["stacks"].items()]
    dt_model = 0.001
    wavelets = [fw.ricker(v["freq_hz"], dt_model, 0.128, v["phase_deg"]) for v in calib["stacks"].values()]
    scales = np.array([v["scale"] for v in calib["stacks"].values()])
    sigma = np.array([v["noise_std_synthetic_units"] for v in calib["stacks"].values()]) * lcfg["noise_inflation"]
    polarity = -1.0

    pad = cfg["pad_ms"] / 1000
    sel = (t_seis_s >= t_top_s - pad) & (t_seis_s <= t_base_s + pad)
    t_out = t_seis_s[sel]
    obs_n = obs[:, sel] / scales[:, None]
    score = (t_out >= t_top_s) & (t_out <= t_base_s)
    t_model, r = fw.model_grid(t_out, dt_model)
    dt_out_ms = (t_out[1] - t_out[0]) * 1000

    f_dom = np.mean([v["freq_hz"] for v in calib["stacks"].values()])
    corr_ms = 1000 / (2 * f_dom) if lcfg["correlation_length_ms"] == "auto" else float(lcfg["correlation_length_ms"])
    temper = 1.0 / max(corr_ms / dt_out_ms, 1.0)
    half = int(round(lcfg["localization_half_window_ms"] / dt_out_ms))

    td_t, td_z = td
    z_of_t = lambda t: float(np.interp(t, td_t, td_z))
    t_start = t_model[0] - 0.002
    t_stop = t_model[-1] + 0.002
    z_start = z_of_t(t_start)
    z_window = (z_of_t(t_top_s), z_of_t(t_base_s))

    def batch(s, nb):
        return simulate_layers(prior, cfg, nb, z_start, t_start, t_stop, z_window, np.random.default_rng(s))

    # pass 1: misfit per realization and output sample
    bs = cfg["batch_size"]
    seeds = [(b0, min(bs, n - b0), int(rng.integers(2 ** 31))) for b0 in range(0, n, bs)]
    mis = np.empty((n, len(t_out)), dtype=np.float32)
    for k, (b0, nb, s) in enumerate(seeds):
        grid = layers_to_grid(batch(s, nb), t_model, ["vp", "vs", "rho"])
        syn = forward_batch(grid, stacks, wavelets, r, polarity)
        mis[b0:b0 + nb] = (((syn - obs_n[None]) / sigma[None, :, None]) ** 2).sum(axis=1)
        if progress:
            progress(0.5 * (k + 1) / len(seeds), "forward modelling")
    mis[:, ~score] = 0.0

    # local log-likelihood: moving sum of the misfit over +/- half samples, only inside the scored window
    c = np.concatenate([np.zeros((n, 1), np.float32), np.cumsum(mis, axis=1, dtype=np.float64)], axis=1)
    j = np.arange(len(t_out))
    lo_i, hi_i = np.clip(j - half, 0, len(t_out)), np.clip(j + half + 1, 0, len(t_out))
    loc_ll = -0.5 * temper * (c[:, hi_i] - c[:, lo_i])
    loc_ll -= loc_ll.max(axis=0, keepdims=True)
    sc_cols = np.flatnonzero(score)

    def local_weights(beta):
        w = np.exp(beta * loc_ll)
        return w / w.sum(axis=0, keepdims=True)

    def min_ess(beta):
        w = np.exp(beta * loc_ll[:, sc_cols])
        w /= w.sum(axis=0, keepdims=True)
        return float((1.0 / (w ** 2).sum(axis=0)).min())

    # adaptive tempering: soften the likelihood only if the prior cannot reproduce the data well enough to give
    # min_ess_target effective realizations everywhere in the window (beta = 1 means no softening)
    beta = 1.0
    target = min(lcfg.get("min_ess_target", 200), n / 10)
    if min_ess(1.0) < target:
        lo_b, hi_b = 0.0, 1.0
        for _ in range(25):
            mid = 0.5 * (lo_b + hi_b)
            lo_b, hi_b = (mid, hi_b) if min_ess(mid) >= target else (lo_b, mid)
        beta = max(lo_b, 1e-4)
    w_loc = local_weights(beta)
    ess_loc = 1.0 / (w_loc ** 2).sum(axis=0)
    glob_ll = -0.5 * temper * beta * mis.sum(axis=1)
    w_glob = np.exp(glob_ll - glob_ll.max())
    w_glob /= w_glob.sum()
    ess_glob = float(1.0 / (w_glob ** 2).sum())
    del c, loc_ll

    # fine-grid sample -> output sample whose local weight applies
    col = np.clip(np.round((t_model - t_out[0]) / (t_out[1] - t_out[0])).astype(int), 0, len(t_out) - 1)
    nt, nf = len(t_model), len(prior.facies)
    acc = {name: {"hist": {p: np.zeros((nt, N_BINS)) for p in HIST_RANGE}, "tvdss": np.zeros((nt, N_BINS)),
                  "facies": np.zeros((nt, nf)), "gas": np.zeros(nt), "syn": np.zeros((len(stacks), len(t_out), N_BINS)),
                  "gas_column": 0.0} for name in ("prior", "post")}
    z_lo, z_hi = z_start - 50.0, z_start + 2500.0
    syn_lo, syn_hi = -0.4, 0.4
    best = np.argsort(glob_ll)[::-1][:lcfg["n_best"]]
    examples = {"best": best, "prior": np.sort(rng.choice(n, size=min(lcfg["n_prior_examples"], n), replace=False))}
    kept = {"best": {}, "prior": {}}
    keys = ["facies", "phi", "vsh", "vp", "vs", "rho", "gr", "sw", "gas", "top_z"]
    for k, (b0, nb, s) in enumerate(seeds):
        lay = batch(s, nb)
        g = layers_to_grid(lay, t_model, keys)
        g["ai"], g["vpvs"] = g["vp"] * g["rho"], g["vp"] / g["vs"]
        syn = forward_batch(g, stacks, wavelets, r, polarity)
        for name in ("prior", "post"):
            if name == "prior":
                wf = np.full((nb, nt), 1.0 / n)
                wo = np.full((nb, len(t_out)), 1.0 / n)
                wg = np.full(nb, 1.0 / n)
            else:
                wo = w_loc[b0:b0 + nb]
                wf = wo[:, col]
                wg = w_glob[b0:b0 + nb]
            a = acc[name]
            for p, (lo, hi) in HIST_RANGE.items():
                _hist_add(a["hist"][p], g[p], wf, lo, hi)
            _hist_add(a["tvdss"], g["tvdss"], wf, z_lo, z_hi)
            fac = g["facies"].astype(int)
            a["facies"] += np.stack([((fac == f) * wf).sum(axis=0) for f in range(nf)], axis=1)
            a["gas"] += (np.nan_to_num(g["gas"]) * wf).sum(axis=0)
            a["gas_column"] += float((lay["has_gas"] * wg).sum())
            for kk in range(len(stacks)):
                _hist_add(a["syn"][kk], syn[:, kk], wo, syn_lo, syn_hi)
        for ex, idx in examples.items():
            local = idx[(idx >= b0) & (idx < b0 + nb)] - b0
            for kk in keys[:-1] + ["tvdss"]:
                kept[ex].setdefault(kk, []).append(g[kk][local])
            kept[ex].setdefault("syn", []).append(syn[local])
            kept[ex].setdefault("has_gas", []).append(lay["has_gas"][local])
            kept[ex].setdefault("loglik", []).append(glob_ll[b0 + local])
        if progress:
            progress(0.5 + 0.5 * (k + 1) / len(seeds), "posterior statistics")
    kept = {ex: {kk: np.concatenate(v) for kk, v in d.items()} for ex, d in kept.items()}
    order = np.argsort(kept["best"]["loglik"])[::-1]
    kept["best"] = {kk: v[order] for kk, v in kept["best"].items()}

    summary = {}
    for name, a in acc.items():
        summary[f"{name}_facies_prob"] = (a["facies"] / a["facies"].sum(axis=1, keepdims=True)).T
        summary[f"{name}_p_gas"] = a["gas"] / a["facies"].sum(axis=1)
        summary[f"{name}_p_gas_column"] = a["gas_column"]
        for p, (lo, hi) in HIST_RANGE.items():
            summary[f"{name}_{p}_q"] = _hist_quantiles(a["hist"][p], lo, hi)
        summary[f"{name}_tvdss_q"] = _hist_quantiles(a["tvdss"], z_lo, z_hi)
        summary[f"{name}_syn_q"] = np.stack([_hist_quantiles(a["syn"][kk], syn_lo, syn_hi)
                                             for kk in range(len(stacks))], axis=1)
    return {"t_model_s": t_model, "t_out_s": t_out, "score_mask": score, "obs_norm": obs_n, "summary": summary,
            "examples": kept, "ess_local": ess_loc, "ess_global": ess_glob, "n": n, "temper": temper,
            "sigma": sigma, "scales": scales, "stacks": [s.name for s in stacks], "beta": beta,
            "z_window": z_window, "facies": prior.facies, "half_window_ms": half * dt_out_ms}


# ----------------------------------------------------------------------------------------- saving
def save_result(path: Path, res: dict, meta: dict) -> dict:
    """npz with grids, summaries and example pseudo-wells + json metadata (one call = one checkpoint)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    s = res["summary"]
    arrays = {"t_model_ms": res["t_model_s"] * 1000, "t_out_ms": res["t_out_s"] * 1000,
              "score_mask": res["score_mask"], "obs_norm": res["obs_norm"], "sigma": res["sigma"],
              "ess_local": res["ess_local"], "facies_names": np.array(res["facies"]),
              "stacks": np.array(res["stacks"]), **{k: v for k, v in s.items() if np.ndim(v) > 0}}
    for ex, d in res["examples"].items():
        for k, v in d.items():
            arrays[f"ex_{ex}_{k}"] = np.asarray(v, dtype=np.float32 if np.ndim(v) > 1 else None)
    np.savez_compressed(path.with_suffix(".npz"), **arrays)
    sc = res["score_mask"]
    meta = {**meta, "n_realizations": res["n"], "likelihood_temper": round(res["temper"], 4),
            "adaptive_beta": round(res["beta"], 4),
            "rms_obs": [round(float(v), 4) for v in np.sqrt((res["obs_norm"][:, sc] ** 2).mean(axis=1))],
            "rms_prior_spread": [round(float(v), 4) for v in
                                   np.sqrt(((0.5 * (s["prior_syn_q"][2] - s["prior_syn_q"][0]))[:, sc] ** 2).mean(axis=1))],
            "localization_half_window_ms": res["half_window_ms"],
            "ess_local_min": round(float(res["ess_local"][sc].min()), 1),
            "ess_local_median": round(float(np.median(res["ess_local"][sc])), 1),
            "ess_global": round(res["ess_global"], 1),
            "p_gas_column_prior": round(s["prior_p_gas_column"], 3),
            "p_gas_column_posterior_global": round(s["post_p_gas_column"], 3),
            "z_window_tvdss_m": [round(v, 1) for v in res["z_window"]],
            "label": "SIMULATED pseudo-well (not measured data)"}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=float), encoding="utf-8")
    return meta


def load_configs(root: Path) -> tuple[dict, dict, dict]:
    cfg = yaml.safe_load((root / "configs" / "prospect_sim.yaml").read_text(encoding="utf-8"))
    calib = yaml.safe_load((root / "configs" / "calibration.yaml").read_text(encoding="utf-8"))
    petro = yaml.safe_load((root / "configs" / "petrophysics.yaml").read_text(encoding="utf-8"))
    return cfg, calib, petro


def location_td(root: Path, x: float, y: float, held_out: list[str]) -> tuple[np.ndarray, np.ndarray, str, float]:
    """Time-depth of the nearest training well with a final tied time-depth curve (TWT s, TVDSS m)."""
    wells = pd.read_csv(root / "data" / "poseidon" / "dashboard" / "wells.csv")
    best = None
    for r in wells.itertuples():
        f = root / "data" / "interim" / "poseidon_td" / f"{r.WELL_ID}.csv"
        if r.WELL_ID in held_out or not f.exists():
            continue
        td = pd.read_csv(f)
        if "twt_final_ms" not in td or td["twt_final_ms"].notna().sum() < 10:
            continue
        d = float(np.hypot(r.X_M - x, r.Y_M - y))
        if best is None or d < best[0]:
            best = (d, r.WELL_ID, td.dropna(subset=["twt_final_ms"]))
    d, well, td = best
    return td["twt_final_ms"].to_numpy() / 1000, td["tvdss"].to_numpy(), well, d / 1000

def prepare_prior(root: Path, cfg: dict, calib: dict, held_out: list[str]) -> Prior:
    """Prior from the training wells, with the between-layer correlation calibrated (if configured)."""
    prior = build_prior(root / "data" / "interim" / "poseidon_logs", held_out, cfg)
    if cfg["training"].get("calibrate_layer_correlation", True):
        cal = calibrate_ar1(prior, cfg, calib, root / "outputs" / "well_ties", held_out)
        prior.ar1, prior.ar1_calibration = cal["ar1_calibrated"], cal
    return prior


def run_at_location(root: Path, il: int, xl: int, t0_ms: float, t1_ms: float, *, n: int | None = None,
                    name: str | None = None, well: str | None = None, prior: Prior | None = None,
                    progress=None) -> dict:
    """Simulate a pseudo-well at (il, xl) over the TWT window [t0_ms, t1_ms] and save it to outputs/simulation."""
    import time

    from .welltie import well_traces

    t_start = time.time()
    cfg, calib, petro = load_configs(root)
    held_out = petro["held_out_wells"]
    tie_cfg = yaml.safe_load((root / "configs" / "well_tie.yaml").read_text(encoding="utf-8"))
    cache = root / "data" / "poseidon" / "seismic" / "cache"
    g = np.load(cache / "poseidon_near.geometry.npz")
    i, j = int(np.searchsorted(g["ilines"], il)), int(np.searchsorted(g["xlines"], xl))
    x, y = float(g["cdp_x"][i, j]), float(g["cdp_y"][i, j])
    if not (g["samples"][0] + cfg["pad_ms"] <= t0_ms < t1_ms <= g["samples"][-1] - cfg["pad_ms"]):
        raise ValueError("window must lie inside the seismic time range (minus the modelling pad)")
    prior = prior or prepare_prior(root, cfg, calib, held_out)
    td_t, td_z, td_well, td_km = location_td(root, x, y, held_out)
    t_seis, obs = well_traces(cache, list(calib["stacks"]), il, xl, tie_cfg["trace_bundle"])
    res = run_simulation(prior=prior, cfg=cfg, calib=calib, t_seis_s=t_seis, obs=obs, t_top_s=t0_ms / 1000,
                         t_base_s=t1_ms / 1000, td=(td_t, td_z), n=n, progress=progress)
    name = name or (f"well_{well}" if well else f"il{il}_xl{xl}_{int(t0_ms)}_{int(t1_ms)}")
    meta = {"name": name, "inline": int(il), "crossline": int(xl), "x": x, "y": y,
            "window_twt_ms": [float(t0_ms), float(t1_ms)], "well": well,
            "well_is_held_out": bool(well in held_out) if well else None,
            "training_wells": prior.wells, "held_out_wells_excluded": held_out,
            "time_depth_from": td_well, "time_depth_distance_km": round(td_km, 2),
            "trace_selection": f"mean of {tie_cfg['trace_bundle']}x{tie_cfg['trace_bundle']} traces",
            "runtime_s": round(time.time() - t_start, 1), "config": cfg}
    out = root / "outputs" / "simulation"
    out.mkdir(parents=True, exist_ok=True)
    (out / "prior.json").write_text(json.dumps(prior.to_dict(), indent=2), encoding="utf-8")
    return save_result(out / name, res, meta)
