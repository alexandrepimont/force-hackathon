"""3D extrapolation of the 1D pseudo-well simulation (prospect_sim.py) over a decimated seismic grid.

Method (same physics and likelihood as the 1D runs, so the 1D blind tests carry over):
1. Target surface: TWT and TVDSS of the first pick of a formation (default Plover) in each TRAINING well with a tied
   time-depth curve, interpolated as a least-squares plane (+ inverse-distance residuals with more than 3 wells).
   The scored window follows this surface. Held-out wells are not used.
2. Shared library: pseudo-wells are simulated and forward-modelled once per depth band of the surface (the prior only
   depends on location through depth), in a time frame relative to the surface.
3. Every trace of the grid re-weights the library of its band with localised importance weighting against its own
   near/mid/far window (adaptive tempering per trace). Outputs per sample: porosity P10/P50/P90, weighted means of
   the elastic properties, facies probabilities and P(gas).
4. Laterally consistent realizations (P-field): a spatially correlated uniform field picks, at every trace and
   sample, a library member from the local posterior ordered by porosity, so neighbouring traces select similar
   members; facies, porosity and gas come from the same member and are mutually consistent.

All results are SIMULATED from the training wells and conditioned on the seismic; they are not measured data.
Units: TWT ms in the interface (s inside the forward model), depth m TVDSS, velocity m/s, density g/cm3.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import gaussian_filter
from scipy.special import ndtr

from . import forward as fw
from . import prospect_sim as ps
from .data import StackDef
from .welltie import well_traces

DT_MS = 4.0
DT_MODEL = 0.001
T_REF = 3.0                      # library time frame: the target surface sits at T_REF seconds
LIB_PROPS = ["phi", "vsh", "vp", "vs", "rho", "sw"]
MEAN_PROPS = ["vp", "vs", "rho", "vsh", "ai"]
STACKS = ["near", "mid", "far"]


def load_config(root: Path) -> dict:
    return yaml.safe_load((root / "configs" / "prospect_sim3d.yaml").read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------------------- target surface
def top_surface(root: Path, held_out: list[str], hcfg: dict) -> dict:
    """Control points of the target surface from the training wells (first pick of the formation)."""
    wells = pd.read_csv(root / "data" / "poseidon" / "dashboard" / "wells.csv")
    tops = pd.read_csv(root / "data" / "poseidon" / "dashboard" / "well_tops.csv")
    pts = []
    for w in wells.itertuples():
        f = root / "data" / "interim" / "poseidon_td" / f"{w.WELL_ID}.csv"
        if w.WELL_ID in held_out or not f.exists():
            continue
        td = pd.read_csv(f)
        if "twt_final_ms" not in td or td["twt_final_ms"].notna().sum() < 10:
            continue
        td = td.dropna(subset=["twt_final_ms"]).sort_values("tvdss")
        t = tops[(tops["WELL_ID"] == w.WELL_ID) & tops["SURFACE"].str.startswith(hcfg["surface_prefix"])]
        if t.empty:
            continue
        first = t.loc[t["TVDSS_M"].idxmin()]
        z = float(first["TVDSS_M"])
        if not td["tvdss"].min() <= z <= td["tvdss"].max():
            continue
        tt = float(np.interp(z, td["tvdss"], td["twt_final_ms"]))
        m = (td["twt_final_ms"] >= tt - 100) & (td["twt_final_ms"] <= tt + 200)
        v = 2000.0 * float(np.polyfit(td.loc[m, "twt_final_ms"], td.loc[m, "tvdss"], 1)[0])
        pts.append({"well": w.WELL_ID, "x": float(w.X_M), "y": float(w.Y_M), "twt_ms": round(tt, 1),
                    "tvdss_m": round(z, 1), "v_int_m_s": round(v, 0), "surface": first["SURFACE"]})
    if len(pts) < 3:
        raise ValueError(f"need at least 3 training wells with a '{hcfg['surface_prefix']}' pick and a tied "
                         f"time-depth curve, found {len(pts)}")
    return {"points": pts, "idw_power": float(hcfg.get("idw_power", 2.0)),
            "v_int_m_s": float(np.mean([p["v_int_m_s"] for p in pts])), "surface_prefix": hcfg["surface_prefix"]}


def eval_surface(surf: dict, x, y) -> tuple[np.ndarray, np.ndarray]:
    """Surface TWT (ms) and TVDSS (m) at x/y: least-squares plane + inverse-distance weighted residuals."""
    p = pd.DataFrame(surf["points"])
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    x0, y0 = p["x"].mean(), p["y"].mean()
    a = np.c_[np.ones(len(p)), (p["x"] - x0) / 1e3, (p["y"] - y0) / 1e3]
    out = []
    for key in ("twt_ms", "tvdss_m"):
        coef, *_ = np.linalg.lstsq(a, p[key].to_numpy(), rcond=None)
        pred = coef[0] + coef[1] * (x - x0) / 1e3 + coef[2] * (y - y0) / 1e3
        res = p[key].to_numpy() - a @ coef
        if len(p) > 3 and np.abs(res).max() > 1e-6:
            d = np.hypot(x[..., None] - p["x"].to_numpy(), y[..., None] - p["y"].to_numpy())
            wgt = 1.0 / np.maximum(d, 1.0) ** surf["idw_power"]
            pred = pred + (wgt * res).sum(-1) / wgt.sum(-1)
        out.append(pred)
    return out[0], out[1]


# ----------------------------------------------------------------------------------------- grid and traces
def well_nodes(geometry, root: Path) -> pd.DataFrame:
    """Nearest grid node (array indices) of every well inside the survey."""
    wells = pd.read_csv(root / "data" / "poseidon" / "dashboard" / "wells.csv")
    rows = []
    for w in wells.itertuples():
        d = np.hypot(geometry["cdp_x"] - w.X_M, geometry["cdp_y"] - w.Y_M)
        i, j = np.unravel_index(np.argmin(d), d.shape)
        if d[i, j] < 50:
            rows.append({"WELL_ID": w.WELL_ID, "i": int(i), "j": int(j), "X_M": w.X_M, "Y_M": w.Y_M,
                         "INLINE": int(geometry["ilines"][i]), "CROSSLINE": int(geometry["xlines"][j])})
    return pd.DataFrame(rows)


def aoi_indices(geometry, nodes: pd.DataFrame, gcfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Decimated array indices covering the bounding box of the wells plus a margin."""
    ni, nj = geometry["cdp_x"].shape
    m = gcfg["margin_traces"]
    i0, i1 = max(nodes["i"].min() - m, 0), min(nodes["i"].max() + m, ni - 1)
    j0, j1 = max(nodes["j"].min() - m, 0), min(nodes["j"].max() + m, nj - 1)
    return np.arange(i0, i1 + 1, gcfg["il_step"]), np.arange(j0, j1 + 1, gcfg["xl_step"])


def read_grid_traces(cache: Path, ii: np.ndarray, jj: np.ndarray, bundle: int, progress=None) -> np.ndarray:
    """Mean of bundle x bundle traces around every node of the ii x jj grid -> (K, ni, nj, ns) raw amplitudes."""
    h = bundle // 2
    out = None
    for k, s in enumerate(STACKS):
        g = np.load(cache / f"poseidon_{s}.geometry.npz")
        cube = np.load(cache / f"poseidon_{s}.f16.npy", mmap_mode="r")
        nx, ns = cube.shape[1], cube.shape[2]
        if out is None:
            out = np.empty((len(STACKS), len(ii), len(jj), ns), dtype=np.float32)
        lo, hi = np.clip(jj - h, 0, nx), np.clip(jj + h + 1, 0, nx)
        for a, i in enumerate(ii):
            blk = np.asarray(cube[max(i - h, 0):i + h + 1], dtype=np.float32).mean(axis=0)
            c = np.concatenate([np.zeros((1, ns), np.float32), np.cumsum(blk, axis=0)], axis=0)
            out[k, a] = (c[hi] - c[lo]) / (hi - lo)[:, None] * float(g["scale"])
            if progress and a % 8 == 0:
                progress((k + a / len(ii)) / len(STACKS), "reading seismic")
    return out


def window_index(t_surface_ms, samples_ms: np.ndarray, rel_ms: np.ndarray) -> np.ndarray:
    """First seismic sample of the window around each surface time (-1 if outside the data)."""
    k0 = np.round((np.asarray(t_surface_ms) + rel_ms[0] - samples_ms[0]) / DT_MS).astype(int)
    ok = (k0 >= 0) & (k0 + len(rel_ms) <= len(samples_ms))
    return np.where(ok, k0, -1)


# ----------------------------------------------------------------------------------------- library
@dataclass
class Library:
    z_top: float
    syn: np.ndarray          # (n, K, nt) calibrated synthetic units
    props: dict              # name -> (n, nt) float32
    facies: np.ndarray       # (n, nt) int8
    gas: np.ndarray          # (n, nt) float32 (0/1)
    phi_order: np.ndarray    # (n, nt) members sorted by porosity per sample
    phi_sorted: np.ndarray   # (n, nt)


def stack_setup(calib: dict):
    stacks = [StackDef(k, v["angle_min"], v["angle_max"]) for k, v in calib["stacks"].items()]
    wavelets = [fw.ricker(v["freq_hz"], DT_MODEL, 0.128, v["phase_deg"]) for v in calib["stacks"].values()]
    return stacks, wavelets


def build_library(prior: ps.Prior, cfg: dict, calib: dict, z_top: float, v_int: float, rel_ms: np.ndarray,
                  win_ms: tuple[float, float], n: int, seed: int) -> Library:
    """n pseudo-wells around a surface at depth z_top, forward-modelled on the relative time grid rel_ms."""
    stacks, wavelets = stack_setup(calib)
    t_out = T_REF + rel_ms / 1000
    t_model, r = fw.model_grid(t_out, DT_MODEL)
    z_of_t = lambda t: z_top + (t - T_REF) * v_int / 2  # noqa: E731
    t_start, t_stop = t_model[0] - 0.002, t_model[-1] + 0.002
    z_window = (z_of_t(T_REF + win_ms[0] / 1000), z_of_t(T_REF + win_ms[1] / 1000))
    rng = np.random.default_rng(seed)
    keys = ["facies", "gas"] + LIB_PROPS
    acc = {k: [] for k in keys + ["syn"]}
    bs = cfg["batch_size"]
    for b0 in range(0, n, bs):
        nb = min(bs, n - b0)
        lay = ps.simulate_layers(prior, cfg, nb, z_of_t(t_start), t_start, t_stop, z_window,
                                 np.random.default_rng(int(rng.integers(2 ** 31))))
        g = ps.layers_to_grid(lay, t_model, keys)
        acc["syn"].append(ps.forward_batch(g, stacks, wavelets, r, -1.0).astype(np.float32))
        for k in keys:
            acc[k].append(np.nan_to_num(g[k][:, r // 2::r]).astype(np.float32))
    a = {k: np.concatenate(v) for k, v in acc.items()}
    props = {k: a[k] for k in LIB_PROPS}
    props["ai"] = props["vp"] * props["rho"]
    order = np.argsort(props["phi"], axis=0, kind="stable")
    return Library(z_top, a["syn"], props, a["facies"].astype(np.int8), a["gas"], order,
                   np.take_along_axis(props["phi"], order, axis=0))


# ----------------------------------------------------------------------------------------- weighting
BETAS = np.r_[2.0 ** (-np.arange(0, 15) / 2), 0.0]     # tempering candidates, finest first


def likelihood_setup(cfg: dict, calib: dict, rel_ms: np.ndarray) -> dict:
    lcfg = cfg["likelihood"]
    sigma = np.array([v["noise_std_synthetic_units"] for v in calib["stacks"].values()]) * lcfg["noise_inflation"]
    f_dom = np.mean([v["freq_hz"] for v in calib["stacks"].values()])
    corr_ms = 1000 / (2 * f_dom) if lcfg["correlation_length_ms"] == "auto" else float(lcfg["correlation_length_ms"])
    return {"sigma": sigma.astype(np.float32),
            "scales": np.array([v["scale"] for v in calib["stacks"].values()]),
            "temper": 1.0 / max(corr_ms / DT_MS, 1.0),
            "half": int(round(lcfg["localization_half_window_ms"] / DT_MS)),
            "min_ess_target": lcfg.get("min_ess_target", 200)}


def weigh(lib: Library, obs: np.ndarray, score: np.ndarray, lk: dict, n_facies: int, u: np.ndarray | None = None,
          prior_only: bool = False) -> dict:
    """Localised importance weights of the library for a batch of traces obs (B, K, nt) and their summaries.
    score: (nt,) scored samples shared by all traces, or (B, nt) per trace."""
    n, nt = lib.facies.shape
    bsz = obs.shape[0]
    sb = np.broadcast_to(np.asarray(score, bool), (bsz, nt))
    mis = np.zeros((bsz, n, nt), np.float32)
    for k in range(obs.shape[1]):
        d = (lib.syn[None, :, k, :] - obs[:, None, k, :]) / lk["sigma"][k]
        mis += d * d
    mis *= sb[:, None, :]
    c = np.concatenate([np.zeros((bsz, n, 1), np.float32), np.cumsum(mis, axis=2)], axis=2)
    j = np.arange(nt)
    lo, hi = np.clip(j - lk["half"], 0, nt), np.clip(j + lk["half"] + 1, 0, nt)
    loc = -0.5 * lk["temper"] * (c[:, :, hi] - c[:, :, lo])
    loc -= loc.max(axis=1, keepdims=True)
    del c
    cols = np.flatnonzero(sb.any(axis=0))
    sbc = sb[:, cols]
    beta = np.zeros(bsz, np.float32)
    if not prior_only:
        target = min(lk["min_ess_target"], n / 10)
        chosen = np.zeros(bsz, bool)
        for bb in BETAS:
            w = np.exp(bb * loc[:, :, cols])
            ess = np.where(sbc, (w.sum(axis=1) ** 2) / (w * w).sum(axis=1), np.inf).min(axis=1)
            ok = ~chosen & (ess >= target)
            beta[ok], chosen = bb, chosen | ok
            if chosen.all():
                break
    w = np.exp(beta[:, None, None] * loc)
    w /= w.sum(axis=1, keepdims=True)
    del loc
    tt = np.arange(nt)
    ess_t = 1.0 / (w * w).sum(axis=1)
    out = {"beta": beta, "ess": np.where(sb, ess_t, np.inf).min(axis=1),
           "facies_prob": np.stack([(w * (lib.facies == f)).sum(axis=1) for f in range(n_facies)], axis=1),
           "p_gas": np.einsum("bnt,nt->bt", w, lib.gas)}
    for p in MEAN_PROPS:
        out[f"{p}_mean"] = np.einsum("bnt,nt->bt", w, lib.props[p])
    cum = np.cumsum(w[:, lib.phi_order, tt], axis=1)
    out["phi_q"] = np.stack([lib.phi_sorted[(cum < q).sum(axis=1).clip(0, n - 1), tt] for q in (0.1, 0.5, 0.9)],
                            axis=1)
    syn_mean = np.einsum("bnt,nkt->bkt", w, lib.syn)
    m = np.broadcast_to(sb[:, None, :], syn_mean.shape)
    cnt = np.maximum(m.sum(axis=(1, 2)), 1)
    a = np.where(m, syn_mean - (syn_mean * m).sum(axis=(1, 2), keepdims=True) / cnt[:, None, None], 0)
    b = np.where(m, obs - (obs * m).sum(axis=(1, 2), keepdims=True) / cnt[:, None, None], 0)
    out["fit_corr"] = (a * b).sum(axis=(1, 2)) / (np.sqrt((a * a).sum(axis=(1, 2)) * (b * b).sum(axis=(1, 2))) + 1e-12)
    if u is not None:
        k = (cum[None] < u[:, :, None, :]).sum(axis=2).clip(0, n - 1)        # (R, B, nt)
        member = lib.phi_order[k, tt]
        out["real_facies"] = lib.facies[member, tt]
        out["real_phi"] = lib.props["phi"][member, tt]
        out["real_gas"] = lib.gas[member, tt] > 0.5
    return out


# ----------------------------------------------------------------------------------------- driver
class Context:
    """Prior, surface, likelihood and libraries shared by grid runs and blind-well checks."""

    def __init__(self, root: Path, c3: dict | None = None, prior: ps.Prior | None = None,
                 win_ms: tuple[float, float] | None = None):
        self.root = root
        self.cfg, self.calib, petro = ps.load_configs(root)
        self.c3 = c3 or load_config(root)
        self.held_out = list(petro["held_out_wells"])
        self.prior = prior or ps.prepare_prior(root, self.cfg, self.calib, self.held_out)
        self.surface = top_surface(root, self.held_out, self.c3["horizon"])
        w = self.c3["window"]
        pad = self.cfg["pad_ms"]
        self.win_ms = win_ms or (-float(w["above_ms"]), float(w["below_ms"]))
        self.rel_ms = np.arange(self.win_ms[0] - pad, self.win_ms[1] + pad + 0.5 * DT_MS, DT_MS)
        self.score = (self.rel_ms >= self.win_ms[0]) & (self.rel_ms <= self.win_ms[1])
        self.lk = likelihood_setup(self.cfg, self.calib, self.rel_ms)
        self.libraries: dict[float, Library] = {}
        self.cache = root / "data" / "poseidon" / "seismic" / "cache"

    def band(self, z_top, rng=None):
        """Depth band (library) per trace. Traces between two band centres are assigned to the lower or upper one
        with linear probability, so the prior changes smoothly with depth instead of in steps."""
        b = float(self.c3["library"]["depth_band_m"])
        x = np.asarray(z_top, dtype=float) / b
        lo = np.floor(x)
        rng = rng or np.random.default_rng(int(self.c3["library"]["seed"]))
        return (lo + (rng.random(x.shape) < x - lo)) * b

    def library(self, z_band: float) -> Library:
        if z_band not in self.libraries:
            lc = self.c3["library"]
            self.libraries[z_band] = build_library(self.prior, self.cfg, self.calib, z_band,
                                                   self.surface["v_int_m_s"], self.rel_ms, self.win_ms, lc["n"],
                                                   int(lc["seed"] + z_band))
        return self.libraries[z_band]

    def run(self, obs: np.ndarray, z_top: np.ndarray, u: np.ndarray | None = None, prior_only: bool = False,
            progress=None, score: np.ndarray | None = None) -> dict:
        """Weigh P traces obs (P, K, nt) raw amplitudes; returns flat per-trace outputs.
        score: optional (P, nt) per-trace scored samples (default: the context window for all traces)."""
        obs_n = (obs / self.lk["scales"][None, :, None]).astype(np.float32)
        bands = self.band(z_top)
        nf = len(self.prior.facies)
        bsz = int(self.c3["trace_batch"])
        res: dict[str, np.ndarray] = {}
        done, total = 0, len(obs)
        for zb in np.unique(bands):
            if progress:
                progress(done / max(total, 1), f"building library for {zb:.0f} m")
            lib = self.library(float(zb))
            idx = np.flatnonzero(bands == zb)
            for s in range(0, len(idx), bsz):
                sel = idx[s:s + bsz]
                o = weigh(lib, obs_n[sel], self.score if score is None else score[sel], self.lk, nf,
                          None if u is None else u[:, sel], prior_only)
                for k, v in o.items():
                    if k not in res:
                        shape = (v.shape[0], total, *v.shape[2:]) if k.startswith("real_") else (total, *v.shape[1:])
                        res[k] = np.empty(shape, dtype=v.dtype)
                    if k.startswith("real_"):
                        res[k][:, sel] = v
                    else:
                        res[k][sel] = v
                done += len(sel)
                if progress:
                    progress(done / max(total, 1), f"weighting traces ({done}/{total})")
        res["band"] = bands
        res["obs_norm"] = obs_n
        return res


def pfield(shape: tuple[int, int, int], spacing_m: tuple[float, float], pcfg: dict) -> np.ndarray:
    """(R, *shape) uniform fields with a Gaussian-shaped spatial correlation."""
    rng = np.random.default_rng(pcfg["seed"])
    s = (pcfg["range_lateral_m"] / spacing_m[0] / 2, pcfg["range_lateral_m"] / spacing_m[1] / 2,
         pcfg["range_vertical_ms"] / DT_MS / 2)
    out = []
    for _ in range(pcfg["n_realizations"]):
        z = gaussian_filter(rng.standard_normal(shape), s, mode="reflect")
        out.append(ndtr((z - z.mean()) / z.std()).astype(np.float32))
    return np.stack(out)


def run_grid(root: Path, name: str = "grid3d", c3: dict | None = None, prior: ps.Prior | None = None,
             progress=None, ctx: Context | None = None) -> dict:
    """Simulate over the decimated grid and save outputs/simulation3d/<name>.npz/.json."""
    t0 = time.time()
    ctx = ctx or Context(root, c3, prior)
    gcfg = ctx.c3["grid"]
    geometry = np.load(ctx.cache / "poseidon_near.geometry.npz")
    nodes = well_nodes(geometry, root)
    ii, jj = aoi_indices(geometry, nodes, gcfg)
    gx, gy = geometry["cdp_x"][np.ix_(ii, jj)], geometry["cdp_y"][np.ix_(ii, jj)]
    t_s, z_s = eval_surface(ctx.surface, gx, gy)
    sub = (lambda f, m: progress(0.15 * f, m)) if progress else None
    traces = read_grid_traces(ctx.cache, ii, jj, gcfg["bundle"], sub)
    samples = geometry["samples"]
    k0 = window_index(t_s, samples, ctx.rel_ms)
    valid = k0 >= 0
    nt = len(ctx.rel_ms)
    vi, vj = np.nonzero(valid)
    obs = np.stack([traces[:, a, b, k0[a, b]:k0[a, b] + nt] for a, b in zip(vi, vj)])
    del traces
    t_snap = np.where(valid, samples[0] + k0 * DT_MS - ctx.rel_ms[0], np.nan)
    di = float(np.median(np.hypot(np.diff(gx, axis=0), np.diff(gy, axis=0))))
    dj = float(np.median(np.hypot(np.diff(gx, axis=1), np.diff(gy, axis=1))))
    u_grid = pfield((len(ii), len(jj), nt), (di, dj), ctx.c3["pfield"])
    u = u_grid[:, vi, vj]
    sub = (lambda f, m: progress(0.15 + 0.85 * f, m)) if progress else None
    res = ctx.run(obs, z_s[vi, vj], u, progress=sub)

    shape = (len(ii), len(jj))
    arrays = {"ilines": geometry["ilines"][ii], "xlines": geometry["xlines"][jj], "x": gx, "y": gy,
              "rel_ms": ctx.rel_ms, "score_mask": ctx.score, "surface_twt_ms": t_snap,
              "surface_tvdss_m": np.where(valid, z_s, np.nan), "valid": valid,
              "facies_names": np.array(ctx.prior.facies)}

    def put(v, dtype, real=False):
        fill = -1 if dtype == np.int8 else np.nan
        if real:   # realizations: (R, P, nt)
            full = np.full((v.shape[0], *shape, *v.shape[2:]), fill, dtype=dtype)
            full[:, vi, vj] = v
            return full
        full = np.full((*shape, *v.shape[1:]), fill, dtype=dtype)
        full[vi, vj] = v
        return full

    for k in ("beta", "ess", "fit_corr", "band"):
        arrays[k] = put(res[k], np.float32)
    for k in ["p_gas", "phi_q", "facies_prob"] + [f"{p}_mean" for p in MEAN_PROPS]:
        arrays[k] = put(res[k], np.float16)
    arrays["real_facies"] = put(res["real_facies"], np.int8, real=True)
    arrays["real_phi"] = put(res["real_phi"], np.float16, real=True)
    arrays["real_gas"] = put(res["real_gas"].astype(np.int8), np.int8, real=True)
    out = root / "outputs" / "simulation3d"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{name}.npz", **arrays)

    min_beta = ctx.c3["support"]["min_beta"]
    meta = {"name": name, "label": "SIMULATED 3D pseudo-well grid (not measured data)",
            "method": "shared library per depth band + localised importance weighting per trace; P-field "
                      "realizations",
            "n_traces": int(valid.sum()), "grid_shape": list(shape),
            "il_step": gcfg["il_step"], "xl_step": gcfg["xl_step"], "node_spacing_m": [round(di, 1), round(dj, 1)],
            "inline_range": [int(arrays["ilines"][0]), int(arrays["ilines"][-1])],
            "crossline_range": [int(arrays["xlines"][0]), int(arrays["xlines"][-1])],
            "window_rel_ms": list(ctx.win_ms), "library_n_per_band": ctx.c3["library"]["n"],
            "depth_bands_m": sorted(float(b) for b in ctx.libraries),
            "surface": ctx.surface, "training_wells": ctx.prior.wells, "held_out_wells_excluded": ctx.held_out,
            "beta_median": round(float(np.median(res["beta"])), 3),
            "low_support_fraction": round(float((res["beta"] < min_beta).mean()), 3),
            "ess_min_median": round(float(np.median(res["ess"])), 1),
            "fit_corr_median": round(float(np.median(res["fit_corr"])), 3),
            "runtime_s": round(time.time() - t0, 1), "config": ctx.c3}
    (out / f"{name}.json").write_text(json.dumps(meta, indent=2, default=float), encoding="utf-8")
    return meta


# ----------------------------------------------------------------------------------------- zone along a line
def zone_twt_from_logs(root: Path, well: str, zone: str) -> tuple[float, float] | None:
    """TWT (ms) of the first and last log sample inside the zone (from the supplied tops) in a well."""
    f = root / "data" / "interim" / "poseidon_logs" / f"{well}.parquet"
    if not f.exists():
        return None
    logs = pd.read_parquet(f, columns=["formation", "twt"])
    t = logs.loc[logs["formation"].astype(str).str.startswith(zone), "twt"].dropna()
    return (float(t.min()), float(t.max())) if len(t) else None


def read_point_traces(cache: Path, i: np.ndarray, j: np.ndarray, bundle: int) -> np.ndarray:
    """Mean of bundle x bundle traces around each (i, j) -> (P, K, ns) raw amplitudes."""
    h = bundle // 2
    out = []
    for s in STACKS:
        g = np.load(cache / f"poseidon_{s}.geometry.npz")
        cube = np.load(cache / f"poseidon_{s}.f16.npy", mmap_mode="r")
        tr = np.stack([np.asarray(cube[max(a - h, 0):a + h + 1, max(b - h, 0):b + h + 1], np.float32)
                       .reshape(-1, cube.shape[2]).mean(axis=0) for a, b in zip(i, j)])
        out.append(tr * float(g["scale"]))
    return np.stack(out, axis=1)


def run_line(root: Path, name: str, i_idx: np.ndarray, j_idx: np.ndarray, distance_m: np.ndarray,
             top_ms: np.ndarray, base_ms: np.ndarray, *, step: int = 4, n: int | None = None,
             prior: ps.Prior | None = None, progress=None, extra_meta: dict | None = None) -> dict:
    """Simulate every step-th trace along a route inside its zone [top_ms, base_ms] (interpreted, per trace).

    One library per depth band (zone top depth from the time-depth of the nearest well in the prior), scored only
    inside each trace's zone. Saves outputs/simulation_lines/<name>.npz/.json."""
    t0 = time.time()
    sel = np.arange(0, len(i_idx), max(int(step), 1))
    if sel[-1] != len(i_idx) - 1:
        sel = np.r_[sel, len(i_idx) - 1]
    i_idx, j_idx, dist = np.asarray(i_idx)[sel], np.asarray(j_idx)[sel], np.asarray(distance_m)[sel]
    top, base = np.asarray(top_ms, float)[sel], np.asarray(base_ms, float)[sel]
    if np.any(base - top < 8):
        raise ValueError("the zone must be at least 8 ms thick everywhere along the line")
    c3 = load_config(root)
    if n:
        c3["library"]["n"] = int(n)
    ctx = Context(root, c3, prior, win_ms=(0.0, float(np.ceil((base - top).max() / DT_MS) * DT_MS)))
    geometry = np.load(ctx.cache / "poseidon_near.geometry.npz")
    samples = geometry["samples"]
    x, y = geometry["cdp_x"][i_idx, j_idx], geometry["cdp_y"][i_idx, j_idx]

    # zone-top depth from the time-depth of the nearest well that is in the prior
    wells = pd.read_csv(root / "data" / "poseidon" / "dashboard" / "wells.csv").set_index("WELL_ID")
    tds = {}
    for w in ctx.prior.wells:
        f = root / "data" / "interim" / "poseidon_td" / f"{w}.csv"
        if f.exists():
            td = pd.read_csv(f).dropna(subset=["twt_final_ms"]).sort_values("twt_final_ms")
            if len(td) >= 10:
                tds[w] = (td["twt_final_ms"].to_numpy(), td["tvdss"].to_numpy())
    if not tds:
        raise ValueError("no well in the prior has a tied time-depth curve")
    names = list(tds)
    d = np.hypot(x[:, None] - wells.loc[names, "X_M"].to_numpy(), y[:, None] - wells.loc[names, "Y_M"].to_numpy())
    near = d.argmin(axis=1)
    z_top = np.array([np.interp(t, *tds[names[k]]) for t, k in zip(top, near)])

    if progress:
        progress(0.02, "reading seismic")
    traces = read_point_traces(ctx.cache, i_idx, j_idx, ctx.c3["grid"]["bundle"])
    nt = len(ctx.rel_ms)
    k0 = np.round((top + ctx.rel_ms[0] - samples[0]) / DT_MS).astype(int)
    if np.any(k0 < 0) or np.any(k0 + nt > len(samples)):
        raise ValueError("the zone (plus the modelling pad) must lie inside the seismic time range")
    obs = np.stack([traces[p, :, k0[p]:k0[p] + nt] for p in range(len(sel))])
    top_snap = samples[0] + k0 * DT_MS - ctx.rel_ms[0]
    thick = base - top_snap
    score = (ctx.rel_ms[None, :] >= 0) & (ctx.rel_ms[None, :] <= thick[:, None])
    res = ctx.run(obs, z_top, score=score,
                  progress=(lambda f, m: progress(0.05 + 0.9 * f, m)) if progress else None)
    # prior (wells only) per depth band, for the "what the seismic adds" view
    prior_fp = np.empty_like(res["facies_prob"])
    for zb in np.unique(res["band"]):
        m = res["band"] == zb
        pr = weigh(ctx.library(float(zb)), obs[:1] * 0, ctx.score, ctx.lk, len(ctx.prior.facies), prior_only=True)
        prior_fp[m] = pr["facies_prob"][0]

    arrays = {"distance_m": dist, "i": i_idx, "j": j_idx, "inline": geometry["ilines"][i_idx],
              "crossline": geometry["xlines"][j_idx], "x": x, "y": y, "top_ms": top_snap, "base_ms": base,
              "z_top_m": z_top, "rel_ms": ctx.rel_ms, "score": score, "facies_names": np.array(ctx.prior.facies),
              "beta": res["beta"], "ess": res["ess"], "fit_corr": res["fit_corr"], "band": res["band"],
              "facies_prob": res["facies_prob"].astype(np.float16), "prior_facies_prob": prior_fp.astype(np.float16),
              "p_gas": res["p_gas"].astype(np.float16), "phi_q": res["phi_q"].astype(np.float16),
              **{f"{p}_mean": res[f"{p}_mean"].astype(np.float16) for p in MEAN_PROPS}}
    out = root / "outputs" / "simulation_lines"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{name}.npz", **arrays)
    min_beta = ctx.c3["support"]["min_beta"]
    meta = {"name": name, "label": "SIMULATED pseudo-wells along a line (not measured data)",
            "n_traces": int(len(sel)), "trace_step": int(step), "length_km": round(float(dist[-1]) / 1000, 2),
            "library_n_per_band": ctx.c3["library"]["n"], "depth_bands_m": sorted(float(b) for b in ctx.libraries),
            "zone_formation": ctx.cfg["training"].get("zone_formation_prefix"),
            "training_wells": ctx.prior.wells, "held_out_wells_excluded": ctx.held_out,
            "time_depth_wells": names, "beta_median": round(float(np.median(res["beta"])), 3),
            "low_support_fraction": round(float((res["beta"] < min_beta).mean()), 3),
            "fit_corr_median": round(float(np.median(res["fit_corr"])), 3),
            "runtime_s": round(time.time() - t0, 1), **(extra_meta or {})}
    (out / f"{name}.json").write_text(json.dumps(meta, indent=2, default=float), encoding="utf-8")
    if progress:
        progress(1.0, "done")
    return meta


# ----------------------------------------------------------------------------------------- blind wells
def blind_check(ctx: Context, well: str) -> tuple[pd.DataFrame, dict]:
    """Run the 3D engine at a (held-out) well location and score prior/posterior against its actual logs.

    The well's logs and tops are used for scoring only."""
    root = ctx.root
    geometry = np.load(ctx.cache / "poseidon_near.geometry.npz")
    node = well_nodes(geometry, root).set_index("WELL_ID").loc[well]
    t_s, z_s = eval_surface(ctx.surface, node["X_M"], node["Y_M"])
    t_seis, obs = well_traces(ctx.cache, STACKS, int(node["INLINE"]), int(node["CROSSLINE"]), ctx.c3["grid"]["bundle"])
    k0 = int(window_index(np.array([t_s]), t_seis * 1000, ctx.rel_ms)[0])
    nt = len(ctx.rel_ms)
    o = obs[None, :, k0:k0 + nt]
    t_abs = t_seis[k0] * 1000 + np.arange(nt) * DT_MS
    res = {"prior": ctx.run(o, np.array([z_s]), prior_only=True), "post": ctx.run(o, np.array([z_s]))}

    logs = pd.read_parquet(root / "data" / "interim" / "poseidon_logs" / f"{well}.parquet")
    sc = np.flatnonzero(ctx.score)
    logs = logs[(logs["twt"] >= t_abs[sc[0]] - DT_MS / 2) & (logs["twt"] < t_abs[sc[-1]] + DT_MS / 2)]
    k = np.round((logs["twt"].to_numpy() - t_abs[0]) / DT_MS).astype(int)
    g = logs.groupby(k)
    idx = g.size().index.to_numpy()
    lith = g["lithotype"].agg(lambda s: s.mode().iat[0] if len(s.mode()) else "unknown")
    facies = ctx.prior.facies
    rows = []
    for name, r in res.items():
        for prop, tol in (("phi", 0.003),):
            a = g[prop].mean().to_numpy()
            q = r["phi_q"][0][:, idx]
            ok = np.isfinite(a)
            rows.append({"well": well, "ensemble": name, "property": "phi", "n_samples": int(ok.sum()),
                         "coverage_p10_p90": round(float(((a >= q[0] - tol) & (a <= q[2] + tol))[ok].mean()), 3),
                         "mae_p50": round(float(np.abs(a - q[1])[ok].mean()), 4)})
        for prop in ("vp", "ai"):
            a = (g["vp"].mean() * g["rho"].mean()).to_numpy() if prop == "ai" else g[prop].mean().to_numpy()
            m = r[f"{prop}_mean"][0][idx]
            ok = np.isfinite(a)
            rows.append({"well": well, "ensemble": name, "property": prop, "n_samples": int(ok.sum()),
                         "mae_mean": round(float(np.abs(a - m)[ok].mean()), 1)})
        prob = r["facies_prob"][0][:, idx]
        known = lith.isin(facies).to_numpy()
        true_i = np.array([facies.index(x) if x in facies else 0 for x in lith])
        pred = np.array(facies)[prob.argmax(axis=0)]
        rows.append({"well": well, "ensemble": name, "property": "facies", "n_samples": int(known.sum()),
                     "accuracy_map": round(float((pred == lith.to_numpy())[known].mean()), 3),
                     "mean_p_true_facies": round(float(prob[true_i, np.arange(len(true_i))][known].mean()), 3)})
    tops = pd.read_csv(root / "data" / "poseidon" / "dashboard" / "well_tops.csv")
    t = tops[(tops["WELL_ID"] == well) & tops["SURFACE"].str.startswith(ctx.surface["surface_prefix"])]
    info = {"well": well, "surface_pred_twt_ms": round(float(t_s), 1), "surface_pred_tvdss_m": round(float(z_s), 1),
            "beta": float(res["post"]["beta"][0]), "ess_min": round(float(res["post"]["ess"][0]), 1),
            "fit_corr": round(float(res["post"]["fit_corr"][0]), 3),
            "window_twt_ms": [float(t_abs[sc[0]]), float(t_abs[sc[-1]])]}
    if len(t):
        z_act = float(t["TVDSS_M"].min())
        info["surface_actual_tvdss_m"] = round(z_act, 1)
        td = pd.read_csv(root / "data" / "interim" / "poseidon_td" / f"{well}.csv").dropna(subset=["twt_final_ms"])
        td = td.sort_values("tvdss")
        info["surface_actual_twt_ms"] = round(float(np.interp(z_act, td["tvdss"], td["twt_final_ms"])), 1)
    return pd.DataFrame(rows), info
