"""End-to-end workflow: calibration, gates, priors, libraries, models A/B/C, blind wells and mapping."""

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from avo import class_names, collapse_fluid, joint_class
from avo import forward as fw
from avo.calibration import calibrate
from avo.features import ig_operator, intercept_gradient, window_features
from avo.gates import gate_g2, gate_g3
from avo.inference import entropy, predict_ig, predict_window, train_ig_baseline, train_window_model
from avo.library import build_library
from avo.priors import sample_prior_b, sample_prior_c

METHODS = ["A", "B", "C"]
METHOD_LABELS = {"A": "A: intercept-gradient", "B": "B: hand-picked prior", "C": "C: learned prior"}


def observed_scaled(cube, well, calib):
    return calib.scale_observed(cube.trace(well.il, well.xl), cube.stack_defs)


def well_labels(well, cube, calib, use_fluid):
    l = well.logs
    lab = fw.labels_on_grid(well.twt(), joint_class(l["litho"], l["fluid"]), cube.t, calib.dt_model, 4,
                            calib.shifts.get(well.name, 0.0))
    return lab if use_fluid else np.where(lab >= 0, collapse_fluid(lab), -1)


def train_bundle(train_wells, cube, cfg, use_fluid, seed, calib=None):
    """Calibrate on the training wells only and fit methods A, B and C."""
    rng = np.random.default_rng(seed)
    calib = calib or calibrate(train_wells, cube, cfg)
    n_classes = len(class_names(use_fluid))
    ig_op = ig_operator(cube.stack_defs)
    n_pw = cfg["library"]["n_pseudo_wells"]

    libs = {
        "B": build_library(sample_prior_b(n_pw, cfg, train_wells[0].dz, rng), calib, cube.stack_defs, cube.dt,
                           cfg, rng, ig_op, use_fluid),
        "C": build_library(sample_prior_c(n_pw, train_wells, cfg, rng), calib, cube.stack_defs, cube.dt,
                           cfg, rng, ig_op, use_fluid),
    }
    models = {m: train_window_model(*libs[m], n_classes, cfg, seed) for m in ("B", "C")}

    I, G, y = [], [], []
    for w in train_wells:
        lab = well_labels(w, cube, calib, use_fluid)
        ok = lab >= 0
        ig = intercept_gradient(observed_scaled(cube, w, calib), ig_op)
        I.append(ig[0, ok]), G.append(ig[1, ok]), y.append(lab[ok])
    models["A"] = train_ig_baseline(np.concatenate(I), np.concatenate(G), np.concatenate(y), n_classes)
    return dict(calib=calib, models=models, libs=libs, ig_op=ig_op, use_fluid=use_fluid, n_classes=n_classes,
                half=cfg["library"]["window_half"], train_wells=[w.name for w in train_wells])


def predict_all(bundle, traces, centers=None):
    out = {m: predict_window(bundle["models"][m], traces, bundle["half"], bundle["ig_op"], centers) for m in ("B", "C")}
    out["A"] = predict_ig(bundle["models"]["A"], traces, bundle["ig_op"], centers)
    return out


def hugin_zone(well, cube, calib, pad=0.02):
    t = np.interp([well.hugin_top, well.hugin_base], well.logs["tvdss"], well.twt()) + calib.shifts.get(well.name, 0.0)
    return (cube.t >= t[0] - pad) & (cube.t <= t[1] + pad)


def _scores(y, p, n_classes, use_fluid):
    present = np.unique(y)
    pred = p.argmax(1)
    q = np.clip(p, 1e-6, 1)
    q = q / q.sum(1, keepdims=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        row = dict(n=len(y), balanced_acc=balanced_accuracy_score(y, pred),
                   macro_f1=f1_score(y, pred, labels=present, average="macro", zero_division=0),
                   log_loss=log_loss(y, q, labels=list(range(n_classes))))
    if use_fluid:
        row["oil_sand_f1"] = f1_score(y == 3, pred == 3, zero_division=0)
    return row


def run_validation(wells, cube, cfg, use_fluid, seed):
    """Leave-one-well-out: everything (calibration, prior C, models) is refit without the blind well."""
    rows, pooled, per_well = [], {m: ([], []) for m in METHODS}, {}
    n_classes = len(class_names(use_fluid))
    for i, blind in enumerate(wells):
        train = [w for j, w in enumerate(wells) if j != i]
        b = train_bundle(train, cube, cfg, use_fluid, seed + i)
        traces = observed_scaled(cube, blind, b["calib"])
        lab = well_labels(blind, cube, b["calib"], use_fluid)
        probs = predict_all(b, traces)
        zones = {"window": lab >= 0, "hugin": (lab >= 0) & hugin_zone(blind, cube, b["calib"])}
        for m in METHODS:
            for zname, z in zones.items():
                rows.append(dict(blind_well=blind.name, method=m, zone=zname,
                                 **_scores(lab[z], probs[m][z], n_classes, use_fluid)))
            pooled[m][0].append(lab[zones["hugin"]]), pooled[m][1].append(probs[m][zones["hugin"]])
        per_well[blind.name] = dict(labels=lab, probs=probs, traces=traces, train_wells=b["train_wells"])
    scores = pd.DataFrame(rows)
    pooled = {m: (np.concatenate(v[0]), np.concatenate(v[1])) for m, v in pooled.items()}
    return scores, pooled, per_well


def reliability(y, p, bins=5):
    conf, pred = p.max(1), p.argmax(1)
    edges = np.linspace(1.0 / p.shape[1], 1.0, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf <= hi)
        if m.any():
            rows.append(dict(conf_lo=lo, conf_hi=hi, n=int(m.sum()), mean_conf=float(conf[m].mean()),
                             accuracy=float((pred[m] == y[m]).mean())))
    return pd.DataFrame(rows)


def prediction_mask(cube, cfg):
    """Samples to predict: export window around the interpreted target horizon (everything if no horizon)."""
    n_il, n_xl = cube.shape
    if cube.horizon_twt is None:
        return np.ones((n_il, n_xl, cube.nt), bool)
    w0, w1 = cfg["export"]["window_ms"]
    h = cube.horizon_twt[..., None]
    return (cube.t >= h + w0 / 1000) & (cube.t <= h + w1 / 1000)


def map_cube(bundle, cube, cfg):
    """Predict all methods inside the prediction window (NaN outside), in trace batches, and derive maps."""
    n_il, n_xl = cube.shape
    nt, n_tr, n_cls = cube.nt, n_il * n_xl, bundle["n_classes"]
    calib, half, ig_op = bundle["calib"], bundle["half"], bundle["ig_op"]
    mask = prediction_mask(cube, cfg).reshape(n_tr, nt)
    if cube.horizon_twt is not None:
        w0, w1 = cfg["mapping"]["window_ms"]
        hz = cube.horizon_twt.reshape(-1, 1)
        in_win = mask & (cube.t[None, :] >= hz + w0 / 1000) & (cube.t[None, :] <= hz + w1 / 1000)
    else:
        in_win = mask.copy()
    counts = in_win.sum(1)

    rng = np.random.default_rng(cfg["seed"])
    X_lib = bundle["libs"]["C"][0]
    sub = X_lib[rng.choice(len(X_lib), min(len(X_lib), cfg["mapping"]["ood_library_subsample"]), replace=False)]
    scaler = StandardScaler().fit(sub)
    # Local Outlier Factor: ~1 where observed windows are as densely covered as nearby library windows.
    lof = LocalOutlierFactor(n_neighbors=cfg["mapping"]["ood_neighbors"], novelty=True).fit(scaler.transform(sub))

    probs = {m: np.full((n_tr, nt, n_cls), np.nan, np.float32) for m in METHODS}
    d = np.full((n_tr, nt), np.nan)
    chunk = cfg["export"]["chunk_traces"]
    for start in range(0, n_tr, chunk):
        X, IG, tr_i, s_i = [], [], [], []
        for t in range(start, min(start + chunk, n_tr)):
            centers = np.flatnonzero(mask[t])
            if centers.size == 0:
                continue
            tr = calib.scale_observed(cube.trace(*divmod(t, n_xl)), cube.stack_defs)
            X.append(window_features(tr, centers, half, ig_op))
            IG.append(intercept_gradient(tr, ig_op)[:, centers].T)
            tr_i.append(np.full(centers.size, t))
            s_i.append(centers)
        if not X:
            continue
        X, IG, tr_i, s_i = np.vstack(X), np.vstack(IG), np.concatenate(tr_i), np.concatenate(s_i)
        for m in ("B", "C"):
            probs[m][tr_i, s_i] = bundle["models"][m].predict_proba(X)
        probs["A"][tr_i, s_i] = bundle["models"]["A"].predict_proba(IG)
        sel = in_win[tr_i, s_i]
        if sel.any():
            d[tr_i[sel], s_i[sel]] = -lof.score_samples(scaler.transform(X[sel]))

    def win_mean(a):
        """Mean over the map window per trace for a (n_tr, nt[, k]) array."""
        m = in_win if a.ndim == 2 else in_win[..., None]
        return np.where(m, a, 0).sum(1) / np.maximum(counts, 1).reshape((-1,) + (1,) * (a.ndim - 2))

    maps = {}
    sand_cls = [2, 3] if bundle["use_fluid"] else [2]
    for m in METHODS:
        p = probs[m]
        pw = win_mean(p)
        maps[f"p_sand_{m}"] = pw[:, sand_cls].sum(1).reshape(n_il, n_xl)
        if bundle["use_fluid"]:
            maps[f"p_oil_{m}"] = pw[:, 3].reshape(n_il, n_xl)
        maps[f"entropy_{m}"] = win_mean(entropy(np.nan_to_num(p))).reshape(n_il, n_xl)
    maps["ood"] = win_mean(d).reshape(n_il, n_xl)

    truth_eval = None
    if cube.truth_class is not None:
        truth = cube.truth_class if bundle["use_fluid"] else np.where(cube.truth_class >= 0,
                                                                      collapse_fluid(cube.truth_class), -1)
        tf = truth.reshape(n_tr, nt)
        if bundle["use_fluid"]:
            maps["truth_oil"] = win_mean((tf == 3).astype(float)).reshape(n_il, n_xl)
        maps["truth_sand"] = win_mean(np.isin(tf, sand_cls).astype(float)).reshape(n_il, n_xl)
        ok = in_win & (tf >= 0)
        truth_eval = pd.DataFrame([dict(method=m, **_scores(tf[ok], probs[m][ok].astype(float),
                                                            bundle["n_classes"], bundle["use_fluid"]))
                                   for m in METHODS])
    probs = {m: p.reshape(n_il, n_xl, nt, n_cls) for m, p in probs.items()}
    return maps, probs, truth_eval


def run_pipeline(wells, cube, cfg):
    """Run gates, blind-well validation and final mapping. Returns a results dict for plotting/reporting."""
    seed = cfg["seed"]
    decisions = []
    n_vs = sum((w.logs["vs_source"] == "measured").all() for w in wells)
    g1 = len(wells) >= 3
    decisions.append(dict(gate="G1", passed=g1, detail=f"{len(wells)} wells ({n_vs} with measured Vs), "
                                                       f"{len(cube.stack_defs)} partial stacks"))
    calib_all = calibrate(wells, cube, cfg)
    g2, corr = gate_g2(calib_all, cfg)
    decisions.append(dict(gate="G2", passed=g2, detail="mean tie correlation per stack: "
                          + ", ".join(f"{k}={v:.2f}" for k, v in corr.items())))
    g3, ratio, g3_table, g3_panels = gate_g3(wells, cube, calib_all, cfg)
    decisions.append(dict(gate="G3", passed=g3, detail=f"median (over wells) of best-stack brine-vs-oil "
                          f"difference / noise = {ratio:.2f} (threshold {cfg['gate_g3']['min_ratio']}). "
                          + ("Fluid kept as a target." if g3 else "Fluid dropped; lithology only.")))
    use_fluid = g3

    scores, pooled, per_well = run_validation(wells, cube, cfg, use_fluid, seed)
    hug = scores[scores.zone == "hugin"].groupby("method")["balanced_acc"].mean()
    best = hug.idxmax()
    g4 = bool(max(hug.get("B", 0), hug.get("C", 0)) > hug.get("A", 0))
    decisions.append(dict(gate="G4", passed=g4, detail="mean Hugin-zone balanced accuracy at blind wells: "
                          + ", ".join(f"{m}={hug[m]:.2f}" for m in METHODS) + f" (n_wells={len(wells)})"))

    final = train_bundle(wells, cube, cfg, use_fluid, seed + 100, calib=calib_all)
    maps, probs, truth_eval = map_cube(final, cube, cfg)
    rel = {m: reliability(*pooled[m]) for m in METHODS}
    return dict(decisions=pd.DataFrame(decisions), calib=calib_all, g3_table=g3_table, g3_panels=g3_panels,
                use_fluid=use_fluid, scores=scores, pooled=pooled, per_well=per_well, reliability=rel,
                best_method=best, final=final, maps=maps, probs=probs, truth_eval=truth_eval)
