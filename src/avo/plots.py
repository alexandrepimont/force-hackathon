"""Figures for the workflow outputs."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from avo import class_names  # noqa: E402
from avo import forward as fw  # noqa: E402
from avo.pipeline import METHOD_LABELS, METHODS, observed_scaled  # noqa: E402

CLASS_COLORS = {"shale": "#7f7f7f", "tight": "#1f77b4", "brine_sand": "#ffdd57", "oil_sand": "#2ca02c",
                "sand": "#ffdd57"}


def _class_cmap(use_fluid):
    return ListedColormap(["white"] + [CLASS_COLORS[c] for c in class_names(use_fluid)])


def well_tie(well, cube, calib, path):
    l = well.logs
    syn = fw.synthesize(well.twt(), l["vp"].to_numpy(), l["vs"].to_numpy(), l["rho"].to_numpy(), cube.stack_defs,
                        calib.wavelet_list(cube.stack_defs), cube.t, calib.dt_model, calib.shifts.get(well.name, 0.0))
    obs = observed_scaled(cube, well, calib)
    fig, axes = plt.subplots(1, len(cube.stack_defs), figsize=(9, 6), sharey=True)
    for k, (ax, s) in enumerate(zip(axes, cube.stack_defs)):
        ax.plot(obs[k], cube.t, "k", lw=1, label="observed / scale")
        ax.plot(syn[k], cube.t, "r", lw=1, label="synthetic")
        corr = calib.ties.query("well == @well.name and stack == @s.name")["corr"]
        ax.set_title(f"{s.name} ({s.angle_min}-{s.angle_max} deg)\ncorr={corr.iloc[0]:.2f}" if len(corr) else s.name)
        ax.set_xlabel("amplitude")
    axes[0].set_ylabel("TWT (s)")
    axes[0].invert_yaxis()
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle(f"Well tie {well.name}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fluid_sensitivity(panels, cube, path):
    name = next(iter(panels))
    p = panels[name]
    fig, axes = plt.subplots(1, len(cube.stack_defs), figsize=(9, 6), sharey=True)
    for k, (ax, s) in enumerate(zip(axes, cube.stack_defs)):
        ax.plot(p["syn"]["brine"][k], cube.t, color="#1f77b4", label="Hugin 100% brine")
        ax.plot(p["syn"]["oil"][k], cube.t, color="#2ca02c", label="Hugin oil")
        ax.axhspan(*p["twt_range"], color="0.9", zorder=0)
        ax.set_title(s.name)
    axes[0].invert_yaxis()
    axes[0].set_ylabel("TWT (s)")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle(f"Gate G3: Gassmann brine vs oil synthetic at {name} (grey = Hugin)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def ig_crossplot(bundle, wells, cube, path, max_points=4000):
    """Intercept-gradient of the learned-prior library versus observed data at the wells."""
    from avo.features import intercept_gradient
    from avo.pipeline import well_labels
    half, names = bundle["half"], class_names(bundle["use_fluid"])
    X, y = bundle["libs"]["C"]
    n_amp = len(cube.stack_defs) * (2 * half + 1)
    I, G = X[:, n_amp + half], X[:, n_amp + (2 * half + 1) + half]
    rng = np.random.default_rng(0)
    sel = rng.choice(len(y), min(max_points, len(y)), replace=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    for c, n in enumerate(names):
        m = y[sel] == c
        axes[0].scatter(I[sel][m], G[sel][m], s=3, alpha=0.4, color=CLASS_COLORS[n], label=n)
    for w in wells:
        lab = well_labels(w, cube, bundle["calib"], bundle["use_fluid"])
        ig = intercept_gradient(observed_scaled(cube, w, bundle["calib"]), bundle["ig_op"])
        for c, n in enumerate(names):
            m = lab == c
            axes[1].scatter(ig[0, m], ig[1, m], s=8, alpha=0.7, color=CLASS_COLORS[n])
    axes[0].set_title("Synthetic library (prior C)")
    axes[1].set_title("Observed at wells (calibrated)")
    for ax in axes:
        ax.set_xlabel("intercept")
        ax.axhline(0, color="k", lw=0.5)
        ax.axvline(0, color="k", lw=0.5)
    axes[0].set_ylabel("gradient")
    axes[0].legend(markerscale=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def blind_well(name, pw, cube, use_fluid, path):
    names = class_names(use_fluid)
    cmap = _class_cmap(use_fluid)
    tracks = [("truth", pw["labels"])] + [(m, pw["probs"][m].argmax(1)) for m in METHODS]
    valid = pw["labels"] >= 0
    fig, axes = plt.subplots(1, 3 + len(tracks), figsize=(11, 6), sharey=True,
                             gridspec_kw={"width_ratios": [2, 2, 2] + [0.6] * len(tracks)})
    for k, s in enumerate(cube.stack_defs):
        axes[k].plot(pw["traces"][k], cube.t, "k", lw=1)
        axes[k].set_title(s.name)
    for ax, (title, lab) in zip(axes[3:], tracks):
        img = np.where(valid, lab + 1, 0)[:, None]
        ax.imshow(img, aspect="auto", cmap=cmap, vmin=0, vmax=len(names),
                  extent=[0, 1, cube.t[-1] + cube.dt / 2, cube.t[0] - cube.dt / 2])
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
    axes[0].set_ylim(cube.t[valid].max() + 0.02, cube.t[valid].min() - 0.02)
    axes[0].set_ylabel("TWT (s)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=CLASS_COLORS[n]) for n in names]
    fig.legend(handles, names, loc="lower center", ncol=len(names), fontsize=8)
    fig.suptitle(f"Blind well {name} (trained without it on: {', '.join(pw['train_wells'])})")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def reliability_plot(rel, path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfectly calibrated")
    for m in METHODS:
        r = rel[m]
        ax.plot(r["mean_conf"], r["accuracy"], "o-", label=METHOD_LABELS[m])
    ax.set_xlabel("predicted confidence (max probability)")
    ax.set_ylabel("observed accuracy at blind wells")
    ax.set_title("Is the uncertainty meaningful? (Hugin zone)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def maps_plot(maps, wells, use_fluid, path):
    key = "oil" if use_fluid else "sand"
    panels = []
    if f"truth_{key}" in maps:
        panels.append((f"truth_{key}", f"TRUTH {key} fraction (synthetic only)", "viridis", (0, 1)))
    panels += [(f"p_{key}_{m}", f"{METHOD_LABELS[m]}\nP({key})", "viridis", (0, 1)) for m in METHODS]
    panels += [("entropy_C", "C: normalised entropy", "magma", (0, 1)),
               ("ood", "C: local outlier factor vs library\n(~1 covered, >1.5 poorly covered)", "coolwarm", (0.5, 2))]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    for ax, (k, title, cmap, lim) in zip(axes.ravel(), panels):
        im = ax.imshow(maps[k], origin="lower", cmap=cmap, vmin=lim[0], vmax=lim[1])
        for w in wells:
            ax.plot(w.xl, w.il, "w^", mec="k", ms=7)
            ax.annotate(w.name, (w.xl, w.il), xytext=(3, 3), textcoords="offset points", fontsize=7, color="w")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("crossline")
        ax.set_ylabel("inline")
        fig.colorbar(im, ax=ax, shrink=0.8)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    fig.suptitle("Maps averaged over the window below the interpreted Hugin top")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def section_plot(probs, cube, il, use_fluid, path):
    names = class_names(use_fluid)
    cmap = _class_cmap(use_fluid)
    panels = []
    if cube.truth_class is not None:
        truth = cube.truth_class[il]
        if not use_fluid:
            truth = np.where(truth == 3, 2, truth)
        panels.append(("TRUTH (synthetic only)", truth + 1))
    panels += [(METHOD_LABELS[m], np.where(np.isnan(probs[m][il][..., 0]), -1,
                                           np.nan_to_num(probs[m][il]).argmax(-1)) + 1) for m in ("A", "C")]
    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 3 * len(panels)), sharex=True)
    for ax, (title, img) in zip(np.atleast_1d(axes), panels):
        ax.imshow(img.T, aspect="auto", cmap=cmap, vmin=0, vmax=len(names),
                  extent=[-0.5, cube.shape[1] - 0.5, cube.t[-1], cube.t[0]])
        ax.plot(np.arange(cube.shape[1]), cube.horizon_twt[il], "k--", lw=0.8)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel("TWT (s)")
    np.atleast_1d(axes)[-1].set_xlabel("crossline")
    handles = [plt.Rectangle((0, 0), 1, 1, color=CLASS_COLORS[n]) for n in names]
    fig.legend(handles, names, loc="lower center", ncol=len(names), fontsize=8)
    fig.suptitle(f"Inline {il}: most likely class (dashed = interpreted Hugin top; blank = outside prediction window)")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=120)
    plt.close(fig)
