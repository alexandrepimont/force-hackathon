"""What-if simulation at a well (task 10): change Hugin fluid, porosity or thickness, re-model and re-predict.

Every answer comes from running the forward model and the trained classifiers; nothing is inferred by hand.

    python -m avo.whatif --well W1 --fluid brine
    python -m avo.whatif --well W5 --phi-delta -0.05 --thickness-scale 0.5
"""

import argparse
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from avo import LITHO_SAND, class_names  # noqa: E402
from avo import forward as fw  # noqa: E402
from avo import rock_physics as rp  # noqa: E402
from avo.pipeline import METHODS, predict_all  # noqa: E402
from avo.plots import CLASS_COLORS  # noqa: E402
from avo.synthetic_earth import build_synthetic_dataset  # noqa: E402


def modify_hugin(well, cfg, fluid=None, phi_delta=0.0, thickness_scale=1.0):
    """Return modified (vp, vs, rho) depth arrays plus a mask of the (possibly stretched) Hugin interval."""
    l = well.logs
    a = {c: l[c].to_numpy().astype(float).copy() for c in ("vp", "vs", "rho", "phi", "vsh", "sw", "litho")}
    z = l["tvdss"].to_numpy()
    hug = (z >= well.hugin_top) & (z < well.hugin_base)
    sand = hug & (a["litho"] == LITHO_SAND)
    rpc = cfg["rock_physics"]
    fluids = rp.fluid_set(rpc["reservoir"])
    if phi_delta:
        a["vp"][sand], a["vs"][sand], a["rho"][sand], a["phi"][sand] = rp.change_porosity(
            a["vp"][sand], a["vs"][sand], a["rho"][sand], a["phi"][sand], a["vsh"][sand], a["sw"][sand], phi_delta,
            fluids, rpc["minerals"], rpc["frame"]["phic"], rpc["frame"]["exponent"])
    if fluid:
        target = 1.0 if fluid == "brine" else cfg["gate_g3"]["sw_oil"]
        a["vp"][sand], a["vs"][sand], a["rho"][sand] = rp.substitute_fluid(
            a["vp"][sand], a["vs"][sand], a["rho"][sand], a["phi"][sand], a["vsh"][sand], a["sw"][sand], target,
            fluids, rpc["minerals"])
    idx = np.arange(len(z))
    if thickness_scale != 1.0:
        h = np.flatnonzero(hug)
        k = max(2, int(round(len(h) * thickness_scale)))
        stretched = h[0] + np.minimum((np.arange(k) / thickness_scale).astype(int), len(h) - 1)
        idx = np.r_[idx[:h[0]], stretched, idx[h[-1] + 1:]]
    mask = np.zeros(len(idx), bool)
    mask[np.flatnonzero(np.isin(idx, np.flatnonzero(hug)))] = True
    return a["vp"][idx], a["vs"][idx], a["rho"][idx], mask


def simulate(well, bundle, cube, vp, vs, rho, mask):
    calib = bundle["calib"]
    t_top = float(np.interp(well.logs["tvdss"].iloc[0], well.checkshot["tvdss"], well.checkshot["twt"]))
    twt = fw.twt_from_vp(vp, well.dz, t_top) + calib.shifts.get(well.name, 0.0)
    traces = fw.synthesize(twt, vp, vs, rho, cube.stack_defs, calib.wavelet_list(cube.stack_defs), cube.t,
                           calib.dt_model)
    probs = predict_all(bundle, traces)
    zone = (cube.t >= twt[mask].min()) & (cube.t <= twt[mask].max())
    return traces, probs, zone


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--well", required=True)
    ap.add_argument("--fluid", choices=["brine", "oil"])
    ap.add_argument("--phi-delta", type=float, default=0.0, help="porosity change in Hugin sands (fraction)")
    ap.add_argument("--thickness-scale", type=float, default=1.0)
    ap.add_argument("--run", default="outputs/synthetic", help="folder with final_bundle.pkl")
    args = ap.parse_args(argv)

    saved = pickle.loads(Path(args.run, "final_bundle.pkl").read_bytes())
    bundle, cfg = saved["bundle"], saved["cfg"]
    wells, cube = build_synthetic_dataset(saved["earth_cfg"])
    well = next(w for w in wells if w.name == args.well)

    base = simulate(well, bundle, cube, *modify_hugin(well, cfg))
    new = simulate(well, bundle, cube, *modify_hugin(well, cfg, args.fluid, args.phi_delta, args.thickness_scale))

    names = class_names(bundle["use_fluid"])
    rows = []
    for label, (_, probs, zone) in (("base", base), ("what-if", new)):
        for m in METHODS:
            frac = np.bincount(probs[m][zone].argmax(1), minlength=len(names)) / max(zone.sum(), 1)
            rows.append(dict(case=label, method=m, **{n: round(float(f), 2) for n, f in zip(names, frac)}))
    table = pd.DataFrame(rows)
    desc = f"fluid={args.fluid or 'in-situ'}, dphi={args.phi_delta:+.3f}, thickness x{args.thickness_scale:g}"
    print(f"What-if at {well.name}: {desc}\nPredicted class fractions in the Hugin time window:")
    print(table.to_string(index=False))

    fig, axes = plt.subplots(1, len(cube.stack_defs) + 2, figsize=(12, 6), sharey=True,
                             gridspec_kw={"width_ratios": [2] * len(cube.stack_defs) + [0.6, 0.6]})
    for k, s in enumerate(cube.stack_defs):
        axes[k].plot(base[0][k], cube.t, "k", lw=1, label="base")
        axes[k].plot(new[0][k], cube.t, "r", lw=1, label="what-if")
        axes[k].set_title(s.name)
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap([CLASS_COLORS[n] for n in names])
    for ax, (title, res) in zip(axes[-2:], (("base C", base), ("what-if C", new))):
        ax.imshow(res[1]["C"].argmax(1)[:, None], aspect="auto", cmap=cmap, vmin=0, vmax=len(names) - 1,
                  extent=[0, 1, cube.t[-1], cube.t[0]])
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
    zone = base[2] | new[2]
    axes[0].set_ylim(cube.t[zone].max() + 0.08, cube.t[zone].min() - 0.08)
    axes[0].set_ylabel("TWT (s)")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"What-if at {well.name}: {desc}")
    fig.tight_layout()
    out = Path(args.run, f"whatif_{well.name}.png")
    fig.savefig(out, dpi=120)
    print(f"Figure: {out}")
    return table


if __name__ == "__main__":
    main()
