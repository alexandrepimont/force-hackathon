"""Run the full workflow on the synthetic Hugin-like test case.

    python -m avo.run_synthetic [--earth configs/synthetic_earth.yaml] [--pipeline configs/pipeline.yaml]
                                [--out outputs/synthetic] [--quick]
"""

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import yaml

from avo import plots
from avo.export import export_cubes
from avo.pipeline import run_pipeline
from avo.report import write_decision_log, write_report
from avo.segy_io import read_cube, write_new
from avo.synthetic_earth import build_synthetic_dataset


def quick_overrides(earth_cfg, cfg):
    """Small, fast settings for smoke tests."""
    earth_cfg["grid"] = {"n_il": 14, "n_xl": 14}
    earth_cfg["wells"] = [{"name": "W1", "il": 7, "xl": 9}, {"name": "W2", "il": 6, "xl": 3},
                          {"name": "W3", "il": 12, "xl": 11}, {"name": "W4", "il": 2, "xl": 7}]
    cfg["library"]["n_pseudo_wells"] = 40
    cfg["model"]["max_iter"] = 30
    cfg["calibration"]["freqs"] = [18, 36, 3]
    cfg["calibration"]["phases"] = [-20, 20, 10]
    return earth_cfg, cfg


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--earth", default="configs/synthetic_earth.yaml")
    ap.add_argument("--pipeline", default="configs/pipeline.yaml")
    ap.add_argument("--out", default="outputs/synthetic")
    ap.add_argument("--data", default="data/synthetic", help="where the synthetic stacks are written as SEG-Y")
    ap.add_argument("--quick", action="store_true", help="small grid and library for a fast smoke test")
    args = ap.parse_args(argv)

    earth_cfg = yaml.safe_load(Path(args.earth).read_text())
    cfg = yaml.safe_load(Path(args.pipeline).read_text())
    if args.quick:
        earth_cfg, cfg = quick_overrides(earth_cfg, cfg)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    wells, synth_cube = build_synthetic_dataset(earth_cfg)
    seis_dir = Path(args.data) / "seismic"
    seis_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for s in synth_cube.stack_defs:
        paths[s.name] = seis_dir / f"synthetic_{s.name}.segy"
        m = synth_cube.meta
        write_new(paths[s.name], synth_cube.stacks[s.name], m["ilines"], m["xlines"], synth_cube.t0, synth_cube.dt,
                  m["cdp_x"], m["cdp_y"], [f"SYNTHETIC {s.name.upper()} STACK {s.angle_min}-{s.angle_max} DEG",
                                           "FORCE HACKATHON 2026 - AVO WORKFLOW TEST DATA"])
    # Read back through the same loader that will be used for the Volve ST10010 stacks.
    cube = read_cube(paths, synth_cube.stack_defs)
    for s in cube.stack_defs:
        assert np.allclose(cube.stacks[s.name], synth_cube.stacks[s.name], rtol=1e-6, atol=1e-3)
    cube.horizon_twt, cube.truth_class = synth_cube.horizon_twt, synth_cube.truth_class
    print(f"[data] synthetic earth: {cube.shape} traces x {cube.nt} samples, {len(wells)} wells; "
          f"stacks read from {seis_dir}")
    res = run_pipeline(wells, cube, cfg)
    print(res["decisions"].to_string(index=False))

    plots.well_tie(wells[0], cube, res["calib"], out / "well_tie.png")
    plots.fluid_sensitivity(res["g3_panels"], cube, out / "fluid_sensitivity.png")
    plots.ig_crossplot(res["final"], wells, cube, out / "ig_crossplot.png")
    for w in wells:
        plots.blind_well(w.name, res["per_well"][w.name], cube, res["use_fluid"], out / f"blind_{w.name}.png")
    plots.reliability_plot(res["reliability"], out / "reliability.png")
    plots.maps_plot(res["maps"], wells, res["use_fluid"], out / "maps.png")
    plots.section_plot(res["probs"], cube, wells[0].il, res["use_fluid"], out / "section.png")

    res["scores"].to_csv(out / "scores.csv", index=False)
    res["calib"].ties.to_csv(out / "well_ties.csv", index=False)
    res["g3_table"].to_csv(out / "fluid_sensitivity.csv", index=False)
    if res["truth_eval"] is not None:
        res["truth_eval"].to_csv(out / "map_truth_scores.csv", index=False)
    write_decision_log(res, out)
    write_report(res, wells, cube, out, synthetic=True)
    manifest = export_cubes(res, cube, cfg, out / "cubes", synthetic=True)
    print(f"[cubes] {len(manifest['files'])} SEG-Y probability cubes -> {out / 'cubes'}")
    with open(out / "final_bundle.pkl", "wb") as f:
        pickle.dump(dict(bundle=res["final"], cfg=cfg, earth_cfg=earth_cfg), f)
    print(f"[done] {time.time() - t0:.0f} s -> {out / 'report.md'}")
    return res


if __name__ == "__main__":
    main()
