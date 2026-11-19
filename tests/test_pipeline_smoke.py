import json

import numpy as np
import pandas as pd
import segyio

from avo import run_synthetic, whatif


def test_quick_pipeline_runs_end_to_end_without_blind_well_leakage(tmp_path):
    out, data = tmp_path / "out", tmp_path / "data"
    res = run_synthetic.main(["--quick", "--out", str(out), "--data", str(data)])

    for name in ("report.md", "decision_log.md", "scores.csv", "maps.png", "final_bundle.pkl"):
        assert (out / name).exists()
    scores = pd.read_csv(out / "scores.csv")
    assert set(scores["method"]) == {"A", "B", "C"}
    assert list(res["decisions"]["gate"]) == ["G1", "G2", "G3", "G4"]
    for blind, pw in res["per_well"].items():
        assert blind not in pw["train_wells"]

    _check_probability_cubes(out / "cubes", data / "seismic" / "synthetic_near.segy")

    table = whatif.main(["--well", "W1", "--fluid", "brine", "--run", str(out)])
    assert set(table["case"]) == {"base", "what-if"}


def _read(path):
    with segyio.open(path) as f:
        return (segyio.tools.cube(f), np.array(f.ilines), np.array(f.xlines), np.array(f.samples),
                f.attributes(segyio.TraceField.CDP_X)[:], f.bin[segyio.BinField.Format])


def _check_probability_cubes(cube_dir, template):
    manifest = json.loads((cube_dir / "cubes_manifest.json").read_text())
    null = manifest["null_value"]
    assert {f["method"] for f in manifest["files"]} == {"A", "B", "C"}
    tmpl = _read(template)

    for m in ("A", "B", "C"):
        lith = {p: _read(cube_dir / f"{m}_litho_p_{p}.segy") for p in ("shale", "tight", "sand")}
        data, il, xl, samples, cdp_x, fmt = lith["sand"]
        assert fmt == 5
        assert np.array_equal(il, tmpl[1]) and np.array_equal(xl, tmpl[2]) and np.array_equal(samples, tmpl[3])
        assert np.array_equal(cdp_x, tmpl[4])

        valid = data != null
        assert valid.any() and (~valid).any(), "expected predictions inside the window and nulls outside"
        total = sum(lith[p][0] for p in lith)
        assert np.allclose(total[valid], 1.0, atol=1e-4)

        if manifest["fluid_is_target"]:
            brine = _read(cube_dir / f"{m}_fluid_p_brine_given_sand.segy")[0]
            oil = _read(cube_dir / f"{m}_fluid_p_oil_given_sand.segy")[0]
            defined = oil != null
            assert np.array_equal(defined, valid & (data >= manifest["fluid_min_p_sand"]))
            assert np.allclose(brine[defined] + oil[defined], 1.0, atol=1e-4)

        ent = _read(cube_dir / f"{m}_uncertainty_entropy.segy")[0]
        assert ent[valid].min() >= -1e-6 and ent[valid].max() <= 1 + 1e-6
        cls = _read(cube_dir / f"{m}_most_likely_class.segy")[0]
        assert set(np.unique(cls[valid]).astype(int)) <= {int(k) for k in manifest["class_codes"]}
