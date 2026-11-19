"""Probability cubes (task 09): lithology, fluid, uncertainty and most-likely class as SEG-Y per method.

Lithology:  P(shale), P(tight), P(sand)                      (sum to 1 where defined)
Fluid:      P(brine | sand), P(oil | sand)                   (sum to 1; null where P(sand) < threshold)
Extra:      normalised entropy of the class probabilities (0 = certain, 1 = uniform), most-likely class code
Null value outside the prediction window around the interpreted target horizon.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from avo import class_names
from avo.inference import entropy
from avo.pipeline import METHOD_LABELS, METHODS
from avo.segy_io import write_like, write_new


def cube_products(p, use_fluid, null, min_p_sand):
    """Derive output cubes from joint class probabilities p (..., n_classes); NaN marks unpredicted samples."""
    valid = ~np.isnan(p[..., 0])
    q = np.nan_to_num(p)
    sand = q[..., 2:].sum(-1)

    def put(a, mask=valid):
        return np.where(mask, a, null).astype(np.float32)

    out = {"litho_p_shale": put(q[..., 0]), "litho_p_tight": put(q[..., 1]), "litho_p_sand": put(sand)}
    if use_fluid:
        defined = valid & (sand >= min_p_sand)
        with np.errstate(invalid="ignore", divide="ignore"):
            out["fluid_p_brine_given_sand"] = put(q[..., 2] / sand, defined)
            out["fluid_p_oil_given_sand"] = put(q[..., 3] / sand, defined)
    out["uncertainty_entropy"] = put(entropy(q))
    out["most_likely_class"] = put(q.argmax(-1).astype(float))
    return out


DESCRIPTIONS = {
    "litho_p_shale": "Probability of shale (0-1).",
    "litho_p_tight": "Probability of tight/cemented rock (0-1).",
    "litho_p_sand": "Probability of sand, any fluid (0-1).",
    "fluid_p_brine_given_sand": "Probability of brine given sand (0-1). Null where P(sand) < threshold.",
    "fluid_p_oil_given_sand": "Probability of oil given sand (0-1). Null where P(sand) < threshold.",
    "uncertainty_entropy": "Normalised entropy of the class probabilities: 0 = certain, 1 = uniform.",
    "most_likely_class": "Most likely class code (see class_codes).",
}


def export_cubes(res, cube, cfg, out_dir, synthetic=False):
    """Write one SEG-Y per product and method, plus cubes_manifest.json describing them."""
    ex = cfg["export"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = class_names(res["use_fluid"])
    meta = cube.meta
    files = []
    for m in METHODS:
        for product, data in cube_products(res["probs"][m], res["use_fluid"], ex["null_value"],
                                           ex["fluid_min_p_sand"]).items():
            path = out_dir / f"{m}_{product}.segy"
            text = ["FORCE HACKATHON 2026 - AVO LITHOLOGY/FLUID PROBABILITY CUBE",
                    f"PRODUCT: {product}",
                    f"METHOD: {METHOD_LABELS[m]}",
                    DESCRIPTIONS[product].upper(),
                    f"NULL VALUE: {ex['null_value']}",
                    f"WINDOW: {ex['window_ms'][0]} TO {ex['window_ms'][1]} MS RELATIVE TO TARGET HORIZON",
                    "CLASS CODES: " + " ".join(f"{i}={n.upper()}" for i, n in enumerate(names)),
                    f"FLUID DEFINED WHERE P(SAND) >= {ex['fluid_min_p_sand']}",
                    "FORMAT: SEG-Y REV1, IEEE FLOAT, TIME DOMAIN",
                    "GEOMETRY COPIED FROM: " + Path(meta.get("template", "")).name,
                    "SYNTHETIC TEST DATA - NOT A GEOLOGICAL RESULT" if synthetic else ""]
            if meta.get("template"):
                write_like(meta["template"], path, data, meta["ilines"], meta["xlines"], text,
                           meta.get("iline_byte", 189), meta.get("xline_byte", 193))
            else:
                write_new(path, data, meta["ilines"], meta["xlines"], cube.t0, cube.dt, meta["cdp_x"],
                          meta["cdp_y"], text)
            files.append(dict(file=path.name, method=m, method_label=METHOD_LABELS[m], product=product,
                              description=DESCRIPTIONS[product]))
    manifest = dict(created=datetime.now(timezone.utc).isoformat(timespec="seconds"), synthetic=synthetic,
                    null_value=ex["null_value"], window_ms_relative_to_horizon=ex["window_ms"],
                    fluid_min_p_sand=ex["fluid_min_p_sand"], fluid_is_target=res["use_fluid"],
                    class_codes={i: n for i, n in enumerate(names)}, template=meta.get("template"),
                    format="SEG-Y rev1, IEEE float (format 5), time domain", files=files)
    (out_dir / "cubes_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
