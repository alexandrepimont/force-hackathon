"""SEG-Y input/output.

- read_cube: partial-stack SEG-Y files -> Cube (used for the synthetic test now and for Volve ST10010 later).
- write_new: write a cube with fresh headers (used to turn the synthetic stacks into SEG-Y).
- write_like: write a cube with geometry and trace headers copied from a template stack, so the output
  overlays the input seismic exactly in interpretation software.

Output is SEG-Y rev1, IEEE float (format 5). Surveys are assumed to be regular 3D grids readable by segyio
with the given inline/crossline header bytes.
"""

import numpy as np
import segyio
from segyio import BinField, TraceField

from avo.data import Cube


def _text_header(lines):
    return segyio.tools.create_text_header({i + 1: str(line)[:76] for i, line in enumerate(lines[:40])})


def _grid(f):
    """Cube array (n_il, n_xl, nt) regardless of inline/crossline sorting."""
    data = segyio.tools.cube(f)
    if f.sorting == segyio.TraceSortingFormat.CROSSLINE_SORTING:
        data = data.transpose(1, 0, 2)
    return data


def _coords(f):
    """CDP X/Y per trace with the SEG-Y coordinate scalar applied (negative = divide, positive = multiply)."""
    scal = f.attributes(TraceField.SourceGroupScalar)[:].astype(float)
    factor = np.where(scal < 0, 1.0 / np.abs(np.where(scal == 0, 1, scal)), np.where(scal == 0, 1.0, scal))
    return f.attributes(TraceField.CDP_X)[:] * factor, f.attributes(TraceField.CDP_Y)[:] * factor


def read_cube(stack_paths, stack_defs, iline=189, xline=193):
    """Read partial stacks (dict stack name -> path) into a Cube. All stacks must share geometry."""
    stacks, ref = {}, None
    for s in stack_defs:
        path = stack_paths[s.name]
        with segyio.open(path, iline=iline, xline=xline) as f:
            f.mmap()
            geom = dict(ilines=np.array(f.ilines), xlines=np.array(f.xlines), samples=np.array(f.samples))
            if ref is None:
                ref = geom
                il_idx = {v: i for i, v in enumerate(geom["ilines"])}
                xl_idx = {v: i for i, v in enumerate(geom["xlines"])}
                cdp_x = np.full((len(geom["ilines"]), len(geom["xlines"])), np.nan)
                cdp_y = cdp_x.copy()
                ils = f.attributes(iline)[:]
                xls = f.attributes(xline)[:]
                x, y = _coords(f)
                for a, b, xx, yy in zip(ils, xls, x, y):
                    cdp_x[il_idx[a], xl_idx[b]], cdp_y[il_idx[a], xl_idx[b]] = xx, yy
            elif any(not np.array_equal(geom[k], ref[k]) for k in geom):
                raise ValueError(f"{path}: geometry differs from {stack_paths[stack_defs[0].name]}")
            stacks[s.name] = _grid(f).astype(np.float64)
    samples = ref["samples"]
    meta = dict(template=str(stack_paths[stack_defs[0].name]), ilines=ref["ilines"], xlines=ref["xlines"],
                cdp_x=cdp_x, cdp_y=cdp_y, iline_byte=iline, xline_byte=xline)
    return Cube(stacks, list(stack_defs), samples[0] / 1000.0, (samples[1] - samples[0]) / 1000.0, meta=meta)


def write_new(path, data, ilines, xlines, t0, dt, cdp_x, cdp_y, text_lines=()):
    """Write (n_il, n_xl, nt) data as inline-sorted SEG-Y rev1 IEEE float with coordinates in cm (scalar -100)."""
    n_il, n_xl, nt = data.shape
    spec = segyio.spec()
    spec.ilines, spec.xlines = list(ilines), list(xlines)
    spec.samples = list(t0 * 1000.0 + np.arange(nt) * dt * 1000.0)
    spec.sorting = segyio.TraceSortingFormat.INLINE_SORTING
    spec.format = 5
    with segyio.create(str(path), spec) as f:
        f.text[0] = _text_header(list(text_lines))
        f.bin.update({BinField.Interval: int(round(dt * 1e6)), BinField.Samples: nt, BinField.Format: 5,
                      BinField.SortingCode: 2, BinField.MeasurementSystem: 1, BinField.SEGYRevision: 0x0100})
        i = 0
        for a, il in enumerate(ilines):
            for b, xl in enumerate(xlines):
                f.header[i] = {TraceField.TRACE_SEQUENCE_FILE: i + 1, TraceField.CDP: i + 1,
                               TraceField.INLINE_3D: int(il), TraceField.CROSSLINE_3D: int(xl),
                               TraceField.SourceGroupScalar: -100,
                               TraceField.CDP_X: int(round(cdp_x[a, b] * 100)),
                               TraceField.CDP_Y: int(round(cdp_y[a, b] * 100)),
                               TraceField.TRACE_SAMPLE_COUNT: nt,
                               TraceField.TRACE_SAMPLE_INTERVAL: int(round(dt * 1e6)),
                               TraceField.DelayRecordingTime: int(round(t0 * 1000))}
                f.trace[i] = np.ascontiguousarray(data[a, b], dtype=np.float32)
                i += 1


def write_like(template, path, data, ilines, xlines, text_lines=(), iline=189, xline=193):
    """Write (n_il, n_xl, nt) data with geometry and trace headers copied from a template SEG-Y."""
    il_idx = {v: i for i, v in enumerate(ilines)}
    xl_idx = {v: i for i, v in enumerate(xlines)}
    with segyio.open(str(template), iline=iline, xline=xline) as src:
        spec = segyio.tools.metadata(src)
        spec.format = 5
        if len(src.samples) != data.shape[2]:
            raise ValueError(f"sample count {data.shape[2]} differs from template {len(src.samples)}")
        ils, xls = src.attributes(iline)[:], src.attributes(xline)[:]
        with segyio.create(str(path), spec) as dst:
            dst.text[0] = _text_header(list(text_lines))
            dst.bin = src.bin
            dst.bin.update({BinField.Format: 5, BinField.SEGYRevision: 0x0100})
            for i in range(src.tracecount):
                dst.header[i] = src.header[i]
                dst.trace[i] = np.ascontiguousarray(data[il_idx[ils[i]], xl_idx[xls[i]]], dtype=np.float32)
