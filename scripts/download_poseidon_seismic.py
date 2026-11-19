"""Download a sub-volume of the Poseidon 3D near, mid and far angle stacks as SEG-Y.

Source: TGS Open Data on AWS (s3://tgs-opendata-poseidon, public, CC BY 4.0), MDIO v1 / Zarr copies of
the Geoscience Australia / ConocoPhillips Poseidon 3D PSTM angle stacks (psdn11_TbsdmF_{Near,Mid,Far}).
The full stacks are ~104 GB each (3437 inlines x 5053 crosslines x 1501 samples @ 4 ms), so only the
requested inline/crossline/time window is fetched, chunk by chunk.

Default window: the 7 Poseidon-area wells (Boreas-1, Kronos-1, Pharos-1, Poseidon-1, Poseidon-2,
Poseidon-North-1, Proteus-1) with ~100 traces margin, and 2560-4092 ms TWT (Jamieson to below TD;
top Plover reservoir is ~3.1 s at Kronos-1). Torosa-1 (inline ~1194) is outside the default window;
use e.g. --il 1094 1294 --xl 2413 2613 --name torosa for a separate cube around it.

Output: data/poseidon/seismic/poseidon_{near,mid,far}[_<name>].segy, SEG-Y rev1, IEEE float, inline
sorted, inline/crossline in bytes 189/193, CDP X/Y (GDA94 / MGA zone 51) with scalar -100. Readable with
avo.segy_io.read_cube.

    python scripts/download_poseidon_seismic.py [--stacks near mid far] [--il 2310 3625]
        [--xl 1292 2769] [--t 2560 4092] [--name NAME] [--workers 16]
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import segyio
from numcodecs import Blosc
from segyio import BinField, TraceField

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "data" / "poseidon" / "seismic"
BUCKET = "https://tgs-opendata-poseidon.s3.amazonaws.com"
STACKS = {"near": "near_stack.mdio", "mid": "mid_stack.mdio", "far": "far_stack.mdio"}

SHAPE = (3437, 5053, 1501)  # inline, crossline, time
CHUNK = 128
COORD_CHUNK = (824, 1212)
_codec = Blosc()


def _get(url: str, retries: int = 12) -> bytes:
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=180) as response:
                return response.read()
        except Exception as exc:
            if attempt == retries - 1:
                raise
            wait = min(60, 2 ** attempt)
            print(f"    retry {attempt + 1}/{retries - 1} in {wait}s ({type(exc).__name__})", flush=True)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def _cached_get(url: str, cache_file: Path) -> bytes:
    """Fetch a chunk once; cached raw bytes let an interrupted download resume."""
    if cache_file.is_file():
        return cache_file.read_bytes()
    raw = _get(url)
    tmp = cache_file.with_suffix(".tmp")
    tmp.write_bytes(raw)
    tmp.replace(cache_file)
    return raw


def _decode(raw: bytes, dtype: str, shape) -> np.ndarray:
    return np.frombuffer(_codec.decode(raw), dtype=dtype).reshape(shape)


def _axis(store: str, name: str) -> np.ndarray:
    return np.frombuffer(_codec.decode(_get(f"{BUCKET}/{store}/{name}/0")), dtype="<u2").astype(int)


def _coords(store: str, name: str, il_slice: slice, xl_slice: slice, workers: int) -> np.ndarray:
    """Read a 2D coordinate array (cdp-x/cdp-y) for the requested inline/crossline window."""
    out = np.full((il_slice.stop - il_slice.start, xl_slice.stop - xl_slice.start), np.nan)
    ci = range(il_slice.start // COORD_CHUNK[0], (il_slice.stop - 1) // COORD_CHUNK[0] + 1)
    cj = range(xl_slice.start // COORD_CHUNK[1], (xl_slice.stop - 1) // COORD_CHUNK[1] + 1)

    def load(key):
        i, j = key
        block = _decode(_get(f"{BUCKET}/{store}/{name}/{i}/{j}"), "<f8", COORD_CHUNK)
        r0, c0 = i * COORD_CHUNK[0], j * COORD_CHUNK[1]
        rs = slice(max(r0, il_slice.start), min(r0 + COORD_CHUNK[0], il_slice.stop))
        cs = slice(max(c0, xl_slice.start), min(c0 + COORD_CHUNK[1], xl_slice.stop))
        out[rs.start - il_slice.start:rs.stop - il_slice.start, cs.start - xl_slice.start:cs.stop - xl_slice.start] = (
            block[rs.start - r0:rs.stop - r0, cs.start - c0:cs.stop - c0]
        )

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(load, [(i, j) for i in ci for j in cj]))
    return out


def _index_window(axis: np.ndarray, first: int, last: int, label: str) -> slice:
    if first > last or first < axis[0] or last > axis[-1]:
        raise SystemExit(f"{label} window {first}-{last} is outside the survey range {axis[0]}-{axis[-1]}")
    return slice(int(np.searchsorted(axis, first)), int(np.searchsorted(axis, last)) + 1)


def download_stack(stack: str, out_path: Path, il_s: slice, xl_s: slice, t_s: slice, axes, coords, workers: int):
    store = STACKS[stack]
    ilines, xlines, times = axes
    cdp_x, cdp_y = coords
    n_il, n_xl, n_t = il_s.stop - il_s.start, xl_s.stop - xl_s.start, t_s.stop - t_s.start
    dt_ms = int(times[1] - times[0])

    spec = segyio.spec()
    spec.ilines = [int(v) for v in ilines[il_s]]
    spec.xlines = [int(v) for v in xlines[xl_s]]
    spec.samples = [float(v) for v in times[t_s]]
    spec.sorting = segyio.TraceSortingFormat.INLINE_SORTING
    spec.format = 5

    tmp_path = out_path.with_suffix(".segy.part")
    cache_dir = out_path.parent / ".chunk_cache" / out_path.stem
    cache_dir.mkdir(parents=True, exist_ok=True)
    xl_chunks = range(xl_s.start // CHUNK, (xl_s.stop - 1) // CHUNK + 1)
    t_chunks = range(t_s.start // CHUNK, (t_s.stop - 1) // CHUNK + 1)
    started = time.time()
    with segyio.create(str(tmp_path), spec) as dst:
        dst.text[0] = segyio.tools.create_text_header({
            1: f"POSEIDON 3D {stack.upper()} ANGLE STACK - SUB-VOLUME",
            2: "SOURCE: TGS OPEN DATA S3://TGS-OPENDATA-POSEIDON (MDIO), CC BY 4.0",
            3: "ORIGINAL: GEOSCIENCE AUSTRALIA / CONOCOPHILLIPS PSDN11_TBSDMF",
            4: f"INLINES {spec.ilines[0]}-{spec.ilines[-1]} XLINES {spec.xlines[0]}-{spec.xlines[-1]}",
            5: f"TWT {spec.samples[0]:.0f}-{spec.samples[-1]:.0f} MS, DT {dt_ms} MS",
            6: "INLINE BYTE 189, CROSSLINE BYTE 193, CDP X/Y BYTES 181/185 SCALAR -100",
            7: "COORDINATES GDA94 / MGA ZONE 51 (EPSG:28351)",
        })
        dst.bin.update({BinField.Interval: dt_ms * 1000, BinField.Samples: n_t, BinField.Format: 5,
                        BinField.SortingCode: 2, BinField.MeasurementSystem: 1,
                        BinField.SEGYRevision: 0x0100})
        trace = 0
        il_chunks = range(il_s.start // CHUNK, (il_s.stop - 1) // CHUNK + 1)
        for n_block, ci in enumerate(il_chunks, 1):
            r0 = ci * CHUNK
            rows = slice(max(r0, il_s.start), min(r0 + CHUNK, il_s.stop))
            block = np.zeros((rows.stop - rows.start, n_xl, n_t), dtype=np.float32)

            def load(key, block=block, r0=r0, rows=rows):
                cj, ck = key
                chunk = _decode(
                    _cached_get(f"{BUCKET}/{store}/seismic/{ci}/{cj}/{ck}", cache_dir / f"{ci}_{cj}_{ck}.blosc"),
                    "<f4",
                    (CHUNK, CHUNK, CHUNK),
                )
                c0, k0 = cj * CHUNK, ck * CHUNK
                cols = slice(max(c0, xl_s.start), min(c0 + CHUNK, xl_s.stop))
                samp = slice(max(k0, t_s.start), min(k0 + CHUNK, t_s.stop))
                block[:, cols.start - xl_s.start:cols.stop - xl_s.start, samp.start - t_s.start:samp.stop - t_s.start] = (
                    chunk[rows.start - r0:rows.stop - r0, cols.start - c0:cols.stop - c0, samp.start - k0:samp.stop - k0]
                )

            with ThreadPoolExecutor(workers) as pool:
                list(pool.map(load, [(cj, ck) for cj in xl_chunks for ck in t_chunks]))
            np.nan_to_num(block, copy=False)

            for a in range(rows.start, rows.stop):
                ia = a - il_s.start
                for b in range(n_xl):
                    x, y = cdp_x[ia, b], cdp_y[ia, b]
                    dst.header[trace] = {
                        TraceField.TRACE_SEQUENCE_FILE: trace + 1, TraceField.CDP: trace + 1,
                        TraceField.INLINE_3D: int(ilines[a]), TraceField.CROSSLINE_3D: int(xlines[xl_s.start + b]),
                        TraceField.SourceGroupScalar: -100,
                        TraceField.CDP_X: int(round(x * 100)) if np.isfinite(x) else 0,
                        TraceField.CDP_Y: int(round(y * 100)) if np.isfinite(y) else 0,
                        TraceField.TRACE_SAMPLE_COUNT: n_t,
                        TraceField.TRACE_SAMPLE_INTERVAL: dt_ms * 1000,
                        TraceField.DelayRecordingTime: int(times[t_s.start]),
                    }
                    dst.trace[trace] = block[ia - (rows.start - il_s.start), b]
                    trace += 1
            elapsed = time.time() - started
            print(f"  {stack}: inline block {n_block}/{len(il_chunks)} done ({elapsed / 60:.1f} min)", flush=True)
    tmp_path.replace(out_path)
    shutil.rmtree(cache_dir, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stacks", nargs="+", choices=list(STACKS), default=list(STACKS))
    parser.add_argument("--il", nargs=2, type=int, default=(2310, 3625), metavar=("FIRST", "LAST"))
    parser.add_argument("--xl", nargs=2, type=int, default=(1292, 2769), metavar=("FIRST", "LAST"))
    parser.add_argument("--t", nargs=2, type=int, default=(2560, 4092), metavar=("FIRST_MS", "LAST_MS"))
    parser.add_argument("--name", default="", help="suffix for output file names")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)

    reference = STACKS[args.stacks[0]]
    ilines, xlines, times = (_axis(reference, name) for name in ("inline", "crossline", "time"))
    il_s = _index_window(ilines, *args.il, "Inline")
    xl_s = _index_window(xlines, *args.xl, "Crossline")
    t_s = _index_window(times, *args.t, "Time")
    n = (il_s.stop - il_s.start, xl_s.stop - xl_s.start, t_s.stop - t_s.start)
    print(f"Window: inlines {args.il[0]}-{args.il[1]}, crosslines {args.xl[0]}-{args.xl[1]}, "
          f"{times[t_s.start]}-{times[t_s.stop - 1]} ms -> {n[0]} x {n[1]} x {n[2]} "
          f"({n[0] * n[1] * n[2] * 4 / 1e9:.2f} GB per stack)")
    coords = tuple(_coords(reference, name, il_s, xl_s, args.workers) for name in ("cdp-x", "cdp-y"))

    args.out.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.name}" if args.name else ""
    for stack in args.stacks:
        out_path = args.out / f"poseidon_{stack}{suffix}.segy"
        if out_path.is_file():
            print(f"{out_path.name} already exists, skipping")
            continue
        print(f"Downloading {stack} stack -> {out_path}")
        download_stack(stack, out_path, il_s, xl_s, t_s, (ilines, xlines, times), coords, args.workers)
        print(f"  wrote {out_path.stat().st_size / 1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
