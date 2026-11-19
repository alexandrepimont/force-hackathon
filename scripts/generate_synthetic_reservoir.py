"""Generate seven synthetic well logs and a faulted pre-stack reservoir model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.synthetic_reservoir import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT,
    generate_reservoir,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output only when it has a valid synthetic manifest.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metadata = generate_reservoir(args.config, args.output_dir, overwrite=args.overwrite)
    print(
        f"Generated {metadata['well_count']} synthetic wells, "
        f"{metadata['seismic']['combined_trace_count']:,} combined angle traces, "
        f"and {len(metadata['files'])} deliverable files in {args.output_dir}."
    )
    print(metadata["warning"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
