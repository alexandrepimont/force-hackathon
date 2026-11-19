"""Run Poseidon well QC and petrophysics (agent task 02).

    python scripts/run_well_qc.py
"""

import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from avo import wellqc  # noqa: E402


def main() -> int:
    cfg = yaml.safe_load((ROOT / "configs" / "petrophysics.yaml").read_text())
    summary = wellqc.run(ROOT, cfg)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(summary.to_string(index=False))
    print(f"\nWrote data/interim/poseidon_logs/, outputs/well_qc/ ({len(summary)} wells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
