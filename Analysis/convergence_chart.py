from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Analysis.internal.reports.convergence_chart import (
    DEFAULT_DPI,
    DEFAULT_METRIC,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RESULTS_ROOT,
    saveConvergenceCharts,
)

RESULTS_ROOT = DEFAULT_RESULTS_ROOT
OUTPUT_DIR = DEFAULT_OUTPUT_DIR
ALGORITHMS = ("ga", "drl")
METRIC = DEFAULT_METRIC
DPI = DEFAULT_DPI
SHOW = False


def main() -> None:
    output_paths = saveConvergenceCharts(
        results_root=RESULTS_ROOT,
        output_dir=OUTPUT_DIR,
        algorithms=ALGORITHMS,
        metric=METRIC,
        dpi=DPI,
        show=SHOW,
    )
    print("Saved convergence charts:")
    for output_path in output_paths:
        print(output_path)


if __name__ == "__main__":
    main()
