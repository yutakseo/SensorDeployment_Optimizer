from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Analysis.internal.reports.ga_statistics import saveGaStatisticsReport

__all__ = ["saveGaStatisticsReport"]


if __name__ == "__main__":
    from Analysis.internal.reports.ga_statistics import main

    main()
