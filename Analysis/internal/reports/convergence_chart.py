from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator

from Analysis.internal.plot_style import DEFAULT_DPI, applySciStyle, styleAxis
from Analysis.internal.result_io import cornerCount, loadRecords
from Analysis.internal.reports.n_sensor import ALGORITHM_NAMES

DEFAULT_RESULTS_ROOT = "__RESULTS__"
DEFAULT_OUTPUT_DIR = "__RESULTS__/analysis/convergence_by_algorithm"
DEFAULT_METRIC = "sensors"
FIGURE_SIZE = (3.5, 2.6)
DRL_X_SCALE = 0.1
AXIS_PADDING_RATIO = 0.06
LINE_WIDTH = 1.3
MARKER_SIZE = 3.0
MARK_COUNT = 10
GA_ALGORITHM = "ga"
DRL_ALGORITHM = "drl"
LINE_STYLES = {
    "drl": (0, (4, 2)),
    "ga": "-",
    "greedy": (0, (1, 1)),
    "pso": "-.",
}
LINE_COLORS = {
    "drl": "0.35",
    "ga": "#2A9FD6",
    "greedy": "0.20",
    "pso": "0.55",
}
MARKERS = {
    "drl": "o",
    "ga": "s",
    "greedy": "^",
    "pso": "D",
}
Y_LABELS = {
    "best_fitness": "Best fitness",
    "best_coverage": "Best coverage (%)",
    "sensors": "Number of sensors",
}


@dataclass(frozen=True, slots=True)
class TrendSeries:
    algorithm: str
    x_values: tuple[float, ...]
    mean_values: tuple[float, ...]


def listMaps(results_root: Path, algorithms: Sequence[str]) -> list[str]:
    map_names: set[str] = set()
    for algorithm in algorithms:
        algorithm_root = results_root / algorithm
        if not algorithm_root.exists():
            continue
        map_names.update(path.name for path in algorithm_root.iterdir() if path.is_dir())
    return sorted(map_names)


def metricValue(
    generation: dict[str, Any],
    *,
    metric: str,
    corner_count: int,
) -> float | None:
    if metric in {"best_fitness", "best_coverage"}:
        value = generation.get(metric)
    elif metric == "sensors":
        value = generation.get("n_inner")
        if not isinstance(value, (int, float)):
            solution = generation.get("best_solution", [])
            value = len(solution) if isinstance(solution, list) else None
        if isinstance(value, (int, float)):
            value = float(value) + float(corner_count)
    else:
        raise ValueError("metric must be one of: best_fitness, best_coverage, sensors")

    if not isinstance(value, (int, float)):
        return None
    if math.isnan(float(value)):
        return None
    return float(value)


def getMetricSeries(run: dict[str, Any], metric: str) -> list[float]:
    generations = run.get("generations", [])
    if not isinstance(generations, list):
        return []

    corners = cornerCount(run)
    values: list[float] = []
    for generation in generations:
        if not isinstance(generation, dict):
            continue
        value = metricValue(generation, metric=metric, corner_count=corners)
        if value is None:
            continue
        values.append(value)
    return values


def meanSeries(run_series: Sequence[Sequence[float]]) -> list[float]:
    clean_series = [list(series) for series in run_series if series]
    if not clean_series:
        return []

    length = min(len(series) for series in clean_series)
    if length <= 0:
        return []

    matrix = np.asarray([series[:length] for series in clean_series], dtype=np.float64)
    return [float(value) for value in np.mean(matrix, axis=0)]


def xValues(algorithm: str, length: int) -> tuple[float, ...]:
    scale = DRL_X_SCALE if algorithm == DRL_ALGORITHM else 1.0
    return tuple(float(index) * scale for index in range(1, length + 1))


def loadTrend(
    *,
    results_root: Path,
    algorithm: str,
    map_name: str,
    metric: str,
) -> TrendSeries | None:
    map_root = results_root / algorithm / map_name
    try:
        records = loadRecords(map_root)
    except (FileNotFoundError, ValueError):
        return None

    run_series = [getMetricSeries(run, metric) for _, run in records]
    means = meanSeries(run_series)
    if not means:
        return None

    return TrendSeries(
        algorithm=algorithm,
        x_values=xValues(algorithm, len(means)),
        mean_values=tuple(means),
    )


def loadMapTrends(
    *,
    results_root: Path,
    map_name: str,
    algorithms: Sequence[str],
    metric: str,
) -> list[TrendSeries]:
    trends: list[TrendSeries] = []
    for algorithm in algorithms:
        trend = loadTrend(
            results_root=results_root,
            algorithm=algorithm,
            map_name=map_name,
            metric=metric,
        )
        if trend is not None:
            trends.append(trend)
    return trends


def markerEvery(series: TrendSeries) -> int:
    return max(1, len(series.x_values) // MARK_COUNT)


def yLimits(series_list: Sequence[TrendSeries]) -> tuple[float, float]:
    values = [
        value
        for series in series_list
        for value in series.mean_values
        if not math.isnan(float(value))
    ]
    if not values:
        return (0.0, 1.0)

    minimum = min(values)
    maximum = max(values)
    if math.isclose(minimum, maximum):
        padding = max(1.0, abs(maximum) * AXIS_PADDING_RATIO)
    else:
        padding = (maximum - minimum) * AXIS_PADDING_RATIO
    return (minimum - padding, maximum + padding)


def drawMap(
    axis: Axes,
    *,
    map_name: str,
    trends: Sequence[TrendSeries],
    metric: str,
) -> None:
    for series in trends:
        algorithm = series.algorithm
        axis.plot(
            series.x_values,
            series.mean_values,
            color=LINE_COLORS.get(algorithm, "0.20"),
            linestyle=LINE_STYLES.get(algorithm, "-"),
            linewidth=LINE_WIDTH,
            marker=MARKERS.get(algorithm, "o"),
            markersize=MARKER_SIZE,
            markevery=markerEvery(series),
            label=algorithm.upper(),
        )

    x_max = max(max(series.x_values) for series in trends)
    axis.set_xlim(0, x_max)
    axis.set_ylim(*yLimits(trends))
    axis.set_xlabel("Generation")
    axis.set_ylabel(Y_LABELS.get(metric, metric))
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    if metric in {"best_coverage", "sensors"}:
        axis.yaxis.set_major_locator(MaxNLocator(integer=True))
    axis.set_axisbelow(True)
    axis.grid(True, axis="both", color="0.88", linewidth=0.5)
    axis.legend(frameon=False, loc="best", ncol=1)
    styleAxis(axis)


def chartPath(output_dir: Path, map_name: str) -> Path:
    return output_dir / f"{map_name}.png"


def saveMapChart(
    *,
    output_dir: Path,
    map_name: str,
    trends: Sequence[TrendSeries],
    metric: str,
    dpi: int,
    show: bool,
) -> Path:
    applySciStyle()
    fig, axis = plt.subplots(figsize=FIGURE_SIZE, dpi=dpi)
    drawMap(axis, map_name=map_name, trends=trends, metric=metric)
    fig.tight_layout()

    output_path = chartPath(output_dir, map_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)
    return output_path


def saveConvergenceCharts(
    *,
    results_root: str = DEFAULT_RESULTS_ROOT,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    algorithms: Sequence[str] = ALGORITHM_NAMES,
    metric: str = DEFAULT_METRIC,
    dpi: int = DEFAULT_DPI,
    show: bool = False,
) -> list[Path]:
    """Create per-map convergence charts across algorithms."""
    source_root = Path(results_root)
    target_dir = Path(output_dir)
    output_paths: list[Path] = []

    for map_name in listMaps(source_root, algorithms):
        trends = loadMapTrends(
            results_root=source_root,
            map_name=map_name,
            algorithms=algorithms,
            metric=metric,
        )
        if not trends:
            continue
        output_paths.append(
            saveMapChart(
                output_dir=target_dir,
                map_name=map_name,
                trends=trends,
                metric=metric,
                dpi=dpi,
                show=show,
            )
        )

    if not output_paths:
        raise ValueError(f"No plottable convergence data found: {source_root}")
    return output_paths


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create per-map convergence charts by algorithm."
    )
    parser.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--algorithms", nargs="+", default=list(ALGORITHM_NAMES))
    parser.add_argument(
        "--metric",
        choices=("best_fitness", "best_coverage", "sensors"),
        default=DEFAULT_METRIC,
    )
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parseArgs()
    output_paths = saveConvergenceCharts(
        results_root=args.results_root,
        output_dir=args.output_dir,
        algorithms=tuple(args.algorithms),
        metric=str(args.metric),
        dpi=args.dpi,
        show=args.show,
    )
    for output_path in output_paths:
        print(output_path)


if __name__ == "__main__":
    main()
