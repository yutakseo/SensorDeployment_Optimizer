from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt

PathLike = Union[str, Path]


@dataclass(frozen=True, slots=True)
class FitnessPoint:
    index: int
    positions: Tuple[Tuple[int, int], ...]
    fitness: float
    sensor_count: int
    coverage: float


def loadCombinatorialFitness(
    path: PathLike,
    *,
    only_feasible: bool = True,
    max_points: Optional[int] = None,
) -> List[FitnessPoint]:
    if max_points is not None and max_points <= 0:
        raise ValueError("max_points must be positive when provided.")

    points: List[FitnessPoint] = []
    step = 1
    accepted = 0
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            payload = json.loads(line)
            if payload.get("type") != "fitness":
                continue
            if only_feasible and not bool(payload.get("feasible", False)):
                continue
            if payload.get("fitness") is None or payload.get("coverage") is None:
                continue
            positions = tuple(
                (int(position[0]), int(position[1]))
                for position in payload.get("positions", [])
            )
            point = FitnessPoint(
                index=int(payload["index"]),
                positions=positions,
                fitness=float(payload["fitness"]),
                sensor_count=int(payload["sensor_count"]),
                coverage=float(payload["coverage"]),
            )
            accepted += 1
            if max_points is None or (accepted - 1) % step == 0:
                points.append(point)
            if max_points is not None and len(points) > max_points:
                points = points[::2]
                step *= 2
    return points


def plotCombinatorialFitness3d(
    path: PathLike,
    *,
    save_path: Optional[PathLike] = None,
    show: bool = False,
    only_feasible: bool = True,
    figsize: tuple[float, float] = (10.0, 8.0),
    dpi: int = 140,
    marker_size: float = 12.0,
    alpha: float = 0.75,
    max_xtick_labels: int = 12,
    max_points: Optional[int] = None,
    title: str = "Combinatorial Fitness Landscape",
) -> str:
    points = loadCombinatorialFitness(
        path,
        only_feasible=only_feasible,
        max_points=max_points,
    )
    if not points:
        raise ValueError(f"No plottable combinatorial fitness records found: {path}")

    output_path = (
        Path(save_path)
        if save_path is not None
        else Path(path).with_suffix(".fitness3d.png")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    axis = fig.add_subplot(111, projection="3d")
    x_values = list(range(len(points)))
    scatter = axis.scatter(
        x_values,
        [point.sensor_count for point in points],
        [point.fitness for point in points],
        c=[point.coverage for point in points],
        s=marker_size,
        alpha=alpha,
        cmap="viridis",
    )
    axis.set_title(title)
    axis.set_xlabel("Sensor Position Set")
    axis.set_ylabel("Sensor Count")
    axis.set_zlabel("Fitness")

    tick_step = max(1, len(points) // max(1, max_xtick_labels))
    tick_indexes = x_values[::tick_step]
    axis.set_xticks(tick_indexes)
    axis.set_xticklabels(
        [_formatPositions(points[index].positions) for index in tick_indexes],
        rotation=60,
        ha="right",
        fontsize=7,
    )

    colorbar = fig.colorbar(scatter, ax=axis, shrink=0.72, pad=0.1)
    colorbar.set_label("Coverage (%)")
    fig.savefig(output_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return str(output_path)


def _formatPositions(positions: Tuple[Tuple[int, int], ...]) -> str:
    return ";".join(f"({x},{y})" for x, y in positions)
