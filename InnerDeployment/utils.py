from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np

Gene = Tuple[int, int]


def to_int_pairs(points: Iterable[Sequence[int]]) -> List[Gene]:
    return [(int(p[0]), int(p[1])) for p in points]


def to_bool_map(m) -> np.ndarray:
    return np.asarray(m) > 0


def min_separation_cells(value: float | None, coverage: int) -> float:
    if value is None:
        return max(0.0, float(coverage) / 5.0)
    return max(0.0, float(value))


def is_far_enough(point: Sequence[int], existing: Iterable[Sequence[int]], min_separation: float) -> bool:
    if float(min_separation) <= 0.0:
        return True
    x, y = int(point[0]), int(point[1])
    min_d2 = float(min_separation) ** 2
    for other in existing:
        ox, oy = int(other[0]), int(other[1])
        dx = float(x - ox)
        dy = float(y - oy)
        if (dx * dx + dy * dy) < min_d2:
            return False
    return True


def filter_min_separation(
    points: Iterable[Sequence[int]],
    *,
    base: Iterable[Sequence[int]] = (),
    min_separation: float = 0.0,
) -> tuple[List[Gene], int]:
    kept: List[Gene] = []
    occupied = to_int_pairs(base)
    removed = 0

    for point in points:
        key = (int(point[0]), int(point[1]))
        if key in kept or key in occupied:
            removed += 1
            continue
        if not is_far_enough(key, occupied + kept, min_separation):
            removed += 1
            continue
        kept.append(key)

    return kept, removed
