from __future__ import annotations

from functools import lru_cache
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .utils import is_far_enough, to_int_pairs

Gene = Tuple[int, int]


@lru_cache(maxsize=64)
def circle_offsets(radius: int) -> np.ndarray:
    radius = max(0, int(radius))
    offsets = [
        (dy, dx)
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
        if dx * dx + dy * dy <= radius * radius
    ]
    return np.asarray(offsets, dtype=np.int32)


@lru_cache(maxsize=64)
def circle_kernel(radius: int) -> np.ndarray:
    radius = max(0, int(radius))
    diameter = 2 * radius + 1
    yy, xx = np.ogrid[:diameter, :diameter]
    dist2 = (yy - radius) ** 2 + (xx - radius) ** 2
    return (dist2 <= radius * radius).astype(np.uint16)


def candidate_mask(
    installable_map,
    *,
    excluded: Iterable[Sequence[int]] = (),
    stride: int = 1,
) -> np.ndarray:
    mask = np.asarray(installable_map) > 0
    mask = mask.copy()
    height, width = mask.shape
    for x, y in to_int_pairs(excluded):
        if 0 <= x < width and 0 <= y < height:
            mask[y, x] = False

    stride = max(1, int(stride))
    if stride <= 1:
        return mask

    ys, xs = np.where(mask)
    sampled = np.zeros_like(mask, dtype=bool)
    sampled[ys[::stride], xs[::stride]] = True
    return sampled


def candidate_points(
    installable_map,
    *,
    excluded: Iterable[Sequence[int]] = (),
    stride: int = 1,
    base: Iterable[Sequence[int]] = (),
    min_separation: float = 0.0,
) -> List[Gene]:
    mask = candidate_mask(installable_map, excluded=excluded, stride=stride)
    base_points = to_int_pairs(base)
    points: List[Gene] = []
    for y, x in zip(*np.where(mask)):
        point = (int(x), int(y))
        if is_far_enough(point, base_points, min_separation):
            points.append(point)
    return points


def covered_indices(
    point: Sequence[int],
    *,
    offsets: np.ndarray,
    target_flat: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    x, y = int(point[0]), int(point[1])
    yy = y + offsets[:, 0]
    xx = x + offsets[:, 1]
    valid = (yy >= 0) & (yy < height) & (xx >= 0) & (xx < width)
    if not np.any(valid):
        return np.empty(0, dtype=np.int64)
    lin = (yy[valid] * width + xx[valid]).astype(np.int64, copy=False)
    return lin[target_flat[lin]]


def nearest_installable_indices(installable_map, *, excluded: Iterable[Sequence[int]] = ()):
    mask = candidate_mask(installable_map, excluded=excluded, stride=1).astype(np.uint8)
    try:
        from scipy.ndimage import distance_transform_edt
    except Exception:
        return None
    invalid = mask == 0
    _, indices = distance_transform_edt(invalid, return_indices=True)
    return indices[0], indices[1]


def snap_to_installable(
    x: float,
    y: float,
    *,
    width: int,
    height: int,
    installable_set: set[Gene],
    point_array: np.ndarray,
    nearest_y: Optional[np.ndarray] = None,
    nearest_x: Optional[np.ndarray] = None,
) -> Gene:
    xi = max(0, min(int(width) - 1, int(round(float(x)))))
    yi = max(0, min(int(height) - 1, int(round(float(y)))))
    if (xi, yi) in installable_set:
        return xi, yi
    if nearest_y is not None and nearest_x is not None:
        return int(nearest_x[yi, xi]), int(nearest_y[yi, xi])

    delta = point_array - np.asarray([xi, yi], dtype=np.float32)
    idx = int(np.argmin(np.einsum("ij,ij->i", delta, delta)))
    x2, y2 = point_array[idx]
    return int(x2), int(y2)
