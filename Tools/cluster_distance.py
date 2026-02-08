"""
평균 군집거리(센서들 간 평균 거리) 계산 모듈.

포인트 집합에 대해 쌍별 유클리드 거리의 평균을 구합니다.
run JSON의 best_solution, corner_points 등 다양한 포인트 형식을 지원합니다.
"""

from __future__ import annotations

import math
import re
from typing import Any, List, Tuple, Union

Point = Tuple[float, float]


def _parse_one(p: Any) -> Point | None:
    """단일 포인트를 (x, y) 튜플로 변환. 실패 시 None."""
    if p is None:
        return None
    if isinstance(p, (list, tuple)) and len(p) >= 2:
        try:
            return (float(p[0]), float(p[1]))
        except (TypeError, ValueError):
            return None
    if isinstance(p, str):
        m = re.match(r"\(\s*([\d.e+-]+)\s*,\s*([\d.e+-]+)\s*\)", p.strip())
        if m:
            try:
                return (float(m.group(1)), float(m.group(2)))
            except ValueError:
                return None
    return None


def as_points(points: Any) -> List[Point]:
    """
    다양한 형식의 포인트 목록을 [(x, y), ...] 로 통일합니다.

    지원 형식:
      - [[x, y], ...] 또는 [(x, y), ...]
      - ["(x,y)", ...] (Logger tuple_str)
      - [x, y, x, y, ...] (flat) 은 지원하지 않음; list/tuple 쌍만 처리

    Returns:
        변환된 (x, y) 리스트. 파싱 실패한 항목은 건너뜀.
    """
    if points is None:
        return []
    out: List[Point] = []
    for p in points:
        q = _parse_one(p)
        if q is not None:
            out.append(q)
    return out


def distance(p: Point, q: Point) -> float:
    """두 점 사이 유클리드 거리."""
    return math.hypot(q[0] - p[0], q[1] - p[1])


def pairwise_distances(points: List[Point]) -> List[float]:
    """
    포인트 집합에서 모든 쌍(i < j)에 대한 거리 리스트를 반환합니다.

    Args:
        points: [(x, y), ...]

    Returns:
        쌍별 거리 리스트. 길이 = n*(n-1)//2 (n = len(points))
    """
    n = len(points)
    if n < 2:
        return []
    dists: List[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(distance(points[i], points[j]))
    return dists


def mean_cluster_distance(points: Any) -> float:
    """
    포인트 집합의 평균 군집거리(모든 쌍별 거리의 평균)를 반환합니다.

    Args:
        points: as_points()가 지원하는 임의 형식의 포인트 목록.

    Returns:
        평균 쌍별 거리. 포인트가 2개 미만이면 0.0.
    """
    pts = as_points(points)
    dists = pairwise_distances(pts)
    if not dists:
        return 0.0
    return sum(dists) / len(dists)


def nearest_neighbor_distances(points: List[Point]) -> List[float]:
    """
    각 포인트에서 가장 가까운 다른 포인트까지의 거리 리스트를 반환합니다.

    Args:
        points: [(x, y), ...]

    Returns:
        길이 n 리스트. i번째 원소 = i번째 점에서 (자신 제외) 가장 가까운 점까지의 거리.
        포인트가 2개 미만이면 [].
    """
    n = len(points)
    if n < 2:
        return []
    result: List[float] = []
    for i in range(n):
        min_d = float("inf")
        for j in range(n):
            if i == j:
                continue
            d = distance(points[i], points[j])
            if d < min_d:
                min_d = d
        result.append(min_d)
    return result


def mean_nearest_neighbor_distance(points: Any) -> float:
    """
    각 포인트에서 가장 가까운 센서(다른 포인트)까지의 거리들의 평균을 반환합니다.

    수식: (1/n) * Σ_i min_{j≠i} d(p_i, p_j)

    Args:
        points: as_points()가 지원하는 임의 형식의 포인트 목록.

    Returns:
        평균 최근접 이웃 거리. 포인트가 2개 미만이면 0.0.
    """
    pts = as_points(points)
    nn_dists = nearest_neighbor_distances(pts)
    if not nn_dists:
        return 0.0
    return sum(nn_dists) / len(nn_dists)


def mean_nearest_neighbor_stats_m(
    points: Any,
    grid_m: float = 5.0,
) -> dict:
    """
    각 점에서 가장 가까운 이웃까지의 거리 통계를 실제 거리(m)로 반환.
    grid_m: 1그리드당 미터.

    Returns:
        {"mean_m", "std_m", "min_m", "max_m", "n_points"} 또는 n_points<2일 때 mean_m=0 등.
    """
    pts = as_points(points)
    nn_dists = nearest_neighbor_distances(pts)
    n = len(nn_dists)
    if n == 0:
        return {"mean_m": 0.0, "std_m": 0.0, "min_m": 0.0, "max_m": 0.0, "n_points": len(pts)}

    mean_d = sum(nn_dists) / n
    std_d = (
        (sum((d - mean_d) ** 2 for d in nn_dists) / (n - 1)) ** 0.5 if n > 1 else 0.0
    )
    return {
        "mean_m": mean_d * grid_m,
        "std_m": std_d * grid_m,
        "min_m": min(nn_dists) * grid_m,
        "max_m": max(nn_dists) * grid_m,
        "n_points": len(pts),
    }


def cluster_distance_stats(points: Any) -> dict:
    """
    포인트 집합에 대한 군집거리 통계를 반환합니다.

    Args:
        points: as_points()가 지원하는 임의 형식의 포인트 목록.

    Returns:
        {
            "mean": float,   # 평균 쌍별 거리
            "std": float,   # 쌍별 거리 표준편차 (쌍이 1개 미만이면 0)
            "min": float,   # 최소 쌍별 거리
            "max": float,   # 최대 쌍별 거리
            "n_points": int,
            "n_pairs": int,
        }
    """
    pts = as_points(points)
    dists = pairwise_distances(pts)
    n_points = len(pts)
    n_pairs = len(dists)

    if n_pairs == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "n_points": n_points,
            "n_pairs": 0,
        }

    mean_d = sum(dists) / n_pairs
    if n_pairs == 1:
        std_d = 0.0
    else:
        var = sum((d - mean_d) ** 2 for d in dists) / (n_pairs - 1)
        std_d = math.sqrt(var)

    return {
        "mean": mean_d,
        "std": std_d,
        "min": min(dists),
        "max": max(dists),
        "n_points": n_points,
        "n_pairs": n_pairs,
    }
