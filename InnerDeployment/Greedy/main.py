from __future__ import annotations

import sys
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..fitnessfunction import FitnessFunc

try:
    from scipy import ndimage
except ImportError:  # pragma: no cover - container requirements include scipy.
    ndimage = None

Gene = Tuple[int, int]
Chromosome = List[Gene]


def to_int_pairs(points: Iterable[Sequence[int]]) -> List[Gene]:
    return [(int(p[0]), int(p[1])) for p in points]


class SensorGreedy:
    """
    Greedy inner-area sensor deployment.

    At each step, choose the installable candidate that covers the largest
    number of currently uncovered jobsite cells. Corner sensors are treated as
    already deployed, so the returned solution contains inner sensors only.
    """

    def __init__(
        self,
        *,
        installable_map,
        jobsite_map,
        coverage: int,
        corner_positions: List[Gene],
        min_sensors: int = 0,
        max_sensors: Optional[int] = None,
        candidate_stride: int = 1,
        fitness_kwargs: Optional[Dict] = None,
    ):
        self.installable_map = (np.asarray(installable_map) > 0)
        self.jobsite_map = (np.asarray(jobsite_map) > 0)

        if self.installable_map.shape != self.jobsite_map.shape:
            raise ValueError(
                "installable_map and jobsite_map must have the same shape. "
                f"Got {self.installable_map.shape} and {self.jobsite_map.shape}."
            )

        self.coverage = int(coverage)
        self.coverage_cells = int(self.coverage / 5)
        self.corner_positions = to_int_pairs(corner_positions)
        self.min_sensors = max(0, int(min_sensors))
        self.max_sensors = None if max_sensors is None else max(0, int(max_sensors))
        self.candidate_stride = max(1, int(candidate_stride))
        self.fitness_kwargs = dict(fitness_kwargs or {})

        self.best_solution: Chromosome = []
        self.best_fitness: float = float("nan")
        self.best_coverage: float = float("nan")
        self.corner_points: List[Gene] = list(self.corner_positions)

        self._height, self._width = self.jobsite_map.shape
        self._target_flat = self.jobsite_map.reshape(-1)
        self._target_area = int(self._target_flat.sum())
        self._offsets = self._circle_offsets(self.coverage_cells)
        self._kernel = self._circle_kernel(self.coverage_cells)
        self._candidate_mask = self._build_candidate_mask()

    @staticmethod
    def _circle_offsets(radius: int) -> np.ndarray:
        offsets = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if (dx * dx + dy * dy) <= (radius * radius):
                    offsets.append((dy, dx))
        return np.asarray(offsets, dtype=np.int32)

    @staticmethod
    def _circle_kernel(radius: int) -> np.ndarray:
        d = 2 * radius + 1
        yy, xx = np.ogrid[:d, :d]
        dist2 = (yy - radius) ** 2 + (xx - radius) ** 2
        return (dist2 <= radius * radius).astype(np.uint16)

    def _build_candidate_mask(self) -> np.ndarray:
        mask = self.installable_map.copy()
        for x, y in self.corner_positions:
            if 0 <= y < self._height and 0 <= x < self._width:
                mask[y, x] = False

        if self.candidate_stride > 1:
            ys, xs = np.where(mask)
            sampled = np.zeros_like(mask, dtype=bool)
            sampled[ys[:: self.candidate_stride], xs[:: self.candidate_stride]] = True
            mask = sampled

        return mask

    def _covered_indices(self, point: Gene) -> np.ndarray:
        x, y = int(point[0]), int(point[1])
        yy = y + self._offsets[:, 0]
        xx = x + self._offsets[:, 1]
        valid = (yy >= 0) & (yy < self._height) & (xx >= 0) & (xx < self._width)
        if not np.any(valid):
            return np.empty(0, dtype=np.int64)
        lin = (yy[valid] * self._width + xx[valid]).astype(np.int64, copy=False)
        return lin[self._target_flat[lin]]

    def _initial_covered(self) -> np.ndarray:
        covered = np.zeros(self._height * self._width, dtype=bool)
        for point in self.corner_positions:
            covered[self._covered_indices(point)] = True
        return covered

    def _coverage_percent(self, covered: np.ndarray) -> float:
        if self._target_area <= 0:
            return 0.0
        return float(100.0 * np.count_nonzero(covered & self._target_flat) / self._target_area)

    def _select_best_candidate(
        self,
        covered: np.ndarray,
        selected: set[Gene],
        evaluator: FitnessFunc,
        solution: Chromosome,
    ) -> Tuple[Optional[Gene], int, Optional[np.ndarray]]:
        if ndimage is None:
            return self._select_best_candidate_loop(covered, selected, evaluator, solution)

        uncovered = (self._target_flat & ~covered).reshape(self._height, self._width)
        gain_map = ndimage.convolve(
            uncovered.astype(np.uint16, copy=False),
            self._kernel,
            mode="constant",
            cval=0,
        )
        gain_map = np.where(self._candidate_mask, gain_map, 0)

        best_point: Optional[Gene] = None
        best_gain = 0
        best_score = float("-inf")
        coverage = self._coverage_percent(covered)
        ys, xs = np.where(gain_map > 0)
        for y, x in zip(ys.tolist(), xs.tolist()):
            point = (int(x), int(y))
            if point in selected:
                continue
            gain = int(gain_map[y, x])
            candidate_coverage = coverage + 100.0 * gain / max(1, self._target_area)
            score = evaluator.fitness_from_coverage(solution + [point], candidate_coverage)
            if score > best_score:
                best_point = point
                best_gain = gain
                best_score = score
        if best_point is None:
            return None, 0, None
        return best_point, best_gain, self._covered_indices(best_point)

    def _select_best_candidate_loop(
        self,
        covered: np.ndarray,
        selected: set[Gene],
        evaluator: FitnessFunc,
        solution: Chromosome,
    ) -> Tuple[Optional[Gene], int, Optional[np.ndarray]]:
        best_point: Optional[Gene] = None
        best_gain = 0
        best_score = float("-inf")
        best_indices: Optional[np.ndarray] = None
        coverage = self._coverage_percent(covered)

        ys, xs = np.where(self._candidate_mask)
        for y, x in zip(ys.tolist(), xs.tolist()):
            point = (int(x), int(y))
            if point in selected:
                continue
            indices = self._covered_indices(point)
            gain = int(np.count_nonzero(~covered[indices]))
            if gain <= 0:
                continue
            candidate_coverage = coverage + 100.0 * gain / max(1, self._target_area)
            score = evaluator.fitness_from_coverage(solution + [point], candidate_coverage)
            if score > best_score:
                best_point = point
                best_gain = gain
                best_score = score
                best_indices = indices

        return best_point, best_gain, best_indices

    def _greedy_recursive(
        self,
        *,
        covered: np.ndarray,
        selected: set[Gene],
        solution: Chromosome,
        evaluator: FitnessFunc,
        target_coverage: float,
        max_sensors: Optional[int],
        logger,
        verbose: bool,
        profile: bool,
        profile_every: int,
        start_time: float,
    ) -> Chromosome:
        coverage = self._coverage_percent(covered)
        fitness = evaluator.fitness_from_coverage(solution, coverage)
        self.best_solution = list(solution)
        self.best_coverage = coverage
        self.best_fitness = fitness

        gen_idx = len(solution)
        if logger is not None:
            logger.log_generation(
                gen=gen_idx,
                sensors_min=float(len(solution)),
                sensors_max=float(len(solution)),
                sensors_avg=float(len(solution)),
                fitness_min=float(fitness),
                fitness_max=float(fitness),
                fitness_avg=float(fitness),
                best_solution=solution,
                best_fitness=float(fitness),
                best_coverage=float(coverage),
            )

        if verbose:
            print(
                f"[Greedy {gen_idx:03d}] sensors={len(solution)} / "
                f"coverage={coverage:.2f}% (target={target_coverage:.2f}%)"
            )
        if profile and profile_every > 0 and gen_idx % int(profile_every) == 0:
            print(f"[Greedy {gen_idx:03d}] time={time.perf_counter() - start_time:.3f}s")

        if coverage >= float(target_coverage):
            return solution
        if max_sensors is not None and len(solution) >= int(max_sensors):
            return solution

        point, gain, indices = self._select_best_candidate(covered, selected, evaluator, solution)
        if point is None or indices is None or gain <= 0:
            return solution

        selected.add(point)
        solution.append(point)
        covered[indices] = True

        return self._greedy_recursive(
            covered=covered,
            selected=selected,
            solution=solution,
            evaluator=evaluator,
            target_coverage=target_coverage,
            max_sensors=max_sensors,
            logger=logger,
            verbose=verbose,
            profile=profile,
            profile_every=profile_every,
            start_time=start_time,
        )

    def run(
        self,
        *,
        target_coverage: float = 100.0,
        max_sensors: Optional[int] = None,
        return_best_only: bool = True,
        verbose: bool = True,
        profile: bool = False,
        profile_every: int = 1,
        logger=None,
        **_,
    ) -> Chromosome:
        del return_best_only
        t0 = time.perf_counter()
        limit = self.max_sensors if max_sensors is None else max(0, int(max_sensors))
        target = max(0.0, min(100.0, float(target_coverage)))
        fitness_kwargs = dict(self.fitness_kwargs)
        fitness_kwargs.setdefault("target_coverage", target)
        evaluator = FitnessFunc(
            jobsite_map=self.jobsite_map,
            corner_positions=self.corner_positions,
            coverage=self.coverage,
            **fitness_kwargs,
        )
        recursion_limit = limit if limit is not None else int(np.count_nonzero(self._candidate_mask))
        sys.setrecursionlimit(max(sys.getrecursionlimit(), int(recursion_limit) + 100))

        covered = self._initial_covered()
        selected = set(self.corner_positions)

        result = self._greedy_recursive(
            covered=covered,
            selected=selected,
            solution=[],
            evaluator=evaluator,
            target_coverage=target,
            max_sensors=limit,
            logger=logger,
            verbose=bool(verbose),
            profile=bool(profile),
            profile_every=max(1, int(profile_every)),
            start_time=t0,
        )

        while len(result) < self.min_sensors:
            point, gain, indices = self._select_best_candidate(covered, selected, evaluator, result)
            if point is None or indices is None or gain <= 0:
                break
            selected.add(point)
            result.append(point)
            covered[indices] = True

        self.best_solution = list(result)
        self.best_coverage = self._coverage_percent(covered)
        self.best_fitness = evaluator.fitness_from_coverage(result, self.best_coverage)
        self.corner_points = list(self.corner_positions)

        if verbose:
            print(
                f"[Greedy Final] inner={len(result)} / corner={len(self.corner_positions)} / "
                f"coverage={self.best_coverage:.2f}% / time={time.perf_counter() - t0:.3f}s"
            )

        return list(result)
