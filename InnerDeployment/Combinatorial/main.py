from __future__ import annotations

import gc
import json
import math
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from ..geometry import candidate_points, circle_offsets, covered_indices
from ..utils import is_far_enough, min_separation_cells, to_int_pairs

Gene = Tuple[int, int]
Chromosome = List[Gene]


@dataclass(frozen=True, slots=True)
class CandidateCover:
    point: Gene
    bits: int


@dataclass(frozen=True, slots=True)
class SearchStats:
    candidates: int
    min_sensors: int
    max_sensors: int
    combinations: int


@dataclass(frozen=True, slots=True)
class FitnessParams:
    corner_positions: Tuple[Gene, ...]
    corner_count: int
    target_area: int
    target_coverage: float
    coverage_weight: float
    sensor_weight: float
    deficit_penalty: float
    overlap_min_dist: Optional[float]
    overlap_penalty: float
    min_separation: float


@dataclass(frozen=True, slots=True)
class CombinationTask:
    chunk: List[Tuple[CandidateCover, ...]]
    start_index: int
    corner_bits: int
    params: FitnessParams
    trace_enabled: bool = False


@dataclass(frozen=True, slots=True)
class FitnessTrace:
    index: int
    sensor_count: int
    positions: Tuple[Gene, ...]
    center_x: Optional[float]
    center_y: Optional[float]
    coverage: Optional[float]
    fitness: Optional[float]
    feasible: bool


@dataclass(frozen=True, slots=True)
class ChunkResult:
    evaluated: int
    best_fitness: float
    best_coverage: float
    best_solution: Chromosome
    traces: Tuple[FitnessTrace, ...] = ()


class CombinatorialFitnessLogger:
    def __init__(self, path: str | os.PathLike[str], metadata: Dict[str, Any]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8")
        self._write({"type": "meta", "metadata": metadata})

    def _write(self, payload: Dict[str, Any]) -> None:
        self._file.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def log(self, traces: Tuple[FitnessTrace, ...]) -> None:
        for trace in traces:
            self._write({"type": "fitness", **asdict(trace)})

    def close(self) -> None:
        self._file.close()


def _shutdownPool(pool: Optional[ProcessPoolExecutor], *, force: bool = False) -> None:
    if pool is None:
        return
    if force:
        for process in list((getattr(pool, "_processes", None) or {}).values()):
            if process.is_alive():
                process.terminate()
    pool.shutdown(wait=True, cancel_futures=True)


def _coveragePercent(bits: int, target_area: int) -> float:
    if target_area <= 0:
        return 0.0
    return float(100.0 * bits.bit_count() / target_area)


def _overlapCost(inner_positions: Chromosome, params: FitnessParams) -> float:
    if params.overlap_min_dist is None or params.overlap_penalty <= 0.0:
        return 0.0
    points = list(params.corner_positions) + list(inner_positions)
    if len(points) < 2:
        return 0.0

    min_dist2 = float(params.overlap_min_dist) ** 2
    overlaps = 0
    for i, first in enumerate(points):
        xi, yi = first
        for second in points[i + 1 :]:
            xj, yj = second
            dx = float(xi - xj)
            dy = float(yi - yj)
            if (dx * dx + dy * dy) < min_dist2:
                overlaps += 1
    return float(overlaps) * params.overlap_penalty


def _isSeparated(inner_positions: Chromosome, params: FitnessParams) -> bool:
    if params.min_separation <= 0.0:
        return True
    selected: Chromosome = []
    for point in inner_positions:
        if not is_far_enough(point, list(params.corner_positions) + selected, params.min_separation):
            return False
        selected.append(point)
    return True


def _fitnessScore(inner_positions: Chromosome, coverage: float, params: FitnessParams) -> float:
    total_count = float(params.corner_count + len(inner_positions))
    deficit = max(0.0, params.target_coverage - float(coverage))
    capped_coverage = min(float(coverage), params.target_coverage)
    return float(
        params.coverage_weight * capped_coverage
        - params.sensor_weight * total_count
        - params.deficit_penalty * deficit
        - _overlapCost(inner_positions, params)
    )


def _scoreCombinationChunk(task: CombinationTask) -> ChunkResult:
    best_solution: Chromosome = []
    best_fitness = float("-inf")
    best_coverage = 0.0
    traces: List[FitnessTrace] = []

    for offset, selected in enumerate(task.chunk):
        bits = task.corner_bits
        solution = [candidate.point for candidate in selected]
        if not _isSeparated(solution, task.params):
            if task.trace_enabled:
                traces.append(_makeTrace(task.start_index + offset, solution, None, None, False))
            continue
        for candidate in selected:
            bits |= candidate.bits
        coverage = _coveragePercent(bits, task.params.target_area)
        fitness = _fitnessScore(solution, coverage, task.params)
        if task.trace_enabled:
            traces.append(_makeTrace(task.start_index + offset, solution, coverage, fitness, True))
        if fitness > best_fitness:
            best_solution = solution
            best_fitness = float(fitness)
            best_coverage = float(coverage)

    return ChunkResult(
        evaluated=len(task.chunk),
        best_fitness=best_fitness,
        best_coverage=best_coverage,
        best_solution=best_solution,
        traces=tuple(traces),
    )


def _makeTrace(
    index: int,
    solution: Chromosome,
    coverage: Optional[float],
    fitness: Optional[float],
    feasible: bool,
) -> FitnessTrace:
    center_x, center_y = _centerPoint(solution)
    return FitnessTrace(
        index=int(index),
        sensor_count=len(solution),
        positions=tuple(solution),
        center_x=center_x,
        center_y=center_y,
        coverage=None if coverage is None else float(coverage),
        fitness=None if fitness is None else float(fitness),
        feasible=bool(feasible),
    )


def _centerPoint(solution: Chromosome) -> Tuple[Optional[float], Optional[float]]:
    if not solution:
        return None, None
    count = float(len(solution))
    return (
        float(sum(point[0] for point in solution) / count),
        float(sum(point[1] for point in solution) / count),
    )


def defaultParallelWorkers() -> int:
    return min(16, max(1, (os.cpu_count() or 2) - 1))


class SensorCombinatorial:
    """
    Exact inner-area sensor deployment over a finite candidate domain.

    This optimizer enumerates every candidate subset in the configured finite
    candidate domain. It returns the global optimum for that explicit domain
    under FitnessFunc.fitness_from_coverage().
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
        max_candidates: Optional[int] = 24,
        max_combinations: Optional[int] = 5_000_000,
        min_separation: Optional[float] = None,
        parallel_workers: Optional[int] = None,
        chunk_size: int = 4096,
        fitness_kwargs: Optional[Dict] = None,
    ):
        self.installable_map = np.asarray(installable_map) > 0
        self.jobsite_map = np.asarray(jobsite_map) > 0
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
        self.max_candidates = None if max_candidates is None else max(0, int(max_candidates))
        self.max_combinations = None if max_combinations is None else max(0, int(max_combinations))
        self.min_separation = min_separation_cells(min_separation, self.coverage)
        if parallel_workers is None:
            self.parallel_workers = defaultParallelWorkers()
        else:
            self.parallel_workers = max(1, int(parallel_workers))
        self.chunk_size = max(1, int(chunk_size))
        self.fitness_kwargs = dict(fitness_kwargs or {})

        self.best_solution: Chromosome = []
        self.best_fitness: float = float("-inf")
        self.best_coverage: float = 0.0
        self.corner_points: List[Gene] = list(self.corner_positions)
        self.search_stats: Optional[SearchStats] = None

        self._height, self._width = self.jobsite_map.shape
        self._target_flat = self.jobsite_map.reshape(-1)
        self._target_area = int(self._target_flat.sum())
        self._offsets = circle_offsets(self.coverage_cells)
        self._target_index = self._buildTargetIndex()
        self._corner_bits = self._buildCornerBits()

    def _buildTargetIndex(self) -> Dict[int, int]:
        target_indices = np.flatnonzero(self._target_flat)
        return {int(index): offset for offset, index in enumerate(target_indices.tolist())}

    def _buildBits(self, indices: np.ndarray) -> int:
        bits = 0
        for index in indices.tolist():
            offset = self._target_index.get(int(index))
            if offset is not None:
                bits |= 1 << offset
        return bits

    def _buildCornerBits(self) -> int:
        bits = 0
        for point in self.corner_positions:
            bits |= self._buildBits(self._coveredIndices(point))
        return bits

    def _coveredIndices(self, point: Gene) -> np.ndarray:
        return covered_indices(
            point,
            offsets=self._offsets,
            target_flat=self._target_flat,
            width=self._width,
            height=self._height,
        )

    def _candidatePoints(self) -> List[Gene]:
        points = candidate_points(
            self.installable_map,
            excluded=self.corner_positions,
            stride=self.candidate_stride,
            base=self.corner_positions,
            min_separation=self.min_separation,
        )
        if self.max_candidates is None or len(points) <= self.max_candidates:
            return points
        return self._rankCandidates(points)[: self.max_candidates]

    def _rankCandidates(self, points: List[Gene]) -> List[Gene]:
        scored: List[Tuple[int, Gene]] = []
        for point in points:
            bits = self._buildBits(self._coveredIndices(point))
            scored.append((bits.bit_count(), point))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [point for _, point in scored]

    def _candidateCovers(self) -> List[CandidateCover]:
        covers: List[CandidateCover] = []
        for point in self._candidatePoints():
            bits = self._buildBits(self._coveredIndices(point))
            covers.append(CandidateCover(point=point, bits=bits))
        return covers

    def _bounds(self, count: int, max_sensors: Optional[int]) -> Tuple[int, int]:
        upper = self.max_sensors if max_sensors is None else max(0, int(max_sensors))
        if upper is None:
            upper = count
        return min(self.min_sensors, count), min(int(upper), count)

    def _domainSize(self, count: int, min_count: int, max_count: int) -> int:
        return int(sum(math.comb(count, size) for size in range(min_count, max_count + 1)))

    def _validateDomain(self, stats: SearchStats) -> None:
        if stats.min_sensors > stats.max_sensors:
            raise ValueError(
                "Combinatorial search domain is empty. "
                f"min_sensors={stats.min_sensors}, max_sensors={stats.max_sensors}."
            )
        if self.max_combinations is None:
            return
        if stats.combinations <= self.max_combinations:
            return
        raise ValueError(
            "Combinatorial search domain is too large for the configured limit. "
            f"candidates={stats.candidates}, sensors={stats.min_sensors}-{stats.max_sensors}, "
            f"combinations={stats.combinations}, max_combinations={self.max_combinations}. "
            "Increase candidate_stride, lower max_candidates, or set max_combinations=None."
        )

    def _fitnessParams(self, target_coverage: float) -> FitnessParams:
        overlap_min_dist = self.fitness_kwargs.get("overlap_min_dist", 15.0)
        return FitnessParams(
            corner_positions=tuple(self.corner_positions),
            corner_count=len(self.corner_positions),
            target_area=self._target_area,
            target_coverage=float(target_coverage),
            coverage_weight=float(self.fitness_kwargs.get("coverage_weight", 1.0)),
            sensor_weight=float(self.fitness_kwargs.get("sensor_weight", 1.0)),
            deficit_penalty=float(self.fitness_kwargs.get("deficit_penalty", 20.0)),
            overlap_min_dist=None if overlap_min_dist is None else float(overlap_min_dist),
            overlap_penalty=float(self.fitness_kwargs.get("overlap_penalty", 5.0)),
            min_separation=float(self.min_separation),
        )

    def _combinationChunks(
        self,
        covers: List[CandidateCover],
        min_count: int,
        max_count: int,
    ) -> Iterable[List[Tuple[CandidateCover, ...]]]:
        chunk: List[Tuple[CandidateCover, ...]] = []
        for size in range(min_count, max_count + 1):
            for selected in combinations(covers, size):
                chunk.append(selected)
                if len(chunk) >= self.chunk_size:
                    yield chunk
                    chunk = []
        if chunk:
            yield chunk

    def _logProgress(
        self,
        *,
        logger,
        generation: int,
        solution: Chromosome,
        fitness: float,
        coverage: float,
    ) -> None:
        if logger is None:
            return
        sensors = float(len(solution))
        logger.log_generation(
            gen=int(generation),
            sensors_min=sensors,
            sensors_max=sensors,
            sensors_avg=sensors,
            fitness_min=float(fitness),
            fitness_max=float(fitness),
            fitness_avg=float(fitness),
            best_solution=solution,
            best_fitness=float(fitness),
            best_coverage=float(coverage),
        )

    def _consumeResult(
        self,
        *,
        result: ChunkResult,
        evaluated: int,
        best_solution: Chromosome,
        best_fitness: float,
        best_coverage: float,
        logger,
    ) -> Tuple[Chromosome, float, float]:
        if result.best_fitness <= best_fitness:
            return best_solution, best_fitness, best_coverage
        next_solution = list(result.best_solution)
        self._logProgress(
            logger=logger,
            generation=evaluated,
            solution=next_solution,
            fitness=result.best_fitness,
            coverage=result.best_coverage,
        )
        return next_solution, float(result.best_fitness), float(result.best_coverage)

    def _logTrace(
        self,
        *,
        trace_logger: Optional[CombinatorialFitnessLogger],
        result: ChunkResult,
    ) -> None:
        if trace_logger is None:
            return
        trace_logger.log(result.traces)

    def _runSequential(
        self,
        *,
        covers: List[CandidateCover],
        min_count: int,
        max_count: int,
        domain_size: int,
        params: FitnessParams,
        logger,
        trace_logger: Optional[CombinatorialFitnessLogger],
        profile: bool,
        profile_every: int,
        start: float,
    ) -> Tuple[Chromosome, float, float, int]:
        evaluated = 0
        best_solution: Chromosome = []
        best_fitness = float("-inf")
        best_coverage = _coveragePercent(self._corner_bits, self._target_area)

        for chunk in self._combinationChunks(covers, min_count, max_count):
            result = _scoreCombinationChunk(
                CombinationTask(
                    chunk=chunk,
                    start_index=evaluated,
                    corner_bits=self._corner_bits,
                    params=params,
                    trace_enabled=trace_logger is not None,
                )
            )
            evaluated += result.evaluated
            self._logTrace(trace_logger=trace_logger, result=result)
            best_solution, best_fitness, best_coverage = self._consumeResult(
                result=result,
                evaluated=evaluated,
                best_solution=best_solution,
                best_fitness=best_fitness,
                best_coverage=best_coverage,
                logger=logger,
            )
            should_print = (
                profile
                and profile_every > 0
                and evaluated % int(profile_every) < result.evaluated
            )
            if should_print:
                self._printProgress(
                    evaluated,
                    domain_size,
                    best_solution,
                    best_coverage,
                    start,
                )

        return best_solution, best_fitness, best_coverage, evaluated

    def _runParallel(
        self,
        *,
        covers: List[CandidateCover],
        min_count: int,
        max_count: int,
        domain_size: int,
        params: FitnessParams,
        logger,
        trace_logger: Optional[CombinatorialFitnessLogger],
        profile: bool,
        profile_every: int,
        start: float,
    ) -> Tuple[Chromosome, float, float, int]:
        pool: Optional[ProcessPoolExecutor] = None
        evaluated = 0
        best_solution: Chromosome = []
        best_fitness = float("-inf")
        best_coverage = _coveragePercent(self._corner_bits, self._target_area)
        pending: Set = set()
        chunks = iter(self._combinationChunks(covers, min_count, max_count))
        max_pending = max(1, self.parallel_workers * 2)
        next_start = 0

        def submitNext() -> bool:
            nonlocal next_start
            try:
                chunk = next(chunks)
            except StopIteration:
                return False
            task = CombinationTask(
                chunk=chunk,
                start_index=next_start,
                corner_bits=self._corner_bits,
                params=params,
                trace_enabled=trace_logger is not None,
            )
            next_start += len(chunk)
            pending.add(pool.submit(_scoreCombinationChunk, task))
            return True

        try:
            pool = ProcessPoolExecutor(max_workers=self.parallel_workers)
            for _ in range(max_pending):
                if not submitNext():
                    break

            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    result = future.result()
                    evaluated += result.evaluated
                    self._logTrace(trace_logger=trace_logger, result=result)
                    best_solution, best_fitness, best_coverage = self._consumeResult(
                        result=result,
                        evaluated=evaluated,
                        best_solution=best_solution,
                        best_fitness=best_fitness,
                        best_coverage=best_coverage,
                        logger=logger,
                    )
                    should_print = (
                        profile
                        and profile_every > 0
                        and evaluated % int(profile_every) < result.evaluated
                    )
                    if should_print:
                        self._printProgress(
                            evaluated,
                            domain_size,
                            best_solution,
                            best_coverage,
                            start,
                        )
                    submitNext()
        except BaseException:
            _shutdownPool(pool, force=True)
            raise
        else:
            _shutdownPool(pool, force=False)

        return best_solution, best_fitness, best_coverage, evaluated

    def _printProgress(
        self,
        evaluated: int,
        domain_size: int,
        best_solution: Chromosome,
        best_coverage: float,
        start: float,
    ) -> None:
        print(
            f"[Combinatorial {evaluated}/{domain_size}] "
            f"best_inner={len(best_solution)} / best_cov={best_coverage:.2f}% / "
            f"time={time.perf_counter() - start:.3f}s"
        )

    def run(
        self,
        *,
        target_coverage: float = 100.0,
        max_sensors: Optional[int] = None,
        return_best_only: bool = True,
        verbose: bool = True,
        profile: bool = False,
        profile_every: int = 100_000,
        parallel_workers: Optional[int] = None,
        chunk_size: Optional[int] = None,
        fitness_log_path: Optional[str] = None,
        logger=None,
        **_,
    ) -> Chromosome:
        del return_best_only
        if parallel_workers is not None:
            self.parallel_workers = max(1, int(parallel_workers))
        if chunk_size is not None:
            self.chunk_size = max(1, int(chunk_size))

        start = time.perf_counter()
        covers = self._candidateCovers()
        min_count, max_count = self._bounds(len(covers), max_sensors)
        domain_size = self._domainSize(len(covers), min_count, max_count)
        self.search_stats = SearchStats(
            candidates=len(covers),
            min_sensors=min_count,
            max_sensors=max_count,
            combinations=domain_size,
        )
        self._validateDomain(self.search_stats)

        effective_target = float(
            self.fitness_kwargs.get("target_coverage", target_coverage)
        )
        params = self._fitnessParams(effective_target)
        trace_logger: Optional[CombinatorialFitnessLogger] = None
        if fitness_log_path is not None:
            trace_logger = CombinatorialFitnessLogger(
                fitness_log_path,
                metadata={
                    "target_coverage": effective_target,
                    "coverage": self.coverage,
                    "candidates": len(covers),
                    "min_sensors": min_count,
                    "max_sensors": max_count,
                    "combinations": domain_size,
                    "parallel_workers": self.parallel_workers,
                    "chunk_size": self.chunk_size,
                },
            )

        if verbose:
            print(
                "[Combinatorial Start] "
                f"candidates={len(covers)} / sensors={min_count}-{max_count} / "
                f"domain={domain_size} / workers={self.parallel_workers} / "
                f"chunk={self.chunk_size}"
            )

        try:
            if self.parallel_workers <= 1:
                best_solution, best_fitness, best_coverage, evaluated = self._runSequential(
                    covers=covers,
                    min_count=min_count,
                    max_count=max_count,
                    domain_size=domain_size,
                    params=params,
                    logger=logger,
                    trace_logger=trace_logger,
                    profile=bool(profile),
                    profile_every=max(1, int(profile_every)),
                    start=start,
                )
            else:
                best_solution, best_fitness, best_coverage, evaluated = self._runParallel(
                    covers=covers,
                    min_count=min_count,
                    max_count=max_count,
                    domain_size=domain_size,
                    params=params,
                    logger=logger,
                    trace_logger=trace_logger,
                    profile=bool(profile),
                    profile_every=max(1, int(profile_every)),
                    start=start,
                )
        finally:
            if trace_logger is not None:
                trace_logger.close()

        self.best_solution = list(best_solution)
        self.best_fitness = float(best_fitness)
        self.best_coverage = float(best_coverage)
        self.corner_points = list(self.corner_positions)

        if verbose:
            print(
                "[Combinatorial Final] "
                f"inner={len(self.best_solution)} / corner={len(self.corner_positions)} / "
                f"coverage={self.best_coverage:.2f}% / fitness={self.best_fitness:.3f} / "
                f"evaluated={evaluated} / time={time.perf_counter() - start:.3f}s"
            )

        return list(self.best_solution)

    def close(self) -> None:
        self.installable_map = np.empty((0, 0), dtype=bool)
        self.jobsite_map = np.empty((0, 0), dtype=bool)
        self._target_flat = np.empty((0,), dtype=bool)
        self._offsets = np.empty((0, 2), dtype=np.int32)
        self._target_index = {}
        self.fitness_kwargs.clear()
        gc.collect()
