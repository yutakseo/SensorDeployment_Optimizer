from __future__ import annotations

import gc
import json
import math
import os
import random
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from ..geometry import candidate_points, circle_offsets, covered_indices
from ..utils import is_far_enough, min_separation_cells, to_int_pairs

try:
    import torch
except ImportError:  # pragma: no cover - torch is optional for CPU-only installs.
    torch = None

Gene = Tuple[int, int]
Chromosome = List[Gene]
MIN_COVERAGE_PRIORITY_MARGIN = 1.0
DEFAULT_GPU_MAX_CELLS = 50_000_000


@dataclass(frozen=True, slots=True)
class CandidateCover:
    point: Gene
    bits: int
    indices: Tuple[int, ...]


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
class AdaptiveParams:
    start_sensors: int
    samples_per_count: int
    intensify_samples: int
    patience: int
    min_delta: float
    regress_delta: float
    uniform_ratio: float
    weighted_ratio: float
    local_ratio: float


@dataclass(frozen=True, slots=True)
class CombinationTask:
    chunk: List[Tuple[CandidateCover, ...]]
    start_index: int
    corner_bits: int
    params: FitnessParams
    trace_enabled: bool = False
    trace_stride: int = 1


@dataclass(frozen=True, slots=True)
class FitnessTrace:
    index: int
    sensor_count: int
    positions: Tuple[Gene, ...]
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
        index = task.start_index + offset
        should_trace = task.trace_enabled and index % task.trace_stride == 0
        bits = task.corner_bits
        solution = [candidate.point for candidate in selected]
        if not _isSeparated(solution, task.params):
            if should_trace:
                traces.append(_makeTrace(index, solution, None, None, False))
            continue
        for candidate in selected:
            bits |= candidate.bits
        coverage = _coveragePercent(bits, task.params.target_area)
        fitness = _fitnessScore(solution, coverage, task.params)
        is_chunk_best = fitness > best_fitness
        if should_trace or (task.trace_enabled and is_chunk_best):
            traces.append(_makeTrace(index, solution, coverage, fitness, True))
        if is_chunk_best:
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
    return FitnessTrace(
        index=int(index),
        sensor_count=len(solution),
        positions=tuple(solution),
        coverage=None if coverage is None else float(coverage),
        fitness=None if fitness is None else float(fitness),
        feasible=bool(feasible),
    )


def defaultParallelWorkers() -> int:
    return max(1, (os.cpu_count() or 2) - 4)


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
        self._corner_indices = self._buildCornerIndices()
        self._corner_bits = self._buildCornerBits()

    def _buildTargetIndex(self) -> Dict[int, int]:
        target_indices = np.flatnonzero(self._target_flat)
        return {int(index): offset for offset, index in enumerate(target_indices.tolist())}

    def _buildBits(self, indices: np.ndarray) -> int:
        return self._buildBitsFromOffsets(self._targetOffsets(indices))

    def _buildBitsFromOffsets(self, offsets: Iterable[int]) -> int:
        bits = 0
        for offset in offsets:
            bits |= 1 << int(offset)
        return bits

    def _targetOffsets(self, indices: np.ndarray) -> Tuple[int, ...]:
        offsets: List[int] = []
        for index in indices.tolist():
            offset = self._target_index.get(int(index))
            if offset is not None:
                offsets.append(int(offset))
        return tuple(offsets)

    def _buildCornerIndices(self) -> Tuple[int, ...]:
        offsets: Set[int] = set()
        for point in self.corner_positions:
            offsets.update(self._targetOffsets(self._coveredIndices(point)))
        return tuple(sorted(offsets))

    def _buildCornerBits(self) -> int:
        return self._buildBitsFromOffsets(self._corner_indices)

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
            indices = self._targetOffsets(self._coveredIndices(point))
            bits = self._buildBitsFromOffsets(indices)
            covers.append(CandidateCover(point=point, bits=bits, indices=indices))
        return covers

    def _bounds(self, count: int, max_sensors: Optional[int]) -> Tuple[int, int]:
        upper = self.max_sensors if max_sensors is None else max(0, int(max_sensors))
        if upper is None:
            upper = count
        return min(self.min_sensors, count), min(int(upper), count)

    def _domainSize(self, count: int, min_count: int, max_count: int) -> int:
        return int(sum(math.comb(count, size) for size in range(min_count, max_count + 1)))

    def _validateDomain(
        self,
        stats: SearchStats,
        sample_combinations: Optional[int],
    ) -> None:
        if stats.min_sensors > stats.max_sensors:
            raise ValueError(
                "Combinatorial search domain is empty. "
                f"min_sensors={stats.min_sensors}, max_sensors={stats.max_sensors}."
            )
        if self.max_combinations is None:
            return
        evaluated_limit = (
            stats.combinations
            if sample_combinations is None
            else min(int(sample_combinations), stats.combinations)
        )
        if evaluated_limit <= self.max_combinations:
            return
        raise ValueError(
            "Combinatorial search domain is too large for the configured limit. "
            f"candidates={stats.candidates}, sensors={stats.min_sensors}-{stats.max_sensors}, "
            f"evaluated_limit={evaluated_limit}, max_combinations={self.max_combinations}. "
            "Increase candidate_stride, lower max_candidates, or set max_combinations=None."
        )

    def _defaultDeficitPenalty(self, max_inner_sensors: int) -> float:
        if self._target_area <= 0:
            return float(self.fitness_kwargs.get("deficit_penalty", 20.0))

        sensor_weight = float(self.fitness_kwargs.get("sensor_weight", 1.0))
        coverage_weight = float(self.fitness_kwargs.get("coverage_weight", 1.0))
        max_total_sensors = max(1, int(max_inner_sensors) + len(self.corner_positions))
        min_coverage_step = 100.0 / float(self._target_area)
        required_weight = sensor_weight * float(max_total_sensors) / min_coverage_step
        return max(20.0, required_weight - coverage_weight + MIN_COVERAGE_PRIORITY_MARGIN)

    def _fitnessParams(self, target_coverage: float, max_inner_sensors: int) -> FitnessParams:
        overlap_min_dist = self.fitness_kwargs.get("overlap_min_dist", 15.0)
        deficit_penalty = self.fitness_kwargs.get(
            "deficit_penalty",
            self._defaultDeficitPenalty(max_inner_sensors),
        )
        return FitnessParams(
            corner_positions=tuple(self.corner_positions),
            corner_count=len(self.corner_positions),
            target_area=self._target_area,
            target_coverage=float(target_coverage),
            coverage_weight=float(self.fitness_kwargs.get("coverage_weight", 1.0)),
            sensor_weight=float(self.fitness_kwargs.get("sensor_weight", 1.0)),
            deficit_penalty=float(deficit_penalty),
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

    def _searchChunks(
        self,
        *,
        covers: List[CandidateCover],
        min_count: int,
        max_count: int,
        domain_size: int,
        sample_combinations: Optional[int],
        sample_seed: int,
    ) -> Iterable[List[Tuple[CandidateCover, ...]]]:
        if sample_combinations is None or int(sample_combinations) >= domain_size:
            yield from self._combinationChunks(covers, min_count, max_count)
            return
        yield from self._sampledCombinationChunks(
            covers=covers,
            min_count=min_count,
            max_count=max_count,
            sample_combinations=int(sample_combinations),
            sample_seed=int(sample_seed),
        )

    def _sampledCombinationChunks(
        self,
        *,
        covers: List[CandidateCover],
        min_count: int,
        max_count: int,
        sample_combinations: int,
        sample_seed: int,
    ) -> Iterable[List[Tuple[CandidateCover, ...]]]:
        count = len(covers)
        rng = random.Random(sample_seed)
        sizes = list(range(min_count, max_count + 1))
        weights = [self._sampleWeight(count, size) for size in sizes]
        sampled: Set[Tuple[int, ...]] = set()
        chunk: List[Tuple[CandidateCover, ...]] = []

        for indices in self._seedSampleIndices(count, min_count, max_count):
            if len(sampled) >= sample_combinations:
                break
            sampled.add(indices)
            chunk.append(tuple(covers[index] for index in indices))
            if len(chunk) >= self.chunk_size:
                yield chunk
                chunk = []

        max_attempts = max(sample_combinations * 20, sample_combinations + 1000)
        attempts = 0
        while len(sampled) < sample_combinations and attempts < max_attempts:
            attempts += 1
            size = rng.choices(sizes, weights=weights, k=1)[0]
            indices = tuple(sorted(rng.sample(range(count), size)))
            if indices in sampled:
                continue
            sampled.add(indices)
            chunk.append(tuple(covers[index] for index in indices))
            if len(chunk) >= self.chunk_size:
                yield chunk
                chunk = []

        if chunk:
            yield chunk

    def _seedSampleIndices(
        self,
        count: int,
        min_count: int,
        max_count: int,
    ) -> Iterable[Tuple[int, ...]]:
        if min_count <= 0 <= max_count:
            yield ()
        if min_count <= 1 <= max_count:
            for index in range(count):
                yield (index,)
        if min_count <= 2 <= max_count:
            for first in range(count):
                for second in range(first + 1, count):
                    yield (first, second)

    def _sampleWeight(self, count: int, size: int) -> float:
        combinations_count = math.comb(count, size)
        return max(1.0, math.sqrt(float(combinations_count)))

    def _adaptiveBudgetLimit(
        self,
        *,
        min_count: int,
        max_count: int,
        params: AdaptiveParams,
        sample_limit: Optional[int],
        refine_samples: int,
        refine_rounds: int,
    ) -> int:
        count_span = max(1, max_count - min_count + 1)
        base_budget = count_span * params.samples_per_count
        intensify_budget = max(1, params.patience + 1) * params.intensify_samples
        refine_budget = max(0, int(refine_samples)) * max(0, int(refine_rounds)) * 3
        budget = base_budget + intensify_budget + refine_budget
        if sample_limit is None:
            return budget
        return min(int(sample_limit), budget)

    def _adaptiveParams(
        self,
        *,
        min_count: int,
        max_count: int,
        adaptive_start_sensors: Optional[int],
        adaptive_samples_per_count: int,
        adaptive_intensify_samples: int,
        adaptive_patience: int,
        adaptive_min_delta: float,
        adaptive_regress_delta: float,
        adaptive_uniform_ratio: float,
        adaptive_weighted_ratio: float,
        adaptive_local_ratio: float,
    ) -> AdaptiveParams:
        default_start = max(min_count, min(max_count, 4))
        start = default_start if adaptive_start_sensors is None else int(adaptive_start_sensors)
        start = max(min_count, min(max_count, start))
        uniform = max(0.0, float(adaptive_uniform_ratio))
        weighted = max(0.0, float(adaptive_weighted_ratio))
        local = max(0.0, float(adaptive_local_ratio))
        total = uniform + weighted + local
        if total <= 0.0:
            uniform, weighted, local = 1.0, 0.0, 0.0
            total = 1.0
        return AdaptiveParams(
            start_sensors=start,
            samples_per_count=max(1, int(adaptive_samples_per_count)),
            intensify_samples=max(0, int(adaptive_intensify_samples)),
            patience=max(1, int(adaptive_patience)),
            min_delta=max(0.0, float(adaptive_min_delta)),
            regress_delta=max(0.0, float(adaptive_regress_delta)),
            uniform_ratio=uniform / total,
            weighted_ratio=weighted / total,
            local_ratio=local / total,
        )

    def _sampleUniformIndices(
        self,
        *,
        count: int,
        sensor_count: int,
        rng: random.Random,
    ) -> Tuple[int, ...]:
        if sensor_count <= 0:
            return ()
        return tuple(sorted(rng.sample(range(count), sensor_count)))

    def _sampleWeightedIndices(
        self,
        *,
        weights: List[float],
        sensor_count: int,
        rng: random.Random,
    ) -> Tuple[int, ...]:
        if sensor_count <= 0:
            return ()
        available = list(range(len(weights)))
        selected: List[int] = []
        local_weights = [max(1.0e-9, float(weight)) for weight in weights]
        for _ in range(sensor_count):
            choice = rng.choices(available, weights=local_weights, k=1)[0]
            index = available.index(choice)
            selected.append(choice)
            available.pop(index)
            local_weights.pop(index)
        return tuple(sorted(selected))

    def _resizeIndices(
        self,
        *,
        base_indices: Tuple[int, ...],
        sensor_count: int,
        candidate_count: int,
        rng: random.Random,
    ) -> Tuple[int, ...]:
        selected = set(base_indices)
        while len(selected) > sensor_count:
            selected.remove(rng.choice(tuple(selected)))
        while len(selected) < sensor_count:
            selected.add(rng.randrange(candidate_count))
        return tuple(sorted(selected))

    def _sampleLocalIndices(
        self,
        *,
        base_indices: Tuple[int, ...],
        sensor_count: int,
        candidate_count: int,
        rng: random.Random,
    ) -> Tuple[int, ...]:
        indices = set(
            self._resizeIndices(
                base_indices=base_indices,
                sensor_count=sensor_count,
                candidate_count=candidate_count,
                rng=rng,
            )
        )
        if sensor_count <= 0:
            return ()
        swaps = max(1, min(sensor_count, sensor_count // 3 + 1))
        for _ in range(swaps):
            if indices:
                indices.remove(rng.choice(tuple(indices)))
            while len(indices) < sensor_count:
                indices.add(rng.randrange(candidate_count))
        return tuple(sorted(indices))

    def _solutionIndices(
        self,
        *,
        solution: Chromosome,
        cover_index: Dict[Gene, int],
    ) -> Tuple[int, ...]:
        return tuple(sorted(cover_index[point] for point in solution if point in cover_index))

    def _adaptiveChunksForCount(
        self,
        *,
        covers: List[CandidateCover],
        sensor_count: int,
        budget: int,
        rng: random.Random,
        best_indices: Tuple[int, ...],
        params: AdaptiveParams,
    ) -> Iterable[List[Tuple[CandidateCover, ...]]]:
        candidate_count = len(covers)
        if sensor_count < 0 or sensor_count > candidate_count or budget <= 0:
            return

        weights = [max(1.0, float(cover.bits.bit_count())) for cover in covers]
        sampled: Set[Tuple[int, ...]] = set()
        chunk: List[Tuple[CandidateCover, ...]] = []
        attempts = 0
        max_attempts = max(budget * 20, budget + 1000)

        while len(sampled) < budget and attempts < max_attempts:
            attempts += 1
            draw = rng.random()
            if best_indices and draw >= (params.uniform_ratio + params.weighted_ratio):
                indices = self._sampleLocalIndices(
                    base_indices=best_indices,
                    sensor_count=sensor_count,
                    candidate_count=candidate_count,
                    rng=rng,
                )
            elif draw < params.uniform_ratio:
                indices = self._sampleUniformIndices(
                    count=candidate_count,
                    sensor_count=sensor_count,
                    rng=rng,
                )
            else:
                indices = self._sampleWeightedIndices(
                    weights=weights,
                    sensor_count=sensor_count,
                    rng=rng,
                )
            if indices in sampled:
                continue
            sampled.add(indices)
            chunk.append(tuple(covers[index] for index in indices))
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

    def _gpuMaxCells(self) -> int:
        return max(
            1,
            int(self.fitness_kwargs.get("gpu_max_cells", DEFAULT_GPU_MAX_CELLS)),
        )

    def _torchDevice(self, device: Optional[str]) -> Optional[object]:
        if torch is None:
            return None
        if device is not None:
            requested = torch.device(str(device))
            if requested.type == "cuda" and not torch.cuda.is_available():
                return None
            return requested
        if torch.cuda.is_available():
            return torch.device("cuda")
        return None

    def _canRunTorch(
        self,
        *,
        covers: List[CandidateCover],
        device: object,
        verbose: bool,
    ) -> bool:
        cells = (len(covers) + 1) * max(1, self._target_area)
        if cells <= self._gpuMaxCells():
            return True
        if verbose:
            print(
                "[Combinatorial GPU Fallback] "
                f"cover_matrix_cells={cells} exceeds gpu_max_cells={self._gpuMaxCells()}"
            )
        return False

    def _coverTensor(
        self,
        covers: List[CandidateCover],
        device: object,
    ):
        matrix = np.zeros((len(covers) + 1, self._target_area), dtype=np.bool_)
        for row, cover in enumerate(covers):
            if cover.indices:
                matrix[row, list(cover.indices)] = True
        return torch.as_tensor(matrix, device=device)

    def _cornerTensor(self, device: object):
        mask = np.zeros((self._target_area,), dtype=np.bool_)
        if self._corner_indices:
            mask[list(self._corner_indices)] = True
        return torch.as_tensor(mask, device=device)

    def _pointTensor(self, covers: List[CandidateCover], device: object):
        points = np.zeros((len(covers) + 1, 2), dtype=np.float32)
        for row, cover in enumerate(covers):
            points[row, 0] = float(cover.point[0])
            points[row, 1] = float(cover.point[1])
        return torch.as_tensor(points, device=device)

    def _cornerPointTensor(self, device: object):
        points = np.asarray(self.corner_positions, dtype=np.float32)
        if points.size == 0:
            points = points.reshape(0, 2)
        return torch.as_tensor(points, device=device)

    def _chunkIndexArray(
        self,
        chunk: List[Tuple[CandidateCover, ...]],
        cover_index: Dict[Gene, int],
        pad_index: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        width = max((len(selected) for selected in chunk), default=0)
        indices = np.full((len(chunk), width), pad_index, dtype=np.int64)
        counts = np.zeros((len(chunk),), dtype=np.int64)
        for row, selected in enumerate(chunk):
            counts[row] = len(selected)
            for col, cover in enumerate(selected):
                indices[row, col] = cover_index[cover.point]
        return indices, counts

    def _pairCounts(self, points, active, threshold: Optional[float]):
        batch, count, _ = points.shape
        if threshold is None or threshold <= 0.0 or count < 2:
            return torch.zeros((batch,), dtype=torch.float32, device=points.device)

        deltas = points.unsqueeze(2) - points.unsqueeze(1)
        dist2 = torch.sum(deltas * deltas, dim=-1)
        pair_mask = active.unsqueeze(2) & active.unsqueeze(1)
        pair_mask &= torch.triu(
            torch.ones((count, count), dtype=torch.bool, device=points.device),
            diagonal=1,
        ).unsqueeze(0)
        return ((dist2 < float(threshold) ** 2) & pair_mask).sum(dim=(1, 2)).to(torch.float32)

    def _scoreTorchChunk(
        self,
        *,
        chunk: List[Tuple[CandidateCover, ...]],
        start_index: int,
        cover_index: Dict[Gene, int],
        cover_tensor,
        corner_tensor,
        point_tensor,
        corner_point_tensor,
        params: FitnessParams,
        trace_enabled: bool,
        trace_stride: int,
    ) -> ChunkResult:
        if not chunk:
            return ChunkResult(0, float("-inf"), 0.0, [])

        pad_index = len(cover_index)
        index_array, count_array = self._chunkIndexArray(chunk, cover_index, pad_index)
        index_tensor = torch.as_tensor(index_array, dtype=torch.long, device=cover_tensor.device)
        count_tensor = torch.as_tensor(count_array, dtype=torch.float32, device=cover_tensor.device)

        if index_array.shape[1] == 0:
            combined = corner_tensor.unsqueeze(0).expand(len(chunk), -1)
            selected_points = point_tensor[index_tensor].reshape(len(chunk), 0, 2)
            selected_active = torch.zeros((len(chunk), 0), dtype=torch.bool, device=cover_tensor.device)
        else:
            selected_cover = cover_tensor[index_tensor]
            combined = torch.any(selected_cover, dim=1) | corner_tensor.unsqueeze(0)
            selected_points = point_tensor[index_tensor]
            selected_active = index_tensor != pad_index

        coverage_tensor = combined.sum(dim=1).to(torch.float32)
        coverage_tensor *= 100.0 / max(1, params.target_area)

        corner_count = int(corner_point_tensor.shape[0])
        if corner_count > 0:
            corner_points = corner_point_tensor.unsqueeze(0).expand(len(chunk), -1, -1)
            points = torch.cat((corner_points, selected_points), dim=1)
            corner_active = torch.ones(
                (len(chunk), corner_count),
                dtype=torch.bool,
                device=cover_tensor.device,
            )
            active = torch.cat((corner_active, selected_active), dim=1)
        else:
            points = selected_points
            active = selected_active

        separated = self._pairCounts(points, active, params.min_separation) <= 0.0
        overlap_cost = self._pairCounts(points, active, params.overlap_min_dist)
        overlap_cost *= float(params.overlap_penalty)

        total_count = count_tensor + float(params.corner_count)
        deficit = torch.clamp(float(params.target_coverage) - coverage_tensor, min=0.0)
        capped = torch.clamp(coverage_tensor, max=float(params.target_coverage))
        fitness_tensor = (
            float(params.coverage_weight) * capped
            - float(params.sensor_weight) * total_count
            - float(params.deficit_penalty) * deficit
            - overlap_cost
        )
        fitness_tensor = torch.where(
            separated,
            fitness_tensor,
            torch.full_like(fitness_tensor, float("-inf")),
        )

        best_index = int(torch.argmax(fitness_tensor).item())
        best_fitness = float(fitness_tensor[best_index].item())
        best_coverage = float(coverage_tensor[best_index].item())
        best_solution = [cover.point for cover in chunk[best_index]]
        traces = self._makeTorchTraces(
            chunk=chunk,
            start_index=start_index,
            trace_enabled=trace_enabled,
            trace_stride=trace_stride,
            coverage=coverage_tensor.detach().cpu().numpy(),
            fitness=fitness_tensor.detach().cpu().numpy(),
            feasible=separated.detach().cpu().numpy(),
            best_index=best_index,
        )
        return ChunkResult(
            evaluated=len(chunk),
            best_fitness=best_fitness,
            best_coverage=best_coverage,
            best_solution=best_solution,
            traces=traces,
        )

    def _makeTorchTraces(
        self,
        *,
        chunk: List[Tuple[CandidateCover, ...]],
        start_index: int,
        trace_enabled: bool,
        trace_stride: int,
        coverage: np.ndarray,
        fitness: np.ndarray,
        feasible: np.ndarray,
        best_index: int,
    ) -> Tuple[FitnessTrace, ...]:
        if not trace_enabled:
            return ()
        traces: List[FitnessTrace] = []
        for offset, selected in enumerate(chunk):
            index = start_index + offset
            should_trace = index % trace_stride == 0 or offset == best_index
            if not should_trace:
                continue
            solution = [cover.point for cover in selected]
            is_feasible = bool(feasible[offset])
            traces.append(
                _makeTrace(
                    index,
                    solution,
                    float(coverage[offset]) if is_feasible else None,
                    float(fitness[offset]) if is_feasible else None,
                    is_feasible,
                )
            )
        return tuple(traces)

    def _runTorch(
        self,
        *,
        covers: List[CandidateCover],
        min_count: int,
        max_count: int,
        domain_size: int,
        progress_size: int,
        sample_combinations: Optional[int],
        sample_seed: int,
        params: FitnessParams,
        logger,
        trace_logger: Optional[CombinatorialFitnessLogger],
        fitness_trace_stride: int,
        profile: bool,
        profile_every: int,
        start: float,
        device: object,
    ) -> Tuple[Chromosome, float, float, int]:
        cover_index = {cover.point: index for index, cover in enumerate(covers)}
        cover_tensor = self._coverTensor(covers, device)
        corner_tensor = self._cornerTensor(device)
        point_tensor = self._pointTensor(covers, device)
        corner_point_tensor = self._cornerPointTensor(device)

        evaluated = 0
        best_solution: Chromosome = []
        best_fitness = float("-inf")
        best_coverage = _coveragePercent(self._corner_bits, self._target_area)
        chunks = self._searchChunks(
            covers=covers,
            min_count=min_count,
            max_count=max_count,
            domain_size=domain_size,
            sample_combinations=sample_combinations,
            sample_seed=sample_seed,
        )

        with torch.no_grad():
            for chunk in chunks:
                result = self._scoreTorchChunk(
                    chunk=chunk,
                    start_index=evaluated,
                    cover_index=cover_index,
                    cover_tensor=cover_tensor,
                    corner_tensor=corner_tensor,
                    point_tensor=point_tensor,
                    corner_point_tensor=corner_point_tensor,
                    params=params,
                    trace_enabled=trace_logger is not None,
                    trace_stride=fitness_trace_stride,
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
                        progress_size,
                        best_solution,
                        best_coverage,
                        start,
                    )

        return best_solution, best_fitness, best_coverage, evaluated

    def _evaluateAdaptiveCount(
        self,
        *,
        covers: List[CandidateCover],
        sensor_count: int,
        budget: int,
        rng: random.Random,
        adaptive_params: AdaptiveParams,
        best_indices: Tuple[int, ...],
        params: FitnessParams,
        evaluated: int,
        progress_size: int,
        best_solution: Chromosome,
        best_fitness: float,
        best_coverage: float,
        logger,
        trace_logger: Optional[CombinatorialFitnessLogger],
        fitness_trace_stride: int,
        profile: bool,
        profile_every: int,
        start: float,
    ) -> Tuple[Chromosome, float, float, Chromosome, float, float, int]:
        local_solution: Chromosome = []
        local_fitness = float("-inf")
        local_coverage = 0.0
        chunks = self._adaptiveChunksForCount(
            covers=covers,
            sensor_count=sensor_count,
            budget=budget,
            rng=rng,
            best_indices=best_indices,
            params=adaptive_params,
        )

        for chunk in chunks:
            result = _scoreCombinationChunk(
                CombinationTask(
                    chunk=chunk,
                    start_index=evaluated,
                    corner_bits=self._corner_bits,
                    params=params,
                    trace_enabled=trace_logger is not None,
                    trace_stride=fitness_trace_stride,
                )
            )
            evaluated += result.evaluated
            self._logTrace(trace_logger=trace_logger, result=result)
            if result.best_fitness > local_fitness:
                local_solution = list(result.best_solution)
                local_fitness = float(result.best_fitness)
                local_coverage = float(result.best_coverage)
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
                    progress_size,
                    best_solution,
                    best_coverage,
                    start,
                )

        return (
            best_solution,
            best_fitness,
            best_coverage,
            local_solution,
            local_fitness,
            local_coverage,
            evaluated,
        )

    def _runAdaptive(
        self,
        *,
        covers: List[CandidateCover],
        min_count: int,
        max_count: int,
        progress_size: int,
        sample_combinations: Optional[int],
        sample_seed: int,
        params: FitnessParams,
        adaptive_params: AdaptiveParams,
        refine_params: AdaptiveParams,
        refine_rounds: int,
        logger,
        trace_logger: Optional[CombinatorialFitnessLogger],
        fitness_trace_stride: int,
        profile: bool,
        profile_every: int,
        start: float,
    ) -> Tuple[Chromosome, float, float, int]:
        rng = random.Random(sample_seed)
        evaluated = 0
        best_solution: Chromosome = []
        best_fitness = float("-inf")
        best_coverage = _coveragePercent(self._corner_bits, self._target_area)
        best_count = adaptive_params.start_sensors
        cover_index = {cover.point: index for index, cover in enumerate(covers)}
        remaining = sample_combinations
        no_improve = 0

        for sensor_count in range(adaptive_params.start_sensors, max_count + 1):
            if remaining is not None and remaining <= 0:
                break
            budget = adaptive_params.samples_per_count
            if remaining is not None:
                budget = min(budget, remaining)
            base_indices = self._solutionIndices(
                solution=best_solution,
                cover_index=cover_index,
            )
            previous_fitness = best_fitness
            (
                best_solution,
                best_fitness,
                best_coverage,
                local_solution,
                local_fitness,
                _,
                evaluated,
            ) = self._evaluateAdaptiveCount(
                covers=covers,
                sensor_count=sensor_count,
                budget=budget,
                rng=rng,
                adaptive_params=adaptive_params,
                best_indices=base_indices,
                params=params,
                evaluated=evaluated,
                progress_size=progress_size,
                best_solution=best_solution,
                best_fitness=best_fitness,
                best_coverage=best_coverage,
                logger=logger,
                trace_logger=trace_logger,
                fitness_trace_stride=fitness_trace_stride,
                profile=profile,
                profile_every=profile_every,
                start=start,
            )
            if remaining is not None:
                remaining -= budget

            improved = best_fitness > previous_fitness + adaptive_params.min_delta
            regressed = local_fitness < previous_fitness - adaptive_params.regress_delta
            if improved:
                best_count = len(best_solution)
                no_improve = 0
                (
                    best_solution,
                    best_fitness,
                    best_coverage,
                    evaluated,
                    remaining,
                ) = self._intensifyAdaptive(
                    covers=covers,
                    sensor_count=sensor_count,
                    remaining=remaining,
                    rng=rng,
                    adaptive_params=adaptive_params,
                    cover_index=cover_index,
                    params=params,
                    evaluated=evaluated,
                    progress_size=progress_size,
                    best_solution=best_solution,
                    best_fitness=best_fitness,
                    best_coverage=best_coverage,
                    logger=logger,
                    trace_logger=trace_logger,
                    fitness_trace_stride=fitness_trace_stride,
                    profile=profile,
                    profile_every=profile_every,
                    start=start,
                )
                continue

            no_improve += 1
            if regressed or no_improve >= adaptive_params.patience:
                (
                    best_solution,
                    best_fitness,
                    best_coverage,
                    evaluated,
                    remaining,
                ) = self._searchBestNeighborhood(
                    covers=covers,
                    best_count=best_count,
                    min_count=min_count,
                    max_count=max_count,
                    remaining=remaining,
                    rng=rng,
                    adaptive_params=adaptive_params,
                    cover_index=cover_index,
                    params=params,
                    evaluated=evaluated,
                    progress_size=progress_size,
                    best_solution=best_solution,
                    best_fitness=best_fitness,
                    best_coverage=best_coverage,
                    logger=logger,
                    trace_logger=trace_logger,
                    fitness_trace_stride=fitness_trace_stride,
                    profile=profile,
                    profile_every=profile_every,
                    start=start,
                )
                break

        if best_coverage < params.target_coverage and refine_rounds > 0:
            (
                best_solution,
                best_fitness,
                best_coverage,
                evaluated,
            ) = self._refineIncompleteCoverage(
                covers=covers,
                best_count=len(best_solution),
                min_count=min_count,
                max_count=max_count,
                remaining=remaining,
                rng=rng,
                refine_params=refine_params,
                refine_rounds=refine_rounds,
                cover_index=cover_index,
                params=params,
                evaluated=evaluated,
                progress_size=progress_size,
                best_solution=best_solution,
                best_fitness=best_fitness,
                best_coverage=best_coverage,
                logger=logger,
                trace_logger=trace_logger,
                fitness_trace_stride=fitness_trace_stride,
                profile=profile,
                profile_every=profile_every,
                start=start,
            )

        return best_solution, best_fitness, best_coverage, evaluated

    def _intensifyAdaptive(
        self,
        *,
        covers: List[CandidateCover],
        sensor_count: int,
        remaining: Optional[int],
        rng: random.Random,
        adaptive_params: AdaptiveParams,
        cover_index: Dict[Gene, int],
        params: FitnessParams,
        evaluated: int,
        progress_size: int,
        best_solution: Chromosome,
        best_fitness: float,
        best_coverage: float,
        logger,
        trace_logger: Optional[CombinatorialFitnessLogger],
        fitness_trace_stride: int,
        profile: bool,
        profile_every: int,
        start: float,
    ) -> Tuple[Chromosome, float, float, int, Optional[int]]:
        budget = adaptive_params.intensify_samples
        if budget <= 0:
            return best_solution, best_fitness, best_coverage, evaluated, remaining
        if remaining is not None:
            if remaining <= 0:
                return best_solution, best_fitness, best_coverage, evaluated, remaining
            budget = min(budget, remaining)

        best_indices = self._solutionIndices(solution=best_solution, cover_index=cover_index)
        (
            best_solution,
            best_fitness,
            best_coverage,
            _,
            _,
            _,
            evaluated,
        ) = self._evaluateAdaptiveCount(
            covers=covers,
            sensor_count=sensor_count,
            budget=budget,
            rng=rng,
            adaptive_params=adaptive_params,
            best_indices=best_indices,
            params=params,
            evaluated=evaluated,
            progress_size=progress_size,
            best_solution=best_solution,
            best_fitness=best_fitness,
            best_coverage=best_coverage,
            logger=logger,
            trace_logger=trace_logger,
            fitness_trace_stride=fitness_trace_stride,
            profile=profile,
            profile_every=profile_every,
            start=start,
        )
        if remaining is not None:
            remaining -= budget
        return best_solution, best_fitness, best_coverage, evaluated, remaining

    def _searchBestNeighborhood(
        self,
        *,
        covers: List[CandidateCover],
        best_count: int,
        min_count: int,
        max_count: int,
        remaining: Optional[int],
        rng: random.Random,
        adaptive_params: AdaptiveParams,
        cover_index: Dict[Gene, int],
        params: FitnessParams,
        evaluated: int,
        progress_size: int,
        best_solution: Chromosome,
        best_fitness: float,
        best_coverage: float,
        logger,
        trace_logger: Optional[CombinatorialFitnessLogger],
        fitness_trace_stride: int,
        profile: bool,
        profile_every: int,
        start: float,
    ) -> Tuple[Chromosome, float, float, int]:
        counts = [
            count
            for count in (best_count - 1, best_count, best_count + 1)
            if min_count <= count <= max_count
        ]
        for sensor_count in dict.fromkeys(counts):
            budget = adaptive_params.intensify_samples
            if budget <= 0:
                continue
            if remaining is not None:
                if remaining <= 0:
                    break
                budget = min(budget, remaining)
            best_indices = self._solutionIndices(solution=best_solution, cover_index=cover_index)
            (
                best_solution,
                best_fitness,
                best_coverage,
                _,
                _,
                _,
                evaluated,
            ) = self._evaluateAdaptiveCount(
                covers=covers,
                sensor_count=sensor_count,
                budget=budget,
                rng=rng,
                adaptive_params=adaptive_params,
                best_indices=best_indices,
                params=params,
                evaluated=evaluated,
                progress_size=progress_size,
                best_solution=best_solution,
                best_fitness=best_fitness,
                best_coverage=best_coverage,
                logger=logger,
                trace_logger=trace_logger,
                fitness_trace_stride=fitness_trace_stride,
                profile=profile,
                profile_every=profile_every,
                start=start,
            )
            if remaining is not None:
                remaining -= budget
        return best_solution, best_fitness, best_coverage, evaluated, remaining

    def _refineIncompleteCoverage(
        self,
        *,
        covers: List[CandidateCover],
        best_count: int,
        min_count: int,
        max_count: int,
        remaining: Optional[int],
        rng: random.Random,
        refine_params: AdaptiveParams,
        refine_rounds: int,
        cover_index: Dict[Gene, int],
        params: FitnessParams,
        evaluated: int,
        progress_size: int,
        best_solution: Chromosome,
        best_fitness: float,
        best_coverage: float,
        logger,
        trace_logger: Optional[CombinatorialFitnessLogger],
        fitness_trace_stride: int,
        profile: bool,
        profile_every: int,
        start: float,
    ) -> Tuple[Chromosome, float, float, int]:
        if not best_solution:
            best_count = max(min_count, min(max_count, refine_params.start_sensors))
        counts = [
            count
            for count in (best_count, best_count + 1, best_count - 1)
            if min_count <= count <= max_count
        ]

        for _ in range(max(0, int(refine_rounds))):
            for sensor_count in dict.fromkeys(counts):
                if best_coverage >= params.target_coverage:
                    return best_solution, best_fitness, best_coverage, evaluated
                budget = refine_params.intensify_samples
                if budget <= 0:
                    continue
                if remaining is not None:
                    if remaining <= 0:
                        return best_solution, best_fitness, best_coverage, evaluated
                    budget = min(budget, remaining)
                best_indices = self._solutionIndices(
                    solution=best_solution,
                    cover_index=cover_index,
                )
                (
                    best_solution,
                    best_fitness,
                    best_coverage,
                    _,
                    _,
                    _,
                    evaluated,
                ) = self._evaluateAdaptiveCount(
                    covers=covers,
                    sensor_count=sensor_count,
                    budget=budget,
                    rng=rng,
                    adaptive_params=refine_params,
                    best_indices=best_indices,
                    params=params,
                    evaluated=evaluated,
                    progress_size=progress_size,
                    best_solution=best_solution,
                    best_fitness=best_fitness,
                    best_coverage=best_coverage,
                    logger=logger,
                    trace_logger=trace_logger,
                    fitness_trace_stride=fitness_trace_stride,
                    profile=profile,
                    profile_every=profile_every,
                    start=start,
                )
                if remaining is not None:
                    remaining -= budget

        return best_solution, best_fitness, best_coverage, evaluated

    def _runSequential(
        self,
        *,
        covers: List[CandidateCover],
        min_count: int,
        max_count: int,
        domain_size: int,
        progress_size: int,
        sample_combinations: Optional[int],
        sample_seed: int,
        params: FitnessParams,
        logger,
        trace_logger: Optional[CombinatorialFitnessLogger],
        fitness_trace_stride: int,
        profile: bool,
        profile_every: int,
        start: float,
    ) -> Tuple[Chromosome, float, float, int]:
        evaluated = 0
        best_solution: Chromosome = []
        best_fitness = float("-inf")
        best_coverage = _coveragePercent(self._corner_bits, self._target_area)

        chunks = self._searchChunks(
            covers=covers,
            min_count=min_count,
            max_count=max_count,
            domain_size=domain_size,
            sample_combinations=sample_combinations,
            sample_seed=sample_seed,
        )
        for chunk in chunks:
            result = _scoreCombinationChunk(
                CombinationTask(
                    chunk=chunk,
                    start_index=evaluated,
                    corner_bits=self._corner_bits,
                    params=params,
                    trace_enabled=trace_logger is not None,
                    trace_stride=fitness_trace_stride,
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
                    progress_size,
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
        progress_size: int,
        sample_combinations: Optional[int],
        sample_seed: int,
        params: FitnessParams,
        logger,
        trace_logger: Optional[CombinatorialFitnessLogger],
        fitness_trace_stride: int,
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
        chunks = iter(
            self._searchChunks(
                covers=covers,
                min_count=min_count,
                max_count=max_count,
                domain_size=domain_size,
                sample_combinations=sample_combinations,
                sample_seed=sample_seed,
            )
        )
        max_pending = max(1, self.parallel_workers * 4)
        next_start = 0

        def submitNext() -> bool:
            nonlocal next_start
            if pool is None:
                raise RuntimeError("Process pool is not initialized.")
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
                trace_stride=fitness_trace_stride,
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
                            progress_size,
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
        fitness_trace_stride: int = 1,
        sample_combinations: Optional[int] = None,
        sample_seed: int = 42,
        search_mode: str = "adaptive",
        adaptive_start_sensors: Optional[int] = None,
        adaptive_samples_per_count: int = 20_000,
        adaptive_intensify_samples: int = 20_000,
        adaptive_patience: int = 2,
        adaptive_min_delta: float = 1.0,
        adaptive_regress_delta: float = 0.0,
        adaptive_uniform_ratio: float = 0.70,
        adaptive_weighted_ratio: float = 0.20,
        adaptive_local_ratio: float = 0.10,
        adaptive_refine_samples: int = 80_000,
        adaptive_refine_rounds: int = 2,
        adaptive_refine_uniform_ratio: float = 0.10,
        adaptive_refine_weighted_ratio: float = 0.30,
        adaptive_refine_local_ratio: float = 0.60,
        use_gpu: bool = False,
        device: Optional[str] = None,
        logger=None,
        **_,
    ) -> Chromosome:
        del return_best_only
        if parallel_workers is not None:
            self.parallel_workers = max(1, int(parallel_workers))
        if chunk_size is not None:
            self.chunk_size = max(1, int(chunk_size))
        trace_stride = max(1, int(fitness_trace_stride))
        sample_limit = (
            None
            if sample_combinations is None
            else max(1, int(sample_combinations))
        )

        start = time.perf_counter()
        covers = self._candidateCovers()
        min_count, max_count = self._bounds(len(covers), max_sensors)
        domain_size = self._domainSize(len(covers), min_count, max_count)
        mode = str(search_mode).lower()
        if mode not in {"adaptive", "exhaustive", "sampled"}:
            raise ValueError("search_mode must be one of: adaptive, exhaustive, sampled")
        adaptive_params = self._adaptiveParams(
            min_count=min_count,
            max_count=max_count,
            adaptive_start_sensors=adaptive_start_sensors,
            adaptive_samples_per_count=adaptive_samples_per_count,
            adaptive_intensify_samples=adaptive_intensify_samples,
            adaptive_patience=adaptive_patience,
            adaptive_min_delta=adaptive_min_delta,
            adaptive_regress_delta=adaptive_regress_delta,
            adaptive_uniform_ratio=adaptive_uniform_ratio,
            adaptive_weighted_ratio=adaptive_weighted_ratio,
            adaptive_local_ratio=adaptive_local_ratio,
        )
        refine_params = self._adaptiveParams(
            min_count=min_count,
            max_count=max_count,
            adaptive_start_sensors=adaptive_start_sensors,
            adaptive_samples_per_count=adaptive_samples_per_count,
            adaptive_intensify_samples=adaptive_refine_samples,
            adaptive_patience=adaptive_patience,
            adaptive_min_delta=adaptive_min_delta,
            adaptive_regress_delta=adaptive_regress_delta,
            adaptive_uniform_ratio=adaptive_refine_uniform_ratio,
            adaptive_weighted_ratio=adaptive_refine_weighted_ratio,
            adaptive_local_ratio=adaptive_refine_local_ratio,
        )
        adaptive_limit = self._adaptiveBudgetLimit(
            min_count=min_count,
            max_count=max_count,
            params=adaptive_params,
            sample_limit=sample_limit,
            refine_samples=adaptive_refine_samples,
            refine_rounds=adaptive_refine_rounds,
        )
        validation_limit = adaptive_limit if mode == "adaptive" else sample_limit
        self.search_stats = SearchStats(
            candidates=len(covers),
            min_sensors=min_count,
            max_sensors=max_count,
            combinations=domain_size,
        )
        self._validateDomain(self.search_stats, validation_limit)
        if mode == "adaptive":
            search_size = min(adaptive_limit, domain_size)
        else:
            search_size = domain_size if sample_limit is None else min(sample_limit, domain_size)

        effective_target = float(
            self.fitness_kwargs.get("target_coverage", target_coverage)
        )
        params = self._fitnessParams(effective_target, max_count)
        torch_device = self._torchDevice(device) if use_gpu and mode != "adaptive" else None
        use_torch = (
            torch_device is not None
            and self._canRunTorch(covers=covers, device=torch_device, verbose=verbose)
        )
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
                    "search_combinations": search_size,
                    "sample_combinations": sample_limit,
                    "sample_seed": int(sample_seed),
                    "search_mode": mode,
                    "adaptive_start_sensors": adaptive_params.start_sensors,
                    "adaptive_samples_per_count": adaptive_params.samples_per_count,
                    "adaptive_intensify_samples": adaptive_params.intensify_samples,
                    "adaptive_patience": adaptive_params.patience,
                    "adaptive_min_delta": adaptive_params.min_delta,
                    "adaptive_regress_delta": adaptive_params.regress_delta,
                    "adaptive_uniform_ratio": adaptive_params.uniform_ratio,
                    "adaptive_weighted_ratio": adaptive_params.weighted_ratio,
                    "adaptive_local_ratio": adaptive_params.local_ratio,
                    "adaptive_refine_samples": int(adaptive_refine_samples),
                    "adaptive_refine_rounds": int(adaptive_refine_rounds),
                    "adaptive_refine_uniform_ratio": refine_params.uniform_ratio,
                    "adaptive_refine_weighted_ratio": refine_params.weighted_ratio,
                    "adaptive_refine_local_ratio": refine_params.local_ratio,
                    "parallel_workers": self.parallel_workers,
                    "chunk_size": self.chunk_size,
                    "fitness_trace_stride": trace_stride,
                    "requested_gpu": bool(use_gpu),
                    "use_gpu": bool(use_torch),
                    "device": None if torch_device is None else str(torch_device),
                },
            )

        if verbose:
            engine = f"torch:{torch_device}" if use_torch else "cpu"
            print(
                "[Combinatorial Start] "
                f"candidates={len(covers)} / sensors={min_count}-{max_count} / "
                f"domain={domain_size} / workers={self.parallel_workers} / "
                f"chunk={self.chunk_size} / sample={sample_limit} / "
                f"mode={mode} / engine={engine}"
            )

        try:
            if mode == "adaptive":
                best_solution, best_fitness, best_coverage, evaluated = self._runAdaptive(
                    covers=covers,
                    min_count=min_count,
                    max_count=max_count,
                    progress_size=search_size,
                    sample_combinations=sample_limit,
                    sample_seed=int(sample_seed),
                    params=params,
                    adaptive_params=adaptive_params,
                    refine_params=refine_params,
                    refine_rounds=int(adaptive_refine_rounds),
                    logger=logger,
                    trace_logger=trace_logger,
                    fitness_trace_stride=trace_stride,
                    profile=bool(profile),
                    profile_every=max(1, int(profile_every)),
                    start=start,
                )
            elif use_torch:
                best_solution, best_fitness, best_coverage, evaluated = self._runTorch(
                    covers=covers,
                    min_count=min_count,
                    max_count=max_count,
                    domain_size=domain_size,
                    progress_size=search_size,
                    sample_combinations=sample_limit,
                    sample_seed=int(sample_seed),
                    params=params,
                    logger=logger,
                    trace_logger=trace_logger,
                    fitness_trace_stride=trace_stride,
                    profile=bool(profile),
                    profile_every=max(1, int(profile_every)),
                    start=start,
                    device=torch_device,
                )
            elif self.parallel_workers <= 1:
                best_solution, best_fitness, best_coverage, evaluated = self._runSequential(
                    covers=covers,
                    min_count=min_count,
                    max_count=max_count,
                    domain_size=domain_size,
                    progress_size=search_size,
                    sample_combinations=sample_limit,
                    sample_seed=int(sample_seed),
                    params=params,
                    logger=logger,
                    trace_logger=trace_logger,
                    fitness_trace_stride=trace_stride,
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
                    progress_size=search_size,
                    sample_combinations=sample_limit,
                    sample_seed=int(sample_seed),
                    params=params,
                    logger=logger,
                    trace_logger=trace_logger,
                    fitness_trace_stride=trace_stride,
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
