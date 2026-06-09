from __future__ import annotations

import random
import time
from typing import List, Optional, Tuple, Union

import numpy as np

from ..fitnessfunction import FitnessFunc
from ..geometry import candidate_points, nearest_installable_indices, snap_to_installable
from ..utils import to_int_pairs
from InnerDeployment.GeneticAlgorithm.initializer import initialize_population
from .acceleration import calculate_acceleration
from .initializer import initialize_swarm
from .position import update_positions

try:
    import torch
except Exception:  # pragma: no cover - optional dependency at import time
    torch = None

Gene = Tuple[int, int]
Chromosome = List[Gene]
Generation = List[Chromosome]


class SensorPSO:
    """
    Particle Swarm Optimization for inner-area sensor deployment.

    Each particle keeps a fixed position matrix sized by max_sensors; only the
    first active_count slots are evaluated.
    """

    def __init__(
        self,
        installable_map,
        jobsite_map,
        coverage: int,
        generations: int,
        corner_positions: List[Gene],
        swarm_size: int = 100,
        min_sensors: int = 10,
        max_sensors: int = 100,
        initial_min_sensors: Optional[int] = None,
        initial_max_sensors: Optional[int] = None,
        fitness_kwargs: Optional[dict] = None,
    ):
        self.installable_map = (np.asarray(installable_map) > 0).astype(np.uint8)
        self.jobsite_map = np.asarray(jobsite_map)
        self.coverage = int(coverage)
        self.generations = int(generations)
        self.corner_positions = to_int_pairs(corner_positions)

        self.swarm_size = int(swarm_size)
        self.min_sensors = max(0, int(min_sensors))
        self.max_sensors = max(self.min_sensors, int(max_sensors))

        self.fitness_kwargs = dict(fitness_kwargs or {})
        self.min_separation = max(
            0.0,
            float(self.fitness_kwargs.pop("pso_min_separation", self.coverage / 5.0)),
        )
        if (
            "device" not in self.fitness_kwargs
            and torch is not None
            and torch.cuda.is_available()
        ):
            self.fitness_kwargs["device"] = "cuda"

        self._corner_set = set(self.corner_positions)
        self._installable_points = candidate_points(
            self.installable_map,
            excluded=self.corner_positions,
        )
        if not self._installable_points:
            raise ValueError("installable_map has no installable cells for inner sensors.")

        self._height, self._width = self.installable_map.shape
        self._point_array = np.asarray(self._installable_points, dtype=np.float32)
        self._point_index = {point: index for index, point in enumerate(self._installable_points)}
        self._installable_set = set(self._installable_points)
        self._nearest_y = None
        self._nearest_x = None
        nearest = nearest_installable_indices(
            self.installable_map,
            excluded=self.corner_positions,
        )
        if nearest is not None:
            self._nearest_y, self._nearest_x = nearest

        init_min = int(initial_min_sensors) if initial_min_sensors is not None else self.min_sensors
        init_max = int(initial_max_sensors) if initial_max_sensors is not None else self.max_sensors
        init_min = max(self.min_sensors, min(self.max_sensors, init_min))
        init_max = max(init_min, min(self.max_sensors, init_max))

        self.init_population: Generation = initialize_population(
            input_map=self.installable_map,
            population_size=self.swarm_size,
            corner_positions=self.corner_positions,
            coverage=self.coverage,
            min_sensors=init_min,
            max_sensors=init_max,
        )
        self.population: Generation = [self._dedupe_chromosome(c) for c in self.init_population]

        self.best_solution: Optional[Chromosome] = None
        self.best_fitness: float = float("-inf")
        self.best_coverage: float = float("nan")
        self.corner_points: List[Gene] = list(self.corner_positions)

        self._positions, self._velocities, self._active_counts = initialize_swarm(
            self.population,
            max_sensors=self.max_sensors,
            min_sensors=self.min_sensors,
            installable_points=self._installable_points,
        )

    def _dedupe_chromosome(self, chromosome: Chromosome) -> Chromosome:
        seen = set()
        out: Chromosome = []
        for p in chromosome:
            key = (int(p[0]), int(p[1]))
            if key in seen or key in self._corner_set:
                continue
            if key not in self._installable_set:
                continue
            seen.add(key)
            out.append(key)
        return out

    def _snap_point(self, x: float, y: float) -> Gene:
        return snap_to_installable(
            x,
            y,
            width=self._width,
            height=self._height,
            installable_set=self._installable_set,
            point_array=self._point_array,
            nearest_y=self._nearest_y,
            nearest_x=self._nearest_x,
        )

    def _mark_nearby_points(self, forbidden: np.ndarray, point: Gene) -> None:
        if self.min_separation <= 0:
            return
        x, y = point
        dx = self._point_array[:, 0] - float(x)
        dy = self._point_array[:, 1] - float(y)
        forbidden |= (dx * dx + dy * dy) < self.min_separation**2

    def _nearest_available_point(
        self,
        x: float,
        y: float,
        *,
        seen: set,
        forbidden: np.ndarray,
    ) -> Optional[Gene]:
        available = np.ones(len(self._point_array), dtype=bool)
        for point in seen:
            index = self._point_index.get(point)
            if index is not None:
                available[index] = False
        if not np.any(available):
            return None

        delta = self._point_array - np.asarray([x, y], dtype=np.float32)
        dist = np.einsum("ij,ij->i", delta, delta)
        separated = available & ~forbidden
        candidates = separated if np.any(separated) else available
        index = int(np.argmin(np.where(candidates, dist, np.inf)))
        return self._installable_points[index]

    def _particle_to_chromosome(self, pos: np.ndarray, active_count: int) -> Chromosome:
        count = max(self.min_sensors, min(self.max_sensors, int(active_count)))
        out: Chromosome = []
        seen = set(self._corner_set)
        forbidden = np.zeros(len(self._point_array), dtype=bool)
        for point in self.corner_positions:
            self._mark_nearby_points(forbidden, point)
        for slot, (x, y) in enumerate(pos[:count]):
            p = self._snap_point(float(x), float(y))
            point_index = self._point_index.get(p)
            if p in seen or (point_index is not None and forbidden[point_index]):
                replacement = self._nearest_available_point(
                    float(x),
                    float(y),
                    seen=seen,
                    forbidden=forbidden,
                )
                if replacement is None:
                    continue
                p = replacement
            pos[slot] = p
            seen.add(p)
            self._mark_nearby_points(forbidden, p)
            out.append(p)

        tries = 0
        while len(out) < self.min_sensors and tries < self.min_sensors * 20:
            tries += 1
            p = random.choice(self._installable_points)
            if p in seen:
                continue
            seen.add(p)
            self._mark_nearby_points(forbidden, p)
            out.append(p)
        return out[: self.max_sensors]

    def _evaluate_swarm(self, evaluator: FitnessFunc):
        chromosomes: Generation = []
        fitness_scores: List[float] = []
        coverages: List[float] = []
        totals: List[int] = []

        for i in range(len(self._positions)):
            chromo = self._particle_to_chromosome(self._positions[i], int(self._active_counts[i]))
            self._active_counts[i] = len(chromo)
            fit, cov, total = evaluator.evaluate(chromo)
            chromosomes.append(chromo)
            fitness_scores.append(float(fit))
            coverages.append(float(cov))
            totals.append(int(total))

        return chromosomes, fitness_scores, coverages, totals

    def _prune_solution(
        self,
        evaluator: FitnessFunc,
        chromosome: Chromosome,
        *,
        target_coverage: float,
    ) -> Tuple[Chromosome, float, float]:
        solution = self._dedupe_chromosome(chromosome)
        fitness, coverage, _ = evaluator.evaluate(solution)
        if coverage < target_coverage:
            return solution, float(fitness), float(coverage)

        while solution:
            best_candidate: Optional[Chromosome] = None
            best_fitness = float("-inf")
            best_coverage = float("-inf")
            for index in range(len(solution)):
                candidate = solution[:index] + solution[index + 1 :]
                candidate_fitness, candidate_coverage, _ = evaluator.evaluate(candidate)
                if candidate_coverage < target_coverage:
                    continue
                if candidate_fitness > best_fitness:
                    best_candidate = candidate
                    best_fitness = float(candidate_fitness)
                    best_coverage = float(candidate_coverage)
            if best_candidate is None:
                break
            solution = best_candidate
            fitness = best_fitness
            coverage = best_coverage

        return solution, float(fitness), float(coverage)

    def _adjust_active_counts(
        self,
        *,
        coverages: List[float],
        target_coverage: float,
        p_add: float,
        p_del: float,
    ) -> None:
        for i, cov in enumerate(coverages):
            if cov < target_coverage and random.random() < p_add:
                self._active_counts[i] = min(self.max_sensors, int(self._active_counts[i]) + 1)
            elif cov >= target_coverage and random.random() < p_del:
                self._active_counts[i] = max(self.min_sensors, int(self._active_counts[i]) - 1)

    def _log_generation(
        self,
        gen_idx: int,
        *,
        best_coverage: float,
        target_coverage: float,
        sensors_min: int,
        sensors_avg: float,
        sensors_max: int,
        best_inner_sensors: int,
        corner_sensor_count: int,
        elapsed_sec: Optional[float] = None,
    ) -> None:
        time_part = f" / time={elapsed_sec:.3f}s" if elapsed_sec is not None else ""
        print(
            f"[PSO {gen_idx:03d}/{self.generations:03d}] "
            f"sensors: (min={sensors_min}, avg={sensors_avg:.1f}, max={sensors_max}) / "
            f"coverage: {best_coverage:.2f}% (target={target_coverage:.2f}%) / "
            f"best_inner={best_inner_sensors} (corner={corner_sensor_count})"
            f"{time_part}"
        )

    def run(
        self,
        *,
        inertia: float = 0.72,
        cognitive: float = 2.0,
        social: float = 2.0,
        velocity_clip: Optional[float] = None,
        count_add_rate: float = 0.40,
        count_del_rate: float = 0.30,
        count_change_rate: float = 0.7,
        verbose: bool = True,
        profile: bool = True,
        profile_every: int = 1,
        early_stop: bool = True,
        early_stop_coverage: float = 90.0,
        early_stop_patience: int = 10,
        return_best_only: bool = True,
        logger=None,
    ) -> Union[Generation, Chromosome]:
        t0 = time.perf_counter()
        evaluator = FitnessFunc(
            jobsite_map=self.jobsite_map,
            corner_positions=self.corner_positions,
            coverage=self.coverage,
            **self.fitness_kwargs,
        )
        tau = float(getattr(evaluator, "target_coverage", early_stop_coverage))

        pbest_pos = self._positions.copy()
        pbest_count = self._active_counts.copy()
        pbest_score = np.full(len(self._positions), -np.inf, dtype=np.float64)

        gbest_pos = self._positions[0].copy()
        gbest_count = int(self._active_counts[0])
        gbest_score = float("-inf")
        gbest_chromosome: Chromosome = []
        gbest_coverage = float("nan")

        stable_count = 0
        last_best_total: Optional[int] = None

        if velocity_clip is None:
            velocity_clip = max(1.0, float(self.coverage) / 5.0)

        for gen_idx in range(1, self.generations + 1):
            gen_t0 = time.perf_counter()
            chromosomes, fitness_scores, coverages, totals = self._evaluate_swarm(evaluator)

            for i, score in enumerate(fitness_scores):
                if score > pbest_score[i]:
                    pbest_score[i] = float(score)
                    pbest_pos[i] = self._positions[i].copy()
                    pbest_count[i] = self._active_counts[i]

            best_idx = int(np.argmax(np.asarray(fitness_scores, dtype=np.float64)))
            if fitness_scores[best_idx] > gbest_score:
                gbest_score = float(fitness_scores[best_idx])
                gbest_pos = self._positions[best_idx].copy()
                gbest_count = int(self._active_counts[best_idx])
                gbest_chromosome = chromosomes[best_idx][:]
                gbest_coverage = float(coverages[best_idx])

            self._adjust_active_counts(
                coverages=coverages,
                target_coverage=tau,
                p_add=float(count_add_rate) * float(count_change_rate),
                p_del=float(count_del_rate) * float(count_change_rate),
            )

            acceleration = calculate_acceleration(
                self._positions,
                pbest_pos,
                gbest_pos,
                cognitive=cognitive,
                social=social,
            )
            self._velocities = float(inertia) * self._velocities + acceleration
            if velocity_clip and velocity_clip > 0:
                np.clip(self._velocities, -float(velocity_clip), float(velocity_clip), out=self._velocities)

            self._positions = update_positions(
                self._positions,
                self._velocities,
                width=self._width,
                height=self._height,
            )

            sensors_min = int(min(len(c) for c in chromosomes)) if chromosomes else 0
            sensors_max = int(max(len(c) for c in chromosomes)) if chromosomes else 0
            sensors_avg = float(sum(len(c) for c in chromosomes) / len(chromosomes)) if chromosomes else 0.0
            best_total = int(totals[best_idx]) if totals else 0

            if logger is not None:
                logger.log_generation(
                    gen=gen_idx,
                    sensors_min=float(sensors_min),
                    sensors_max=float(sensors_max),
                    sensors_avg=float(sensors_avg + len(self.corner_positions)),
                    fitness_min=float(min(fitness_scores)) if fitness_scores else float("nan"),
                    fitness_max=float(max(fitness_scores)) if fitness_scores else float("nan"),
                    fitness_avg=float(sum(fitness_scores) / len(fitness_scores)) if fitness_scores else float("nan"),
                    best_solution=gbest_chromosome,
                    best_fitness=float(gbest_score),
                    best_coverage=float(gbest_coverage),
                )

            gen_dt = time.perf_counter() - gen_t0
            if verbose:
                self._log_generation(
                    gen_idx,
                    best_coverage=float(gbest_coverage),
                    target_coverage=float(tau),
                    sensors_min=sensors_min,
                    sensors_avg=sensors_avg,
                    sensors_max=sensors_max,
                    best_inner_sensors=len(gbest_chromosome),
                    corner_sensor_count=len(self.corner_positions),
                    elapsed_sec=gen_dt,
                )

            if early_stop and float(gbest_coverage) >= float(early_stop_coverage):
                if last_best_total is None or best_total != last_best_total:
                    last_best_total = best_total
                    stable_count = 1
                else:
                    stable_count += 1
                if stable_count >= int(early_stop_patience):
                    if verbose:
                        print(
                            f"[PSO EarlyStop] Gen={gen_idx:03d} | "
                            f"Coverage(best)={gbest_coverage:.2f}% >= {early_stop_coverage:.2f}% and "
                            f"BestSensors={best_total} stable for {stable_count} generations."
                        )
                    break
            else:
                stable_count = 0
                last_best_total = None

        self.best_solution, self.best_fitness, self.best_coverage = self._prune_solution(
            evaluator,
            gbest_chromosome,
            target_coverage=tau,
        )
        self.corner_points = list(self.corner_positions)
        self.population = [
            self._particle_to_chromosome(self._positions[i], int(self._active_counts[i]))
            for i in range(len(self._positions))
        ]

        if verbose:
            print(f"[PSO Total Time] {time.perf_counter() - t0:.3f}s")

        if return_best_only:
            return self.best_solution
        return self.population
