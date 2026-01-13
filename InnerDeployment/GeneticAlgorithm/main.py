# InnerDeployment/GeneticAlgorithm/main.py
from __future__ import annotations

import inspect
import random
import time
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .crossover import crossover as crossover_op
from .fitnessfunction import FitnessFunc
from .initializer import initialize_population
from .mutation import mutation as mutation_op
from .selection import elite, roulette, sus, tournament

Gene = Tuple[int, int]
Chromosome = List[Gene]
Generation = List[Chromosome]


@contextmanager
def _timer(name: str, acc: Dict[str, float]):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        acc[name] = acc.get(name, 0.0) + (time.perf_counter() - t0)


def _as_int_pairs(points) -> List[Gene]:
    return [tuple(map(int, p)) for p in points]


def _filter_kwargs(fn, kw: dict) -> dict:
    sig = inspect.signature(fn)
    allowed = set(sig.parameters.keys())
    return {k: v for k, v in (kw or {}).items() if k in allowed}


def _fitness_norm_0_100(
    *,
    internal_fitness: float,
    coverage: float,
    target_coverage: float,
    total_sensors: int,
    sensors_min: int,
    sensors_max: int,
    alpha: float = 10.0,
) -> float:
    """
    Reporting-only normalization to [0,100].

    IMPORTANT (requested):
      - If internal_fitness <= 0: return 0.0 (regardless of coverage/sensors).
      - Else: normal mapping:
          * infeasible (coverage < target): map by coverage ratio -> [0, <100)
          * feasible: penalize sensor count linearly within [sensors_min, sensors_max]
              score = 100 - alpha * frac, frac in [0,1]
    """
    if float(internal_fitness) <= 0.0:
        return 0.0

    tau = float(target_coverage)
    cov = float(coverage)

    # no target => just clamp coverage
    if tau <= 0:
        return float(max(0.0, min(100.0, cov)))

    # infeasible: ratio to target (strictly < 100)
    if cov < tau:
        return float(max(0.0, min(99.9999, 100.0 * cov / max(tau, 1e-6))))

    # feasible: sensor-penalty score
    if sensors_max <= sensors_min:
        return 100.0

    frac = (int(total_sensors) - int(sensors_min)) / float(int(sensors_max) - int(sensors_min))
    score = 100.0 - float(alpha) * frac
    return float(max(0.0, min(100.0, score)))


class SensorGA:
    """
    Sensor Deployment GA (your intended design)

    - Chromosome order is meaningful: genes are sorted by contribution.
    - Minimization under coverage constraint is the core goal.
    - Fitness is feasibility-first internally (used for selection).
    - Console log fitness is normalized to [0,100] for readability ONLY.
    """

    def __init__(
        self,
        installable_map,
        jobsite_map,
        coverage: int,
        generations: int,
        corner_positions: List[Gene],
        initial_size: int = 100,
        selection_size: int = 50,
        child_chromo_size: int = 100,
        min_sensors: int = 10,
        max_sensors: int = 100,
        fitness_kwargs: Optional[dict] = None,
        mutation_kwargs: Optional[dict] = None,
    ):
        self.installable_map = (np.asarray(installable_map) > 0).astype(np.uint8)
        self.jobsite_map = np.asarray(jobsite_map)

        self.coverage = int(coverage)
        self.generations = int(generations)
        self.corner_positions = _as_int_pairs(corner_positions)

        self.generation_size = int(initial_size)
        self.selection_size = int(selection_size)
        self.child_size = int(child_chromo_size)

        # INNER gene bounds
        self.min_sensors = int(min_sensors)
        self.max_sensors = int(max_sensors)

        # FitnessFunc.__init__ allowed kwargs only
        self.fitness_kwargs = _filter_kwargs(FitnessFunc.__init__, fitness_kwargs or {})
        self.mutation_kwargs = mutation_kwargs or {}

        # Ensure mutation receives target_coverage if your fitness uses it
        if "target_coverage" in self.fitness_kwargs and "target_coverage" not in self.mutation_kwargs:
            self.mutation_kwargs["target_coverage"] = float(self.fitness_kwargs["target_coverage"])

        self.init_population: Generation = initialize_population(
            input_map=self.installable_map,
            population_size=self.generation_size,
            corner_positions=self.corner_positions,
            coverage=self.coverage,
            min_sensors=self.min_sensors,
            max_sensors=self.max_sensors,
        )
        self.population: Generation = self.init_population

        self._mutation_allowed = set(inspect.signature(mutation_op).parameters.keys())

        self.best_solution: Optional[Chromosome] = None
        self.best_fitness: float = float("-inf")  # internal fitness
        self.best_coverage: float = float("nan")
        self.corner_points: List[Gene] = list(self.corner_positions)

    # -------------------------
    # Logging
    # -------------------------
    def _log_generation(
        self,
        gen_idx: int,
        best_fitness_norm: float,
        avg_fitness_norm: float,
        worst_fitness_norm: float,
        *,
        best_coverage: float,
        target_coverage: float,
        corner_sensor_count: int,
        sensors_min: int,
        sensors_avg: float,
        sensors_max: int,
        best_total_sensors: int,
    ) -> None:
        print(
            f"[Gen:{gen_idx:03d}/{self.generations:03d}] "
            f"Fitness(norm) : (best={best_fitness_norm:.4f}, avg={avg_fitness_norm:.4f}, worst={worst_fitness_norm:.4f}) | "
            f"Coverage(best)={best_coverage:.2f}% (target={target_coverage:.2f}%) | "
            f"Numb of sensors : (min={sensors_min}, avg={sensors_avg:.1f}, max={sensors_max}) | "
            f"BestSensors={best_total_sensors} (corner={corner_sensor_count})"
        )

    def _log_profile(
        self,
        gen_idx: int,
        prof: Dict[str, float],
        *,
        child_size: int,
        mutation_rate: float,
    ) -> None:
        exp_mut = child_size * float(mutation_rate)
        print(
            f"[Profile Gen {gen_idx:03d}] "
            f"fitness={prof.get('fitness_total', 0.0):.3f}s | "
            f"selection={prof.get('selection_total', 0.0):.3f}s | "
            f"repro={prof.get('reproduction_total', 0.0):.3f}s "
            f"(crossover={prof.get('crossover_total', 0.0):.3f}s, "
            f"mutation={prof.get('mutation_total', 0.0):.3f}s) | "
            f"calls: crossover~{child_size}, mutation~{exp_mut:.1f}"
        )
        if "fitness_ordering" in prof or "fitness_prefix" in prof or "fitness_score" in prof:
            print(
                f"               fitness_breakdown: "
                f"ordering={prof.get('fitness_ordering', 0.0):.3f}s | "
                f"prefix={prof.get('fitness_prefix', 0.0):.3f}s | "
                f"score={prof.get('fitness_score', 0.0):.3f}s | "
                f"pop={prof.get('fitness_pop', 0.0):.0f} \n"
            )

    # -------------------------
    # Prefix selection
    # -------------------------
    def _minimal_prefix_meeting_target(self, evaluator: FitnessFunc, ordered: Chromosome) -> Chromosome:
        """
        Select the smallest prefix that meets target_coverage.
        If none meets, return the prefix with maximum coverage.
        """
        tau = float(getattr(evaluator, "target_coverage", 0.0))

        best_k = 0
        best_cov = float("-inf")

        for k in range(len(ordered) + 1):
            cand = ordered[:k]
            cov = float(evaluator.computeCoverage(cand))

            if cov >= tau:
                return cand

            if cov > best_cov:
                best_cov = cov
                best_k = k

        return ordered[:best_k]

    # -------------------------
    # Fitness (population)
    # -------------------------
    def fitness(
        self,
        generation: Generation,
        *,
        profile_acc: Optional[Dict[str, float]] = None,
        profile_breakdown: bool = False,
    ) -> Tuple[Generation, List[float]]:
        evaluator = FitnessFunc(
            jobsite_map=self.jobsite_map,
            corner_positions=self.corner_positions,
            coverage=self.coverage,
            **self.fitness_kwargs,
        )

        scored: List[Tuple[float, Chromosome]] = []
        t_order = 0.0
        t_prefix = 0.0
        t_score = 0.0

        for chromosome in generation:
            chrom = _as_int_pairs(chromosome)

            if profile_breakdown:
                t0 = time.perf_counter()
                ordered = evaluator.ordering_sensors(chrom, return_score=False)
                t_order += time.perf_counter() - t0

                t0 = time.perf_counter()
                best_inner = self._minimal_prefix_meeting_target(evaluator, ordered)
                t_prefix += time.perf_counter() - t0

                t0 = time.perf_counter()
                score = float(evaluator.fitness_min_sensors(best_inner))  # INTERNAL FITNESS
                t_score += time.perf_counter() - t0
            else:
                ordered = evaluator.ordering_sensors(chrom, return_score=False)
                best_inner = self._minimal_prefix_meeting_target(evaluator, ordered)
                score = float(evaluator.fitness_min_sensors(best_inner))  # INTERNAL FITNESS

            scored.append((score, best_inner))

        scored.sort(key=lambda x: x[0], reverse=True)

        if profile_acc is not None and profile_breakdown:
            profile_acc["fitness_ordering"] = profile_acc.get("fitness_ordering", 0.0) + t_order
            profile_acc["fitness_prefix"] = profile_acc.get("fitness_prefix", 0.0) + t_prefix
            profile_acc["fitness_score"] = profile_acc.get("fitness_score", 0.0) + t_score
            profile_acc["fitness_pop"] = len(generation)

        sorted_generation = [ch for (s, ch) in scored]
        fitness_scores = [s for (s, _) in scored]  # INTERNAL FITNESS
        return sorted_generation, fitness_scores

    # -------------------------
    # Selection
    # -------------------------
    def selection(
        self,
        sorted_generation: Generation,
        fitness_scores: List[float],
        method: str = "elite",
        tournament_size: int = 3,
    ) -> Generation:
        if method == "elite":
            return elite(sorted_generation, self.selection_size)
        if method == "tournament":
            return tournament(sorted_generation, fitness_scores, tournament_size, self.selection_size)
        if method == "roulette":
            return roulette(sorted_generation, fitness_scores, self.selection_size)
        if method == "sus":
            return sus(sorted_generation, fitness_scores, self.selection_size)
        raise ValueError(method)

    # -------------------------
    # Crossover / Mutation
    # -------------------------
    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> Chromosome:
        return crossover_op(parent1, parent2, self.installable_map)

    def mutation(self, chromosome: Chromosome, *, mutation_kwargs: Optional[dict] = None) -> Chromosome:
        kw = dict(self.mutation_kwargs)
        if mutation_kwargs:
            kw.update(mutation_kwargs)

        max_total = len(self.corner_positions) + int(self.max_sensors)

        payload = dict(
            chromosome=chromosome,
            installable_map=self.installable_map,
            jobsite_map=self.jobsite_map,
            corner_positions=self.corner_positions,
            coverage=self.coverage,
        )
        if "max_total_sensors" in self._mutation_allowed:
            payload["max_total_sensors"] = max_total

        # align mutation target_coverage with fitness target_coverage
        if "target_coverage" in self._mutation_allowed and "target_coverage" not in kw:
            if "target_coverage" in self.fitness_kwargs:
                kw["target_coverage"] = float(self.fitness_kwargs["target_coverage"])

        for k, v in kw.items():
            if k in self._mutation_allowed:
                payload[k] = v

        return mutation_op(**payload)

    # -------------------------
    # Run
    # -------------------------
    def run(
        self,
        selection_method: str = "elite",
        tournament_size: int = 3,
        mutation_rate: float = 0.7,
        verbose: bool = True,
        profile: bool = True,
        profile_every: int = 1,
        profile_fitness_breakdown: bool = True,
        early_stop: bool = True,
        early_stop_coverage: float = 90.0,
        early_stop_patience: int = 10,
        return_best_only: bool = True,
        mutation_kwargs: Optional[dict] = None,
        logger=None,
        *,
        elitism_k: int = 2,
        log_alpha: float = 10.0,      # sensor-penalty strength for normalized log fitness
        log_eval_exact: bool = True,  # compute exact avg/worst normalized via evaluate()
    ) -> Union[Generation, Chromosome]:
        population = self.population

        log_eval = FitnessFunc(
            jobsite_map=self.jobsite_map,
            corner_positions=self.corner_positions,
            coverage=self.coverage,
            **self.fitness_kwargs,
        )
        tau = float(getattr(log_eval, "target_coverage", early_stop_coverage))

        stable_count = 0
        last_best_total: Optional[int] = None
        best_so_far_inner: Optional[Chromosome] = None
        best_so_far_fit_internal = float("-inf")

        for gen_idx in range(1, self.generations + 1):
            prof: Dict[str, float] = {}

            with _timer("fitness_total", prof):
                sorted_generation, fitness_scores = self.fitness(
                    population,
                    profile_acc=prof if profile else None,
                    profile_breakdown=bool(profile and profile_fitness_breakdown),
                )
            if not fitness_scores:
                break

            # INTERNAL fitness stats (used for selection; not printed directly)
            best_inner = sorted_generation[0]
            best_fit_internal = float(fitness_scores[0])
            worst_fit_internal = float(fitness_scores[-1])
            avg_fit_internal = float(sum(fitness_scores) / len(fitness_scores))

            if best_fit_internal > best_so_far_fit_internal:
                best_so_far_fit_internal = best_fit_internal
                best_so_far_inner = best_inner

            corner_cnt = len(self.corner_positions)

            # ---- Reporting-only: compute coverage/total for normalization ----
            cov_tot: List[Tuple[float, int]] = []
            if log_eval_exact:
                for inner in sorted_generation:
                    _, cov_i, tot_i = log_eval.evaluate(inner)
                    cov_tot.append((float(cov_i), int(tot_i)))
            else:
                _, cov_best, tot_best = log_eval.evaluate(best_inner)
                cov_tot = [(float(cov_best), int(tot_best))]

            # total sensor counts for generation (for printing)
            if log_eval_exact:
                total_counts = [tot for (_, tot) in cov_tot]
            else:
                total_counts = [corner_cnt + len(ch) for ch in sorted_generation]

            sensors_min = int(min(total_counts)) if total_counts else 0
            sensors_max = int(max(total_counts)) if total_counts else 0

            # feasible_min for normalization (IMPORTANT FIX)
            feasible_counts = [tot for (cov, tot) in cov_tot if cov >= tau] if log_eval_exact else []
            sensors_min_for_norm = int(min(feasible_counts)) if feasible_counts else sensors_min

            # best coverage / best total (aligned with sorted_generation[0])
            best_cov = float(cov_tot[0][0])
            best_total = int(cov_tot[0][1])

            # ---- Normalized fitness (REPORTING ONLY) ----
            if log_eval_exact:
                # NOTE: sorted_generation and fitness_scores share the same order by construction.
                norms = [
                    _fitness_norm_0_100(
                        internal_fitness=float(fitness_scores[i]),
                        coverage=float(cov_tot[i][0]),
                        target_coverage=float(tau),
                        total_sensors=int(cov_tot[i][1]),
                        sensors_min=int(sensors_min_for_norm),
                        sensors_max=int(sensors_max),
                        alpha=float(log_alpha),
                    )
                    for i in range(len(cov_tot))
                ]
                best_fit_norm = float(norms[0]) if norms else 0.0
                avg_fit_norm = float(sum(norms) / len(norms)) if norms else 0.0
                worst_fit_norm = float(min(norms)) if norms else 0.0
            else:
                best_fit_norm = _fitness_norm_0_100(
                    internal_fitness=float(best_fit_internal),
                    coverage=float(best_cov),
                    target_coverage=float(tau),
                    total_sensors=int(best_total),
                    sensors_min=int(sensors_min_for_norm),
                    sensors_max=int(sensors_max),
                    alpha=float(log_alpha),
                )
                avg_fit_norm = float(best_fit_norm)
                worst_fit_norm = float(best_fit_norm)

            if verbose:
                self._log_generation(
                    gen_idx,
                    best_fitness_norm=float(best_fit_norm),
                    avg_fitness_norm=float(avg_fit_norm),
                    worst_fitness_norm=float(worst_fit_norm),
                    best_coverage=float(best_cov),
                    target_coverage=float(tau),
                    corner_sensor_count=corner_cnt,
                    sensors_min=sensors_min,
                    sensors_avg=float(sum(total_counts) / len(total_counts)) if total_counts else 0.0,
                    sensors_max=sensors_max,
                    best_total_sensors=int(best_total),
                )

            if logger is not None:
                # keep internal metrics in logger
                logger.log_generation(
                    gen=gen_idx,
                    sensors_min=float(sensors_min),
                    sensors_max=float(sensors_max),
                    sensors_avg=float(sum(total_counts) / len(total_counts)) if total_counts else 0.0,
                    fitness_min=float(worst_fit_internal),
                    fitness_max=float(best_fit_internal),
                    fitness_avg=float(avg_fit_internal),
                    best_solution=best_inner,
                    best_fitness=float(best_fit_internal),
                    best_coverage=float(best_cov),
                )

            # early stop: coverage satisfied and best sensor count stable
            if early_stop and (float(best_cov) >= float(early_stop_coverage)):
                if last_best_total is None:
                    last_best_total = int(best_total)
                    stable_count = 1
                else:
                    if int(best_total) == int(last_best_total):
                        stable_count += 1
                    else:
                        last_best_total = int(best_total)
                        stable_count = 1

                if stable_count >= int(early_stop_patience):
                    if verbose:
                        print(
                            f"[EarlyStop] Gen={gen_idx:03d} | "
                            f"Coverage(best)={best_cov:.2f}% >= {early_stop_coverage:.2f}% and "
                            f"BestSensors={best_total} stable for {stable_count} generations."
                        )
                    break
            else:
                stable_count = 0
                last_best_total = None

            with _timer("selection_total", prof):
                parents = self.selection(
                    sorted_generation,
                    fitness_scores,
                    method=selection_method,
                    tournament_size=tournament_size,
                )
            if len(parents) < 2:
                break

            # Elitism (keep top-k INTERNAL-best)
            k = max(0, min(int(elitism_k), self.child_size))
            elites = [c[:] for c in sorted_generation[:k]]
            children: Generation = elites

            with _timer("reproduction_total", prof):
                while len(children) < self.child_size:
                    p1, p2 = random.sample(parents, 2)

                    with _timer("crossover_total", prof):
                        child = self.crossover(p1, p2)

                    if mutation_rate > 0 and random.random() < float(mutation_rate):
                        with _timer("mutation_total", prof):
                            child = self.mutation(child, mutation_kwargs=mutation_kwargs)

                    children.append(child)

            population = children

            if profile and (gen_idx % int(profile_every) == 0):
                self._log_profile(gen_idx, prof, child_size=self.child_size, mutation_rate=float(mutation_rate))

        self.population = population

        # persist best (internal)
        self.best_solution = best_so_far_inner if best_so_far_inner is not None else (population[0] if population else [])
        self.best_fitness = float(best_so_far_fit_internal) if best_so_far_inner is not None else float("nan")

        try:
            _, final_cov, _ = log_eval.evaluate(self.best_solution)
            self.best_coverage = float(final_cov)
        except Exception:
            self.best_coverage = float("nan")

        self.corner_points = list(self.corner_positions)

        if return_best_only:
            return self.best_solution
        return self.population
