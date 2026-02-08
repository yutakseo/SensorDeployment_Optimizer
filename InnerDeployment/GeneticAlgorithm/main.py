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
    if float(internal_fitness) <= 0.0:
        return 0.0

    tau = float(target_coverage)
    cov = float(coverage)

    if tau <= 0:
        return float(max(0.0, min(100.0, cov)))

    if cov < tau:
        return float(max(0.0, min(99.9999, 100.0 * cov / max(tau, 1e-6))))

    if sensors_max <= sensors_min:
        return 100.0

    frac = (int(total_sensors) - int(sensors_min)) / float(int(sensors_max) - int(sensors_min))
    score = 100.0 - float(alpha) * frac
    return float(max(0.0, min(100.0, score)))


class SensorGA:
    """
    Sensor Deployment GA (genotype/phenotype 분리 + ordering 상위 K만 적용)

    - population(염색체)은 끝까지 genotype(전체 유전자 리스트)로 유지
    - phenotype(최소 prefix / 디코딩 결과)은 평가/로그/최종 해에만 사용
    - ordering/decoder는 상위 K개에만 적용 (병목 제거)
    - run() 시그니처/입출력은 그대로 유지
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
        init_min_sensors: Optional[int] = None,
        init_max_sensors: Optional[int] = None,
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

        self.min_sensors = int(min_sensors)
        self.max_sensors = int(max_sensors)

        self.fitness_kwargs = _filter_kwargs(FitnessFunc.__init__, fitness_kwargs or {})
        self.mutation_kwargs = mutation_kwargs or {}

        if "target_coverage" in self.fitness_kwargs and "target_coverage" not in self.mutation_kwargs:
            self.mutation_kwargs["target_coverage"] = float(self.fitness_kwargs["target_coverage"])
        if "overlap_min_dist" in self.fitness_kwargs and "min_separation" not in self.mutation_kwargs:
            ov = self.fitness_kwargs.get("overlap_min_dist")
            if ov is not None:
                self.mutation_kwargs["min_separation"] = max(15.0, float(ov))

        init_min = int(init_min_sensors) if init_min_sensors is not None else int(self.min_sensors)
        init_max = int(init_max_sensors) if init_max_sensors is not None else int(self.max_sensors)
        self.init_population: Generation = initialize_population(
            input_map=self.installable_map,
            population_size=self.generation_size,
            corner_positions=self.corner_positions,
            coverage=self.coverage,
            min_sensors=init_min,
            max_sensors=init_max,
        )
        self.population: Generation = self.init_population

        self._mutation_allowed = set(inspect.signature(mutation_op).parameters.keys())

        self.best_solution: Optional[Chromosome] = None  # phenotype
        self.best_fitness: float = float("-inf")
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
    # Decoder
    # -------------------------
    def _minimal_prefix_meeting_target(self, evaluator: FitnessFunc, ordered: Chromosome) -> Chromosome:
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
    #   - ordering_top_k: 상위 K개만 ordering/decoder 적용
    # -------------------------
    def fitness(
        self,
        generation: Generation,
        *,
        profile_acc: Optional[Dict[str, float]] = None,
        profile_breakdown: bool = False,
        ordering_top_k: int = 1,
    ) -> Tuple[Generation, List[float], List[Chromosome]]:
        evaluator = FitnessFunc(
            jobsite_map=self.jobsite_map,
            corner_positions=self.corner_positions,
            coverage=self.coverage,
            **self.fitness_kwargs,
        )

        ordering_top_k = max(0, int(ordering_top_k))

        # 1) 빠른 1차 점수(근사): ordering 없이 genotype 그대로 평가
        approx_scored: List[Tuple[float, Chromosome]] = []
        for genotype in generation:
            geno = _as_int_pairs(genotype)
            fit = float(evaluator.fitness_min_sensors(geno))  # 근사(빠름)
            approx_scored.append((fit, geno))

        approx_scored.sort(key=lambda x: x[0], reverse=True)

        # 2) 상위 K개만 정확 평가(= ordering + prefix decoder)
        phenotypes: List[Chromosome] = [g for (_, g) in approx_scored]  # 기본은 phenotype=genotype(근사)
        fitness_scores: List[float] = [float(f) for (f, _) in approx_scored]

        t_order = 0.0
        t_prefix = 0.0
        t_score = 0.0

        k = min(ordering_top_k, len(approx_scored))
        for i in range(k):
            geno = approx_scored[i][1]

            if profile_breakdown:
                t0 = time.perf_counter()
                ordered = evaluator.ordering_sensors(geno, return_score=False)
                t_order += time.perf_counter() - t0

                t0 = time.perf_counter()
                pheno = self._minimal_prefix_meeting_target(evaluator, ordered)
                t_prefix += time.perf_counter() - t0

                t0 = time.perf_counter()
                fit = float(evaluator.fitness_min_sensors(pheno))
                t_score += time.perf_counter() - t0
            else:
                ordered = evaluator.ordering_sensors(geno, return_score=False)
                pheno = self._minimal_prefix_meeting_target(evaluator, ordered)
                fit = float(evaluator.fitness_min_sensors(pheno))

            phenotypes[i] = pheno
            fitness_scores[i] = fit

        # 3) 최종 정렬은 "내부 fitness" 기준 (선택 압력 유지)
        merged = list(zip(fitness_scores, [g for (_, g) in approx_scored], phenotypes))
        merged.sort(key=lambda x: x[0], reverse=True)

        sorted_genotypes = [g for (f, g, p) in merged]
        fitness_scores = [float(f) for (f, g, p) in merged]
        phenotypes = [p for (f, g, p) in merged]

        if profile_acc is not None and profile_breakdown:
            profile_acc["fitness_ordering"] = profile_acc.get("fitness_ordering", 0.0) + t_order
            profile_acc["fitness_prefix"] = profile_acc.get("fitness_prefix", 0.0) + t_prefix
            profile_acc["fitness_score"] = profile_acc.get("fitness_score", 0.0) + t_score
            profile_acc["fitness_pop"] = len(generation)

        return sorted_genotypes, fitness_scores, phenotypes

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
    # Crossover / Mutation (genotype only)
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

        if "target_coverage" in self._mutation_allowed and "target_coverage" not in kw:
            if "target_coverage" in self.fitness_kwargs:
                kw["target_coverage"] = float(self.fitness_kwargs["target_coverage"])

        for k, v in kw.items():
            if k in self._mutation_allowed:
                payload[k] = v

        return mutation_op(**payload)

    # -------------------------
    # Run (signature/IO 유지)
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
        log_alpha: float = 10.0,
        log_eval_exact: bool = True,
        ordering_top_k: int = 1,   # ✅ 추가: ordering 적용 개체 수(기본 1=best만)
    ) -> Union[Generation, Chromosome]:
        population = self.population  # genotype population

        log_eval = FitnessFunc(
            jobsite_map=self.jobsite_map,
            corner_positions=self.corner_positions,
            coverage=self.coverage,
            **self.fitness_kwargs,
        )
        tau = float(getattr(log_eval, "target_coverage", early_stop_coverage))

        stable_count = 0
        last_best_total: Optional[int] = None
        best_so_far_inner: Optional[Chromosome] = None  # phenotype
        best_so_far_fit_internal = float("-inf")

        for gen_idx in range(1, self.generations + 1):
            prof: Dict[str, float] = {}

            with _timer("fitness_total", prof):
                sorted_generation, fitness_scores, phenotypes = self.fitness(
                    population,
                    profile_acc=prof if profile else None,
                    profile_breakdown=bool(profile and profile_fitness_breakdown),
                    ordering_top_k=int(ordering_top_k),
                )
            if not fitness_scores:
                break

            best_inner = phenotypes[0]  # phenotype
            best_fit_internal = float(fitness_scores[0])
            worst_fit_internal = float(fitness_scores[-1])
            avg_fit_internal = float(sum(fitness_scores) / len(fitness_scores))

            if best_fit_internal > best_so_far_fit_internal:
                best_so_far_fit_internal = best_fit_internal
                best_so_far_inner = best_inner

            corner_cnt = len(self.corner_positions)

            cov_tot: List[Tuple[float, int]] = []
            if log_eval_exact:
                for inner in phenotypes:
                    _, cov_i, tot_i = log_eval.evaluate(inner)
                    cov_tot.append((float(cov_i), int(tot_i)))
            else:
                _, cov_best, tot_best = log_eval.evaluate(best_inner)
                cov_tot = [(float(cov_best), int(tot_best))]

            if log_eval_exact:
                total_counts = [tot for (_, tot) in cov_tot]
            else:
                total_counts = [corner_cnt + len(best_inner)]

            sensors_min = int(min(total_counts)) if total_counts else 0
            sensors_max = int(max(total_counts)) if total_counts else 0

            feasible_counts = [tot for (cov, tot) in cov_tot if cov >= tau] if log_eval_exact else []
            sensors_min_for_norm = int(min(feasible_counts)) if feasible_counts else sensors_min

            best_cov = float(cov_tot[0][0])
            best_total = int(cov_tot[0][1])

            if log_eval_exact:
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

        self.best_solution = best_so_far_inner if best_so_far_inner is not None else []
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
