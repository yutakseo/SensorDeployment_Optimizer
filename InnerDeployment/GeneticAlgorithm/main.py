# InnerDeployment/GeneticAlgorithm/main.py
import random
import time
import inspect
from contextlib import contextmanager
from typing import List, Tuple, Dict, Optional, Union

import numpy as np

from .initializer import initialize_population
from .fitnessfunction import FitnessFunc
from .crossover import crossover as crossover_op
from .mutation import mutation as mutation_op
from .selection import elite, tournament, roulette, sus  # ✅ 새 이름으로 import

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


class SensorGA:
    """
    Sensor Deployment GA

    - FitnessFunc 기반 fitness maximize
    - ordering + best-prefix로 센서 수 최소화 압력
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
        # map normalize (0/255, bool 등 들어와도 0/1로 통일)
        self.installable_map = (np.asarray(installable_map) > 0).astype(np.uint8)
        self.jobsite_map = np.asarray(jobsite_map)

        self.coverage = int(coverage)
        self.generations = int(generations)

        self.corner_positions = _as_int_pairs(corner_positions)

        self.generation_size = int(initial_size)
        self.selection_size = int(selection_size)
        self.child_size = int(child_chromo_size)

        # ✅ 이 값들은 "inner gene 개수" 기준으로 사용 중(initializer가 그렇게 동작)
        self.min_sensors = int(min_sensors)
        self.max_sensors = int(max_sensors)

        # FitnessFunc가 실제로 받는 키만 통과 (예전 키 남아도 안전)
        # FitnessFunc.__init__(self, jobsite_map, corner_positions, coverage, *, useCache, sampleCount, gainLimit)
        self.fitness_kwargs = _filter_kwargs(FitnessFunc.__init__, fitness_kwargs or {})

        # mutation kwargs는 mutation() 시그니처에 맞춰서 run 때 필터링하되,
        # 여기서는 그대로 보관
        self.mutation_kwargs = mutation_kwargs or {}

        self.init_population: Generation = initialize_population(
            input_map=self.installable_map,
            population_size=self.generation_size,
            corner_positions=self.corner_positions,
            coverage=self.coverage,
            min_sensors=self.min_sensors,
            max_sensors=self.max_sensors,
        )
        self.population: Generation = self.init_population

        # mutation signature cache (매 호출 inspect 비용 감소)
        self._mutation_allowed = set(inspect.signature(mutation_op).parameters.keys())

        # ✅ run 후 외부에서 접근 가능한 최종 결과 저장용(초기화)
        self.best_solution: Optional[Chromosome] = None
        self.best_fitness: float = float("-inf")
        self.best_coverage: float = float("nan")
        self.corner_points: List[Gene] = list(self.corner_positions)

    # -------------------------
    # Logging (console)
    # -------------------------
    def _log_generation(
        self,
        gen_idx: int,
        best_fitness: float,
        avg_fitness: float,
        worst_fitness: float,
        *,
        best_coverage: float,
        corner_sensor_count: int,
        sensors_min: int,
        sensors_avg: float,
        sensors_max: int,
        best_total_sensors: int,
    ) -> None:
        print(
            f"[Gen:{gen_idx:03d}/{self.generations:03d}] "
            f"Fitness : (best={best_fitness:.4f}, avg={avg_fitness:.4f}, worst={worst_fitness:.4f}) | "
            f"Coverage(best)={best_coverage:.2f}% | "
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
        if "fitness_ordering" in prof or "fitness_score" in prof:
            print(
                f"               fitness_breakdown: "
                f"ordering={prof.get('fitness_ordering', 0.0):.3f}s | "
                f"score={prof.get('fitness_score', 0.0):.3f}s | "
                f"pop={prof.get('fitness_pop', 0.0):.0f} \n"
            )

    # -------------------------
    # best prefix
    # -------------------------
    def _best_prefix_by_fitness(self, evaluator: FitnessFunc, ordered: Chromosome) -> Chromosome:
        best_fit = float("-inf")
        best_k = 0
        for k in range(len(ordered) + 1):
            cand = ordered[:k]
            fit = float(evaluator.fitness_score(cand))
            if fit > best_fit:
                best_fit = fit
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
        t_score = 0.0

        for chromosome in generation:
            chrom = _as_int_pairs(chromosome)

            if profile_breakdown:
                t0 = time.perf_counter()
                ordered = evaluator.ordering_sensors(chrom, return_score=False)
                t_order += time.perf_counter() - t0

                t0 = time.perf_counter()
                best_chrom = self._best_prefix_by_fitness(evaluator, ordered)
                score = float(evaluator.fitness_score(best_chrom))
                t_score += time.perf_counter() - t0
            else:
                ordered = evaluator.ordering_sensors(chrom, return_score=False)
                best_chrom = self._best_prefix_by_fitness(evaluator, ordered)
                score = float(evaluator.fitness_score(best_chrom))

            scored.append((score, best_chrom))

        scored.sort(key=lambda x: x[0], reverse=True)

        if profile_acc is not None and profile_breakdown:
            profile_acc["fitness_ordering"] = profile_acc.get("fitness_ordering", 0.0) + t_order
            profile_acc["fitness_score"] = profile_acc.get("fitness_score", 0.0) + t_score
            profile_acc["fitness_pop"] = len(generation)

        sorted_generation = [chrom for (score, chrom) in scored]
        fitness_scores = [score for (score, _) in scored]
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

        # ✅ max_sensors는 inner 한도이므로 total 한도로 변환해서 넘김
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

        # knob도 allowed에 있는 것만
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
        logger=None,  # ✅ 추가: Tools.Logger.GAJsonLogger
    ) -> Union[Generation, Chromosome]:
        population = self.population

        log_eval = FitnessFunc(
            jobsite_map=self.jobsite_map,
            corner_positions=self.corner_positions,
            coverage=self.coverage,
            **self.fitness_kwargs,
        )

        stable_count = 0
        last_best_total: Optional[int] = None

        best_so_far_inner: Optional[Chromosome] = None
        best_so_far_fit = float("-inf")

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

            best_inner = sorted_generation[0]
            best_fitness = float(fitness_scores[0])
            worst_fitness = float(fitness_scores[-1])
            avg_fitness = float(sum(fitness_scores) / len(fitness_scores))

            _, best_cov, best_total = log_eval.evaluate(best_inner)

            if best_fitness > best_so_far_fit:
                best_so_far_fit = best_fitness
                best_so_far_inner = best_inner

            corner_cnt = len(self.corner_positions)
            total_counts = [corner_cnt + len(ch) for ch in sorted_generation]

            if verbose:
                self._log_generation(
                    gen_idx,
                    best_fitness=best_fitness,
                    avg_fitness=avg_fitness,
                    worst_fitness=worst_fitness,
                    best_coverage=best_cov,
                    corner_sensor_count=corner_cnt,
                    sensors_min=min(total_counts),
                    sensors_avg=sum(total_counts) / len(total_counts),
                    sensors_max=max(total_counts),
                    best_total_sensors=int(best_total),
                )

            # ✅ 세대별 로거 기록 (JSON용)
            if logger is not None:
                logger.log_generation(
                    gen=gen_idx,
                    sensors_min=float(min(total_counts)),
                    sensors_max=float(max(total_counts)),
                    sensors_avg=float(sum(total_counts) / len(total_counts)),
                    fitness_min=float(worst_fitness),   # min
                    fitness_max=float(best_fitness),    # max
                    fitness_avg=float(avg_fitness),
                    best_solution=best_inner,
                    best_fitness=float(best_fitness),
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

            children: Generation = []
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

        # ✅ 최종 best 결과를 객체에 저장 (Experiment에서 logger.finalize에 사용)
        self.best_solution = best_so_far_inner if best_so_far_inner is not None else (population[0] if population else [])
        self.best_fitness = float(best_so_far_fit) if best_so_far_inner is not None else float("nan")
        try:
            _, final_cov, _ = log_eval.evaluate(self.best_solution)
            self.best_coverage = float(final_cov)
        except Exception:
            self.best_coverage = float("nan")

        self.corner_points = list(self.corner_positions)

        if return_best_only:
            return self.best_solution
        return self.population
