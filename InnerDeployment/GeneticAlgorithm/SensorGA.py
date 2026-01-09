import random
import time
from contextlib import contextmanager
from typing import List, Tuple, Dict, Optional

from .initializer import initialize_population
from .FitnessFunction import FitnessFunc
from .crossover import crossover as crossover_op
from .mutation import mutation as mutation_op
from .selection import (
    elite_selection,
    tournament_selection,
    roulette_wheel_selection,
    stochastic_universal_sampling,
)

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


class SensorGA:
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
        min_sensors: int = 70,
        max_sensors: int = 100,
    ):
        self.installable_map = installable_map
        self.jobsite_map = jobsite_map
        self.coverage = int(coverage) / 5  # 사용자 코드 유지
        self.generations = int(generations)
        self.corner_positions = [tuple(p) for p in corner_positions]
        self.generation_size = int(initial_size)
        self.selection_size = int(selection_size)
        self.child_size = int(child_chromo_size)
        self.min_sensors = int(min_sensors)
        self.max_sensors = int(max_sensors)

        self._fitness_cache: Dict[Tuple[Gene, ...], float] = {}

        self.init_population: Generation = initialize_population(
            input_map=self.installable_map,
            population_size=self.generation_size,
            corner_positions=self.corner_positions,
            coverage=self.coverage,
            min_sensors=self.min_sensors,
            max_sensors=self.max_sensors,
        )
        self.population: Generation = self.init_population

    def _log_generation(self, gen_idx: int, best_fitness: float, avg_fitness: float, best_sensor_count: int) -> None:
        print(
            f"[Gen:{gen_idx:03d}/{self.generations:03d}] "
            f"BestFit:{best_fitness:6.2f}  "
            f"AvgFit:{avg_fitness:6.2f}  "
            f"Sensors:{best_sensor_count}"
        )

    def _log_profile(self, gen_idx: int, prof: Dict[str, float], *, child_size: int, mutation_rate: float) -> None:
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
                f"pop={prof.get('fitness_pop', 0.0):.0f}"
            )

    # -------------------------
    # Fitness
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
        )

        scored: List[Tuple[float, Chromosome]] = []
        t_order = 0.0
        t_score = 0.0

        for chromosome in generation:
            if profile_breakdown:
                t0 = time.perf_counter()
                ordered = evaluator.ordering_sensors(chromosome, return_score=False)
                t_order += (time.perf_counter() - t0)

                t0 = time.perf_counter()
                score = evaluator.fitness_score(ordered)
                t_score += (time.perf_counter() - t0)
            else:
                ordered = evaluator.ordering_sensors(chromosome, return_score=False)
                score = evaluator.fitness_score(ordered)

            scored.append((score, ordered))

        scored.sort(key=lambda x: x[0], reverse=True)

        if profile_acc is not None and profile_breakdown:
            profile_acc["fitness_ordering"] = profile_acc.get("fitness_ordering", 0.0) + t_order
            profile_acc["fitness_score"] = profile_acc.get("fitness_score", 0.0) + t_score
            profile_acc["fitness_pop"] = len(generation)

        sorted_generation: Generation = [chrom for (score, chrom) in scored]
        fitness_scores: List[float] = [score for (score, chrom) in scored]
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
        if not sorted_generation:
            return []

        if method == "elite":
            parents = elite_selection(sorted_generation=sorted_generation, next_generation=self.selection_size)
        elif method == "tournament":
            parents = tournament_selection(
                sorted_generation=sorted_generation,
                fitness_scores=fitness_scores,
                tournament_size=tournament_size,
                next_generation=self.selection_size,
            )
        elif method == "roulette":
            parents = roulette_wheel_selection(
                sorted_generation=sorted_generation,
                fitness_scores=fitness_scores,
                next_generation=self.selection_size,
            )
        elif method == "sus":
            parents = stochastic_universal_sampling(
                sorted_generation=sorted_generation,
                fitness_scores=fitness_scores,
                next_generation=self.selection_size,
            )
        else:
            raise ValueError(f"Unknown selection method: {method}. Choose from ['elite', 'tournament', 'roulette', 'sus']")

        if len(parents) < 2 and len(sorted_generation) >= 2:
            parents = [sorted_generation[0][:], sorted_generation[1][:]]

        return parents

    # -------------------------
    # Crossover  (ordering 제거)
    # -------------------------
    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> Chromosome:
        return crossover_op(parent1, parent2, self.installable_map)

    # -------------------------
    # Mutation  (ordering 제거)
    # -------------------------
    def mutation(self, chromosome: Chromosome) -> Chromosome:
        return mutation_op(
            chromosome,
            installable_map=self.installable_map,
            jobsite_map=self.jobsite_map,
            corner_positions=self.corner_positions,
            coverage=self.coverage,
        )

    # -------------------------
    # Run GA
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
    ) -> Generation:
        population = self.population

        for gen_idx in range(1, self.generations + 1):
            prof: Dict[str, float] = {}

            with _timer("fitness_total", prof):
                sorted_generation, fitness_scores = self.fitness(
                    population,
                    profile_acc=prof if profile else None,
                    profile_breakdown=bool(profile and profile_fitness_breakdown),
                )

            if verbose and fitness_scores:
                best_fitness = fitness_scores[0]
                avg_fitness = sum(fitness_scores) / len(fitness_scores)
                best_inner = len(sorted_generation[0]) if sorted_generation else 0
                best_sensor_count = len(self.corner_positions) + best_inner
                self._log_generation(gen_idx, best_fitness, avg_fitness, best_sensor_count)

            with _timer("selection_total", prof):
                parents = self.selection(
                    sorted_generation=sorted_generation,
                    fitness_scores=fitness_scores,
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
                            child = self.mutation(child)

                    children.append(child)

            population = children

            if profile and (gen_idx % int(profile_every) == 0):
                self._log_profile(gen_idx, prof, child_size=self.child_size, mutation_rate=float(mutation_rate))

        self.population = population
        return self.population
