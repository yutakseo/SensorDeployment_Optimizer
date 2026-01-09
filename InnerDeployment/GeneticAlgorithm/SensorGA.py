import random
from typing import List, Tuple, Dict

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

        # (현재 미사용) 추후 fitness 캐싱에 사용 가능
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

    # -------------------------
    # Logger (internal)
    # -------------------------
    def _log_generation(
        self,
        gen_idx: int,
        best_fitness: float,
        avg_fitness: float,
        best_sensor_count: int,
    ) -> None:
        print(
            f"[Gen:{gen_idx:03d}/{self.generations:03d}] "
            f"BestFit:{best_fitness:6.2f}  "
            f"AvgFit:{avg_fitness:6.2f}  "
            f"Sensors:{best_sensor_count}"
        )

    # -------------------------
    # Fitness
    # -------------------------
    def fitness(self, generation: Generation) -> Tuple[Generation, List[float]]:
        evaluator = FitnessFunc(
            jobsite_map=self.jobsite_map,
            corner_positions=self.corner_positions,
            coverage=self.coverage,
        )

        scored: List[Tuple[float, Chromosome]] = []
        for chromosome in generation:
            # 1) 염색체 내부 좌표를 greedy 순서로 정렬 (정렬 상태 유지)
            ordered_chromosome = evaluator.ordering_sensors(chromosome, return_score=False)

            # 2) fitness는 정렬된 염색체 기준으로 계산
            score = evaluator.fitness_score(ordered_chromosome)

            scored.append((score, ordered_chromosome))

        # 3) 세대 전체를 fitness 내림차순 정렬
        scored.sort(key=lambda x: x[0], reverse=True)

        sorted_generation: Generation = [chrom for (score, chrom) in scored]
        fitness_scores: List[float] = [score for (score, chrom) in scored]
        return sorted_generation, fitness_scores

    # -------------------------
    # Selection (API-based)
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
            parents = elite_selection(
                sorted_generation=sorted_generation,
                next_generation=self.selection_size,
            )
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
            raise ValueError(
                f"Unknown selection method: {method}. Choose from ['elite', 'tournament', 'roulette', 'sus']"
            )

        # 부모는 최소 2개 보장 (교배 안정성)
        if len(parents) < 2 and len(sorted_generation) >= 2:
            parents = [sorted_generation[0][:], sorted_generation[1][:]]

        return parents

    # -------------------------
    # Crossover
    # -------------------------
    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> Chromosome:
        # crossover_op 시그니처: (parent1, parent2, installable_map) -> Chromosome
        child = crossover_op(parent1, parent2, self.installable_map)

        # 교차로 인해 내부 순서가 깨질 수 있으니 정규화
        evaluator = FitnessFunc(
            jobsite_map=self.jobsite_map,
            corner_positions=self.corner_positions,
            coverage=self.coverage,
        )
        child = evaluator.ordering_sensors(child, return_score=False)
        return child

    # -------------------------
    # Mutation
    # -------------------------
    def mutation(self, chromosome: Chromosome) -> Chromosome:
        """
        Mutation:
        - corner + chromosome 기준 uncovered ∩ installable 후보 중 랜덤 1개를 chromosome 맨 뒤에 append
        - append 후 greedy 정규화로 내부 순서 유지
        """
        mutated = mutation_op(
            chromosome,
            installable_map=self.installable_map,
            jobsite_map=self.jobsite_map,
            corner_positions=self.corner_positions,
            coverage=self.coverage,
        )

        # append로 인해 순서가 깨질 수 있으니 정규화
        evaluator = FitnessFunc(
            jobsite_map=self.jobsite_map,
            corner_positions=self.corner_positions,
            coverage=self.coverage,
        )
        mutated = evaluator.ordering_sensors(mutated, return_score=False)
        return mutated

    # -------------------------
    # Run GA
    # -------------------------
    def run(
        self,
        selection_method: str = "elite",
        tournament_size: int = 3,
        mutation_rate: float = 0.7,
        verbose: bool = True,
    ) -> Generation:
        """
        mutation_rate:
            - 0.0 ~ 1.0
            - child 생성 시 mutation 적용 확률
        """
        population = self.population

        for gen_idx in range(1, self.generations + 1):
            # 1) fitness 평가 + 세대 정렬
            sorted_generation, fitness_scores = self.fitness(population)

            # logging: 세대수(현재/전체), 적합도(best/avg), 센서수(corner+inner)
            if verbose and fitness_scores:
                best_fitness = fitness_scores[0]
                avg_fitness = sum(fitness_scores) / len(fitness_scores)

                best_inner = len(sorted_generation[0]) if sorted_generation else 0
                best_sensor_count = len(self.corner_positions) + best_inner  # ✅ corner + inner

                self._log_generation(
                    gen_idx=gen_idx,
                    best_fitness=best_fitness,
                    avg_fitness=avg_fitness,
                    best_sensor_count=best_sensor_count,
                )

            # 2) selection
            parents = self.selection(
                sorted_generation=sorted_generation,
                fitness_scores=fitness_scores,
                method=selection_method,
                tournament_size=tournament_size,
            )
            if len(parents) < 2:
                break

            # 3) crossover + mutation → children 생성
            children: Generation = []
            while len(children) < self.child_size:
                p1, p2 = random.sample(parents, 2)

                child = self.crossover(p1, p2)

                # mutation 확률 적용
                if mutation_rate > 0 and random.random() < float(mutation_rate):
                    child = self.mutation(child)

                children.append(child)

            population = children

        self.population = population
        return self.population
