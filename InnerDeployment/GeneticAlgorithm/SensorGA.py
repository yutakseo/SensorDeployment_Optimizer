import random
from typing import List, Tuple, Dict

from .initializer import initialize_population
from .FitnessFunction import fitnessFunc
from .crossover import crossover as crossover_op
#from .selection import tournament, elitism, stochastic

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
        self.coverage = int(coverage)
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

    # -------------------------
    # Fitness (with caching)
    # -------------------------
    def fitness(self, individual: Chromosome) -> float:
        # TODO: 나중에 교체
        key = tuple(sorted(individual))
        if key in self._fitness_cache:
            return self._fitness_cache[key]
        score = float(fitnessFunc(self.map, self.corner_positions, individual))
        self._fitness_cache[key] = score
        return score

    # -------------------------
    # Selection
    # -------------------------
    def selection(self, generation: Generation) -> Generation:
        # TODO: 나중에 너가 만든 elitism/tournament/stochastic 조합으로 교체
        # 임시: 앞에서 selection_size만큼 반환 (최소 2 보장)
        k = max(2, min(self.selection_size, len(generation)))
        return generation[:k]

    # -------------------------
    # Crossover
    # -------------------------
    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> Chromosome:
        # TODO: 나중에 교체
        return crossover_op(parent1, parent2)

    # -------------------------
    # Mutation
    # -------------------------
    def mutation(self, chromosome: Chromosome) -> Chromosome:
        # TODO: 나중에 교체
        return chromosome

    # -------------------------
    # Run GA
    # -------------------------
    def run(self) -> Generation:
        population = self.population

        for _ in range(self.generations):
            parents = self.selection(population)
            if len(parents) < 2:
                break

            children: Generation = []
            random.shuffle(parents)

            while len(children) < self.child_size:
                p1 = random.choice(parents)
                p2 = random.choice(parents)
                if p1 is p2 and len(parents) > 1:
                    continue

                child = self.crossover(p1, p2)
                child = self.mutation(child)
                children.append(child)

            population = children

        self.population = population
        return self.population
