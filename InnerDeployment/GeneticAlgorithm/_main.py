import random
from typing import List, Tuple, Optional, Dict
from initializer import initialize_population
from FitnessFunction import fitnessFunc
from crossover import crossover
from selection import *
from FitnessFunction import fitnessFunc

XY = Tuple[int, int]
Chromosome = List[XY]

class SensorGA:
    def __init__(
        self,
        input_map,
        coverage: int,
        generations: int,
        corner_positions: List[XY],
        initial_size: int = 100,
        selection_size: int = 50,
        child_chromo_size: int = 100,
        # 초기 염색체 길이 범위 (추가 센서 개수)
        min_sensors: int = 70,
        max_sensors: int = 100,
    ):
        self.map = input_map
        self.coverage = int(coverage)
        self.generations = int(generations)
        self.corner_positions = [tuple(p) for p in corner_positions]

        self.generation_size = int(initial_size)
        self.selection_size = int(selection_size)
        self.child_size = int(child_chromo_size)

        self.min_sensors = int(min_sensors)
        self.max_sensors = int(max_sensors)

        # fitness cache: key=tuple(sorted(chromosome)) or tuple(chromosome)
        self._fitness_cache: Dict[Tuple[XY, ...], float] = {}

        # 1) 초기 population 생성 (코너 배치 -> uncovered 후보 -> 랜덤 샘플)
        self.population: List[Chromosome] = initialize_population(
            input_map=self.map,
            population_size=self.generation_size,
            corner_positions=self.corner_positions,
            coverage=self.coverage,
            min_sensors=self.min_sensors,
            max_sensors=self.max_sensors,
        )

    # -------------------------
    # Fitness (with caching)
    # -------------------------
    def fitness(self, individual: Chromosome) -> float:
        # 염색체는 좌표 순서가 의미 없으면 정렬해서 캐시 키를 만들면 더 캐시가 잘 먹습니다.
        # (순서가 의미 있으면 tuple(individual)로 바꾸세요.)
        key = tuple(sorted(individual))
        if key in self._fitness_cache:
            return self._fitness_cache[key]

        score = fitnessFunc(self.map, self.corner_positions, individual, coverage=self.coverage)
        self._fitness_cache[key] = float(score)
        return float(score)

    # -------------------------
    # Selection: elitism + tournament
    # returns: selected parents (Chromosome list)
    # -------------------------
    def selection(self, generation: List[Chromosome]) -> List[Chromosome]:
        if len(generation) == 0:
            return []

        # 전체 개체 fitness 계산
        scored = [(self.fitness(ch), ch) for ch in generation]
        # fitness가 "클수록 좋다" 가정 (반대면 reverse=False로 바꾸거나 score=-score 처리)
        scored.sort(key=lambda x: x[0], reverse=True)

        # elitism
        elites = [ch for _, ch in scored[: max(0, self.elitism)]]

        # tournament selection으로 나머지 채우기
        parents: List[Chromosome] = elites[:]
        need = max(0, self.selection_size - len(parents))

        if need == 0:
            return parents

        # tournament 후보군은 generation 전체
        for _ in range(need):
            cand = random.sample(scored, k=min(self.tournament_k, len(scored)))
            # cand: [(score, chrom), ...]
            cand.sort(key=lambda x: x[0], reverse=True)
            parents.append(cand[0][1])

        return parents

    # -------------------------
    # Crossover (delegated)
    # -------------------------
    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> Chromosome:
        return crossover(parent1, parent2)

    # -------------------------
    # Mutation (simple example)
    # - 여기서는 "유효 후보공간"이 필요합니다.
    #   가장 깔끔한 방식: initializer가 uncovered_points를 같이 반환/보관.
    #   지금은 placeholder로 pass.
    # -------------------------
    def mutation(self, chromosome: Chromosome) -> Chromosome:
        # TODO: uncovered 후보 공간(또는 installable 후보 공간)에서 1~2개 gene 치환
        return chromosome

    # -------------------------
    # Run GA
    # -------------------------
    def run(self) -> List[Chromosome]:
        population = self.population

        for gen in range(self.generations):
            # 1) selection
            parents = self.selection(population)
            if len(parents) == 0:
                break

            # 2) crossover -> children 생성
            children: List[Chromosome] = []

            # parents를 섞어서 짝짓기
            random.shuffle(parents)

            # child_size 만큼 만들기
            while len(children) < self.child_size:
                p1 = random.choice(parents)
                p2 = random.choice(parents)
                if p1 is p2 and len(parents) > 1:
                    continue

                child = self.crossover(p1, p2)
                child = self.mutation(child)
                children.append(child)

            # 3) 다음 세대 구성
            population = children

        self.population = population
        return self.population