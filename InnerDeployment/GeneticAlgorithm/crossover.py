import random

def crossover(self, selected_idx: list) -> None:
    new_population = {}
    next_idx = 0

    # 1. 엘리트 개체 상위 N개 보존
    num_elites = min(self.next_population_size, len(selected_idx))
    elite_indices = sorted(selected_idx, key=lambda idx: self.population[idx][1], reverse=True)[:num_elites]
    for idx in elite_indices:
        new_population[next_idx] = self.population[idx]
        next_idx += 1

    # 2. 엘리트 외 나머지는 교배로 생성
    while next_idx < self.candidate_population_size:
        mom_idx, dad_idx = random.sample(selected_idx, 2)
        mom, _ = self.population[mom_idx]
        dad, _ = self.population[dad_idx]
        child = self._crossover(mom, dad)
        fitness_score = self._fitness_func(child)
        new_population[next_idx] = (child, fitness_score)
        next_idx += 1

    # 새로운 세대로 population 교체
    self.population = new_population
