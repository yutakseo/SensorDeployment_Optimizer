import random

def elite_selection(self, next_generation) -> list:
    top_k = next_generation if next_generation is not None else self.initial_population_size

    # 적합도 기준 내림차순 정렬 → (idx, (chromosome, fitness)) 형태 유지
    sorted_population = sorted(self.population.items(), key=lambda item: item[1][1], reverse=True)
    # 상위 top_k 개체 인덱스만 추출
    selected_chromosome = [idx for idx, _ in sorted_population[:top_k]]
    return selected_chromosome


def tournament_selection(self, tournament_size: int, next_generation: int) -> list:
    selected_chromosome = []
    population_indices = list(self.population.keys())
    while len(selected_chromosome) < next_generation:
        # 토너먼트 참가자 무작위 선택
        tournament_contestants = random.sample(population_indices, tournament_size)
        # 참가자 중 최고 적합도 개체 선택
        best_contestant = max(tournament_contestants, key=lambda idx: self.population[idx][1])
        selected_chromosome.append(best_contestant)
    return selected_chromosome


def roulette_wheel_selection(self, next_generation: int) -> list:
    total_fitness = sum(fitness for _, fitness in self.population.values())
    if total_fitness == 0:
        # 모든 개체의 적합도가 0인 경우 무작위 선택
        return random.choices(list(self.population.keys()), k=next_generation)
    selection_probs = [
        fitness / total_fitness for _, fitness in self.population.values()
    ]
    population_indices = list(self.population.keys())
    selected_chromosome = random.choices(
        population_indices, weights=selection_probs, k=next_generation
    )
    return selected_chromosome

def stochastic_universal_sampling(self, next_generation: int) -> list:
    total_fitness = sum(fitness for _, fitness in self.population.values())
    if total_fitness == 0:
        # 모든 개체의 적합도가 0인 경우 무작위 선택
        return random.choices(list(self.population.keys()), k=next_generation)
    selection_probs = [
        fitness / total_fitness for _, fitness in self.population.values()
    ]
    population_indices = list(self.population.keys())
    pointers = []
    pointer_distance = 1.0 / next_generation
    start_pointer = random.uniform(0, pointer_distance)
    for i in range(next_generation):
        pointers.append(start_pointer + i * pointer_distance)
    selected_chromosome = []
    cumulative_prob = 0.0
    current_member = 0

    for pointer in pointers:
        while cumulative_prob < pointer:
            cumulative_prob += selection_probs[current_member]
            current_member += 1
        selected_chromosome.append(population_indices[current_member - 1])
    return selected_chromosome