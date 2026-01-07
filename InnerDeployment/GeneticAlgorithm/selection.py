def elite_selection(self, next_generation) -> list:
    top_k = next_generation if next_generation is not None else self.initial_population_size

    # 적합도 기준 내림차순 정렬 → (idx, (chromosome, fitness)) 형태 유지
    sorted_population = sorted(self.population.items(), key=lambda item: item[1][1], reverse=True)
    # 상위 top_k 개체 인덱스만 추출
    selected_chromosome = [idx for idx, _ in sorted_population[:top_k]]
    return selected_chromosome