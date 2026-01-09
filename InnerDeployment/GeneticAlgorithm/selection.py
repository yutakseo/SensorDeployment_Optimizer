import random
from typing import List, Tuple

Gene = Tuple[int, int]
Chromosome = List[Gene]
Generation = List[Chromosome]


def elite_selection(
    sorted_generation: Generation,
    next_generation: int,
) -> Generation:
    """
    Elite selection (truncation):
    - sorted_generation은 fitness 내림차순 정렬되어 있다고 가정
    - 상위 next_generation개를 그대로 반환
    """
    n = len(sorted_generation)
    if n == 0:
        return []

    k = max(0, min(int(next_generation), n))
    return [chrom[:] for chrom in sorted_generation[:k]]


def tournament_selection(
    sorted_generation: Generation,
    fitness_scores: List[float],
    tournament_size: int,
    next_generation: int,
) -> Generation:
    """
    Tournament selection:
    - 토너먼트 참가자는 random.sample로 "중복 없이" 뽑음
    - 각 토너먼트에서 최고 fitness 개체를 1개 선택
    """
    n = len(sorted_generation)
    if n == 0:
        return []

    k = max(0, min(int(next_generation), n))
    t = max(2, min(int(tournament_size), n))

    selected: Generation = []
    indices = list(range(n))

    while len(selected) < k:
        contestants = random.sample(indices, t)
        best_idx = max(contestants, key=lambda i: fitness_scores[i])
        selected.append(sorted_generation[best_idx][:])  # 순서 유지 + 복사

    return selected


def roulette_wheel_selection(
    sorted_generation: Generation,
    fitness_scores: List[float],
    next_generation: int,
) -> Generation:
    """
    Roulette wheel selection:
    - fitness 비례 확률로 선택
    - fitness_scores는 sorted_generation과 같은 순서라고 가정
    - (안전) 음수 fitness가 들어오면 shift하여 0 이상으로 만든 뒤 사용
    """
    n = len(sorted_generation)
    if n == 0:
        return []

    k = max(0, min(int(next_generation), n))
    if k == 0:
        return []

    min_fs = min(fitness_scores) if fitness_scores else 0.0
    shifted = [fs - min_fs for fs in fitness_scores] if min_fs < 0 else list(fitness_scores)

    total = float(sum(shifted))
    if total <= 0:
        picks = random.choices(range(n), k=k)
        return [sorted_generation[i][:] for i in picks]

    picks = random.choices(range(n), weights=shifted, k=k)
    return [sorted_generation[i][:] for i in picks]


def stochastic_universal_sampling(
    sorted_generation: Generation,
    fitness_scores: List[float],
    next_generation: int,
) -> Generation:
    """
    Stochastic Universal Sampling (SUS):
    - roulette보다 분산이 적고 안정적으로 선택됨
    - (안전) 음수 fitness가 들어오면 shift하여 0 이상으로 만든 뒤 사용
    """
    n = len(sorted_generation)
    if n == 0:
        return []

    k = max(0, min(int(next_generation), n))
    if k == 0:
        return []

    min_fs = min(fitness_scores) if fitness_scores else 0.0
    shifted = [fs - min_fs for fs in fitness_scores] if min_fs < 0 else list(fitness_scores)

    total = float(sum(shifted))
    if total <= 0:
        picks = random.choices(range(n), k=k)
        return [sorted_generation[i][:] for i in picks]

    # 누적 분포(CDF)
    weights = [fs / total for fs in shifted]
    cdf: List[float] = []
    cum = 0.0
    for w in weights:
        cum += w
        cdf.append(cum)

    step = 1.0 / k
    start = random.uniform(0.0, step)
    pointers = [start + i * step for i in range(k)]

    selected: Generation = []
    idx = 0
    for p in pointers:
        while idx < n and cdf[idx] < p:
            idx += 1
        if idx >= n:
            idx = n - 1
        selected.append(sorted_generation[idx][:])  # 순서 유지 + 복사

    return selected
