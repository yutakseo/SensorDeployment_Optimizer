# InnerDeployment/GeneticAlgorithm/selection.py
from __future__ import annotations

import random
from typing import List, Tuple

Gene = Tuple[int, int]
Chromosome = List[Gene]
Generation = List[Chromosome]


def _clip(gen: Generation, count: int) -> int:
    n = len(gen)
    return 0 if n == 0 else max(0, min(int(count), n))


def _shift(scores: List[float]) -> List[float]:
    if not scores:
        return []
    low = min(scores)
    return [s - low for s in scores] if low < 0 else list(scores)


def elite(gen: Generation, count: int) -> Generation:
    k = _clip(gen, count)
    return [c[:] for c in gen[:k]]


def tournament(gen: Generation, scores: List[float], size: int, count: int) -> Generation:
    n = len(gen)
    if n == 0:
        return []
    if len(scores) != n:
        raise ValueError("scores length must match generation length")

    k = _clip(gen, count)
    if k == 0:
        return []

    if n == 1:
        return [gen[0][:] for _ in range(k)]

    t = max(2, min(int(size), n))

    idx = list(range(n))
    out: Generation = []
    while len(out) < k:
        cand = random.sample(idx, t)
        best = max(cand, key=lambda i: scores[i])
        out.append(gen[best][:])
    return out


def roulette(gen: Generation, scores: List[float], count: int) -> Generation:
    n = len(gen)
    if n == 0:
        return []
    if len(scores) != n:
        raise ValueError("scores length must match generation length")

    k = _clip(gen, count)
    if k == 0:
        return []

    w = _shift(scores)
    total = float(sum(w))
    if total <= 0:
        pick = random.choices(range(n), k=k)
        return [gen[i][:] for i in pick]

    pick = random.choices(range(n), weights=w, k=k)
    return [gen[i][:] for i in pick]


def sus(gen: Generation, scores: List[float], count: int) -> Generation:
    n = len(gen)
    if n == 0:
        return []
    if len(scores) != n:
        raise ValueError("scores length must match generation length")

    k = _clip(gen, count)
    if k == 0:
        return []

    w = _shift(scores)
    total = float(sum(w))
    if total <= 0:
        pick = random.choices(range(n), k=k)
        return [gen[i][:] for i in pick]

    cdf: List[float] = []
    cum = 0.0
    inv = 1.0 / total
    for s in w:
        cum += s * inv
        cdf.append(cum)

    step = 1.0 / k
    start = random.uniform(0.0, step)

    out: Generation = []
    j = 0
    for i in range(k):
        p = start + i * step
        while j < n and cdf[j] < p:
            j += 1
        if j >= n:
            j = n - 1
        out.append(gen[j][:])
    return out


# -------------------------
# backward compatible aliases
# -------------------------
elite_selection = elite
tournament_selection = tournament
roulette_wheel_selection = roulette
stochastic_universal_sampling = sus
