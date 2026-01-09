import random
from typing import List, Tuple, Optional

import numpy as np

from .FitnessFunction import FitnessFunc

Gene = Tuple[int, int]
Chromosome = List[Gene]


def mutation(
    chromosome: Chromosome,
    *,
    installable_map,
    jobsite_map,
    corner_positions: List[Gene],
    coverage: int,
    max_tries: int = 64,
    max_total_sensors: Optional[int] = None,
    # -------- soft bias knobs (NO thresholds) --------
    p_delete_base: float = 0.35,
    p_delete_slope: float = 0.50,
    min_inner_sensors: int = 0,
) -> Chromosome:
    """
    Fitness-driven mutation (no thresholds)

    - DELETE 먼저 시도 (fitness 개선 시에만 채택)
    - ADD는 uncovered ∩ installable에서 fitness 개선 시에만 채택
    - coverage 임계값 없음
    """

    mutated: Chromosome = [tuple(map(int, g)) for g in chromosome]

    evaluator = FitnessFunc(
        jobsite_map=jobsite_map,
        corner_positions=corner_positions,
        coverage=coverage,
    )

    # 현재 fitness
    cur_fit = evaluator.fitness_score(mutated)

    corner_cnt = len(corner_positions)
    inner_cnt = len(mutated)
    total_cnt = corner_cnt + inner_cnt

    # ------------------------------
    # DELETE probability (soft bias)
    # ------------------------------
    # 센서가 많을수록 삭제 쪽으로 자연스럽게 유도
    # 하드 임계값 없음
    ref = float(max_total_sensors) if max_total_sensors is not None else max(20.0, float(total_cnt))
    ratio = min(1.0, total_cnt / ref)

    p_delete = min(
        0.95,
        max(0.0, float(p_delete_base) + float(p_delete_slope) * ratio),
    )

    # ==================================================
    # 1) DELETE attempt
    # ==================================================
    if inner_cnt > int(min_inner_sensors) and random.random() < p_delete:
        idx = random.randrange(inner_cnt)
        removed = mutated.pop(idx)

        new_fit = evaluator.fitness_score(mutated)

        # fitness 유지 또는 개선 → 삭제 확정
        if new_fit >= cur_fit:
            return mutated

        # rollback
        mutated.insert(idx, removed)

    # ==================================================
    # 2) ADD attempt
    # ==================================================
    # 센서 상한 (hard constraint)
    if max_total_sensors is not None:
        if (corner_cnt + len(mutated) + 1) > int(max_total_sensors):
            return mutated

    uncovered = evaluator.uncovered_map(mutated)
    uncovered = np.asarray(uncovered)

    installable = np.asarray(installable_map)
    installable01 = (installable == 1) | (installable == True)

    cand_mask = (uncovered == 1) & installable01
    if not np.any(cand_mask):
        return mutated

    existing = set(mutated)
    existing.update(tuple(map(int, c)) for c in corner_positions)

    flat_idx = np.flatnonzero(cand_mask)
    if flat_idx.size == 0:
        return mutated

    h, w = cand_mask.shape

    for _ in range(int(max_tries)):
        idx = int(random.choice(flat_idx))
        y = idx // w
        x = idx - y * w
        g = (int(x), int(y))

        if g in existing:
            continue

        mutated.append(g)
        new_fit = evaluator.fitness_score(mutated)

        # fitness 개선 → 추가 확정
        if new_fit >= cur_fit:
            return mutated

        # rollback
        mutated.pop()

    return mutated
