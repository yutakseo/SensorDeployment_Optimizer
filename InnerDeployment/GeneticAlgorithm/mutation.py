import random
from typing import List, Tuple

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
) -> Chromosome:
    """
    Mutation:
    - corner + chromosome 기준으로 센서 배치했을 때 uncovered 영역을 계산
    - uncovered ∩ installable 후보 중 랜덤 1개를 뽑아 chromosome 맨 뒤에 추가
    - 후보가 없으면 원본 chromosome 반환(복사본)

    Args:
        chromosome: inner sensor positions [(x,y), ...]
        installable_map: (H,W) 설치 가능 그리드 (1/True가 installable)
        jobsite_map: (H,W) 커버 대상 영역(보통 1/0)
        corner_positions: corner sensor positions
        coverage: 센서 커버리지

    Returns:
        mutated: 새 유전자가 append된 chromosome (새 리스트)
    """
    # 원본 보호
    mutated: Chromosome = [tuple(map(int, g)) for g in chromosome]

    evaluator = FitnessFunc(
        jobsite_map=jobsite_map,
        corner_positions=corner_positions,
        coverage=coverage,
    )

    # uncovered_map: (H,W)에서 1이면 "jobsite_map=1 이면서 커버 안 된 셀"
    uncovered = evaluator.uncovered_map(mutated)

    # numpy로 통일
    if not isinstance(uncovered, np.ndarray):
        uncovered = np.asarray(uncovered)

    if not isinstance(installable_map, np.ndarray):
        installable = np.asarray(installable_map)
    else:
        installable = installable_map

    # installable 값이 1/True인 것을 설치 가능으로 간주
    installable01 = (installable == 1) | (installable == True)

    # 후보: uncovered==1 AND installable==True
    cand_yx = np.argwhere((uncovered == 1) & installable01)  # (N,2) with (y,x)
    if cand_yx.size == 0:
        return mutated  # 추가할 곳이 없음

    # 이미 들어있는 좌표(중복 방지)
    existing = set(mutated)
    # corner 중복도 방지(원하면 제거 가능)
    existing.update(tuple(map(int, c)) for c in corner_positions)

    # 후보 중에서 기존에 없는 좌표만 남기기
    filtered: List[Gene] = []
    for (y, x) in cand_yx:
        g = (int(x), int(y))
        if g not in existing:
            filtered.append(g)

    if not filtered:
        return mutated

    new_gene = random.choice(filtered)
    mutated.append(new_gene)  # “맨 뒤에 추가”

    return mutated
