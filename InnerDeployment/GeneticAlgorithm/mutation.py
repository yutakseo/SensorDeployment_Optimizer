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
    max_tries: int = 64,
) -> Chromosome:
    """
    Mutation (optimized):
    - corner + chromosome 기준 uncovered 영역을 계산
    - uncovered ∩ installable 후보 중 랜덤 1개를 뽑아 chromosome 맨 뒤에 추가
    - 후보가 없으면 원본 chromosome 반환(복사본)

    최적화 포인트:
    - np.argwhere((H,W)->(N,2)) 대신 np.flatnonzero((H,W)->(N,)) 사용
    - 후보 전체를 (x,y) 리스트로 만들지 않고, 랜덤 샘플링으로 중복 회피
    """
    # 원본 보호
    mutated: Chromosome = [tuple(map(int, g)) for g in chromosome]

    evaluator = FitnessFunc(
        jobsite_map=jobsite_map,
        corner_positions=corner_positions,
        coverage=coverage,
    )

    uncovered = evaluator.uncovered_map(mutated)

    # numpy로 통일
    uncovered = np.asarray(uncovered)
    installable = np.asarray(installable_map)

    # installable 값이 1/True인 것을 설치 가능으로 간주
    installable01 = (installable == 1) | (installable == True)

    # 후보 마스크
    cand_mask = (uncovered == 1) & installable01
    if not np.any(cand_mask):
        return mutated

    # 이미 들어있는 좌표(중복 방지)
    existing = set(mutated)
    existing.update(tuple(map(int, c)) for c in corner_positions)

    # 후보를 (N,2)로 만들지 말고 1D flat index로 뽑는다
    flat_idx = np.flatnonzero(cand_mask)  # shape (N,)
    if flat_idx.size == 0:
        return mutated

    h, w = cand_mask.shape

    # 랜덤으로 여러 번 뽑아 existing 중복만 회피 (전체 후보 리스트 생성 X)
    # 중복 후보가 많아도 max_tries 내에서 대부분 해결됨
    for _ in range(int(max_tries)):
        k = random.randrange(int(flat_idx.size))
        idx = int(flat_idx[k])
        y = idx // w
        x = idx - y * w
        g = (int(x), int(y))
        if g not in existing:
            mutated.append(g)
            return mutated

    # 드물게: 후보가 거의 다 existing에 포함된 경우
    # (여기서는 안전하게 “추가 없이 반환”)
    return mutated
