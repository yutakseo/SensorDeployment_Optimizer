import random
from typing import List, Tuple

import numpy as np

Gene = Tuple[int, int]
Chromosome = List[Gene]


def _is_installable(installable_map, x: int, y: int) -> bool:
    """
    installable_map[y, x] 기준으로 설치 가능 여부 판단.
    값이 1(True)이면 설치 가능으로 간주.
    """
    try:
        v = installable_map[y, x]
    except Exception:
        v = installable_map[y][x]
    return bool(v == 1 or v is True)


def _crossover_gene_fast(g1: Gene, g2: Gene, installable_map, *, max_tries: int = 64) -> Gene:
    """
    두 부모 유전자(g1, g2)가 만드는 사각형 범위 내에서 installable 지점을 선택.

    최적화 전략:
    - installable_map이 numpy ndarray면:
        슬라이스 -> np.argwhere로 후보 좌표를 한 번에 얻고, 그중 랜덤 1개 선택
    - list-of-list 등 일반 구조면:
        candidates 전체 스캔 대신, (x,y) 랜덤 샘플링을 여러 번 시도(rejection sampling)
    """
    x1, y1 = map(int, g1)
    x2, y2 = map(int, g2)

    xmin, xmax = sorted([x1, x2])
    ymin, ymax = sorted([y1, y2])

    # numpy fast path
    if isinstance(installable_map, np.ndarray):
        # 경계 체크(안전)
        h, w = installable_map.shape[:2]
        xmin2 = max(0, min(xmin, w - 1))
        xmax2 = max(0, min(xmax, w - 1))
        ymin2 = max(0, min(ymin, h - 1))
        ymax2 = max(0, min(ymax, h - 1))

        sub = installable_map[ymin2 : ymax2 + 1, xmin2 : xmax2 + 1]
        # installable: 1 또는 True
        mask = (sub == 1) | (sub == True)
        ys, xs = np.where(mask)
        if ys.size > 0:
            k = random.randrange(int(ys.size))
            # sub 좌표 -> 원본 좌표
            return (int(xmin2 + xs[k]), int(ymin2 + ys[k]))

        return (x1, y1)

    # generic fallback: rejection sampling
    for _ in range(max_tries):
        x = random.randint(xmin, xmax)
        y = random.randint(ymin, ymax)
        if _is_installable(installable_map, x, y):
            return (x, y)

    # 최후 fallback: 사각형이 좁거나 installable이 거의 없을 때만 전체 스캔(드물게)
    candidates: List[Gene] = []
    for y in range(ymin, ymax + 1):
        for x in range(xmin, xmax + 1):
            if _is_installable(installable_map, x, y):
                candidates.append((x, y))
    if candidates:
        return random.choice(candidates)

    return (x1, y1)


def crossover(parent1: Chromosome, parent2: Chromosome, installable_map) -> Chromosome:
    """
    1:1 유전자 대응 교차.
    - child 길이 = min(len(parent1), len(parent2))
    - 각 i번째 유전자는 parent1[i]~parent2[i] 사각형 범위 내 installable 후보 중 선택
    - parent2의 tail(len(parent2)>len(parent1))은 버림
    """
    m = min(len(parent1), len(parent2))
    child: Chromosome = []
    for i in range(m):
        child.append(_crossover_gene_fast(parent1[i], parent2[i], installable_map))
    return child
