import random
from typing import List, Tuple

Gene = Tuple[int, int]
Chromosome = List[Gene]


def _is_installable(installable_map, x: int, y: int) -> bool:
    """
    installable_map[y, x] 기준으로 설치 가능 여부 판단.
    - numpy array면 installable_map[y, x]
    - list of list면 installable_map[y][x]
    값이 1(True)이면 설치 가능으로 간주.
    """
    try:
        v = installable_map[y, x]
    except Exception:
        v = installable_map[y][x]
    return bool(v == 1 or v is True)


def _crossover_gene(g1: Gene, g2: Gene, installable_map) -> Gene:
    """
    두 부모 유전자(g1, g2)가 만드는 사각형 범위 내의 installable 후보 중
    랜덤 1개를 뽑아 자식 유전자를 생성.
    """
    x1, y1 = map(int, g1)
    x2, y2 = map(int, g2)

    xmin, xmax = sorted([x1, x2])
    ymin, ymax = sorted([y1, y2])

    candidates: List[Gene] = []
    for y in range(ymin, ymax + 1):
        for x in range(xmin, xmax + 1):
            if _is_installable(installable_map, x, y):
                candidates.append((x, y))

    if candidates:
        return random.choice(candidates)

    # fallback: 후보가 없으면 g1(또는 g2)로
    # (원하면 random.choice([g1, g2])로 바꿔도 됨)
    return (x1, y1)


def crossover(parent1: Chromosome, parent2: Chromosome, installable_map) -> Chromosome:
    """
    1:1 유전자 대응 교차.
    - child 길이 = min(len(parent1), len(parent2))
    - 각 i번째 유전자는 parent1[i]~parent2[i] 사각형 범위 내 installable 후보 중 랜덤 선택
    - parent2의 tail(len(parent2)>len(parent1))은 버림 (요구사항 반영)
    """
    m = min(len(parent1), len(parent2))
    child: Chromosome = []
    for i in range(m):
        child.append(_crossover_gene(parent1[i], parent2[i], installable_map))
    return child
