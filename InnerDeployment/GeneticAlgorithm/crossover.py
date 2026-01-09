# InnerDeployment/GeneticAlgorithm/crossover.py
import random
from typing import List, Tuple

Gene = Tuple[int, int]
Chromosome = List[Gene]


def _getShape(m):
    try:
        h, w = int(m.shape[0]), int(m.shape[1])  # (H, W)
        return h, w
    except Exception:
        h = len(m) if m is not None else 0
        w = len(m[0]) if (h > 0 and m[0] is not None) else 0
        return h, w


def _isInstallable(m, x: int, y: int, h: int, w: int) -> bool:
    if x < 0 or y < 0 or x >= w or y >= h:
        return False
    try:
        v = m[y, x]
    except Exception:
        v = m[y][x]
    try:
        return bool(v > 0)  # 0/1, 0/255, bool 모두 안전
    except Exception:
        return bool(int(v) > 0)


def _pickGene(g1: Gene, g2: Gene, m, h: int, w: int) -> Gene:
    x1, y1 = map(int, g1)
    x2, y2 = map(int, g2)

    xmin, xmax = sorted((x1, x2))
    ymin, ymax = sorted((y1, y2))

    xmin = max(0, min(xmin, w - 1))
    xmax = max(0, min(xmax, w - 1))
    ymin = max(0, min(ymin, h - 1))
    ymax = max(0, min(ymax, h - 1))

    cand: List[Gene] = []
    for y in range(ymin, ymax + 1):
        for x in range(xmin, xmax + 1):
            if _isInstallable(m, x, y, h, w):
                cand.append((x, y))

    if cand:
        return random.choice(cand)

    return random.choice([(x1, y1), (x2, y2)])


def crossover(parent1: Chromosome, parent2: Chromosome, installable_map) -> Chromosome:
    h, w = _getShape(installable_map)
    m = min(len(parent1), len(parent2))

    # ✅ 빈 맵 방어: 이 경우 crossover 의미가 없으니 부모 중 하나를 복사해서 반환
    if h <= 0 or w <= 0 or m <= 0:
        return [tuple(map(int, p)) for p in parent1[:m]]

    out: Chromosome = []
    for i in range(m):
        out.append(_pickGene(parent1[i], parent2[i], installable_map, h, w))
    return out
