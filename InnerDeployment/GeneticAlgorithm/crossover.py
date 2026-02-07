# InnerDeployment/GeneticAlgorithm/crossover.py
import random
from typing import List, Tuple

Gene = Tuple[int, int]
Chromosome = List[Gene]


def _get_shape(m):
    try:
        h, w = int(m.shape[0]), int(m.shape[1])  # (H, W)
        return h, w
    except Exception:
        h = len(m) if m is not None else 0
        w = len(m[0]) if (h > 0 and m[0] is not None) else 0
        return h, w


def _is_installable(m, x: int, y: int, h: int, w: int) -> bool:
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


def _pick_gene(
    g1: Gene,
    g2: Gene,
    m,
    h: int,
    w: int,
    *,
    explore_rate: float = 0.25,
    expand_ratio: float = 0.35,
) -> Gene:
    x1, y1 = map(int, g1)
    x2, y2 = map(int, g2)

    # With some probability, explore a wider area by sampling any installable cell.
    if explore_rate > 0 and random.random() < float(explore_rate):
        for _ in range(64):
            rx = random.randrange(w)
            ry = random.randrange(h)
            if _is_installable(m, rx, ry, h, w):
                return (rx, ry)

    xmin, xmax = sorted((x1, x2))
    ymin, ymax = sorted((y1, y2))

    # expand search box to widen exploration between two genes
    dx = max(1, int(abs(x2 - x1) * float(expand_ratio)))
    dy = max(1, int(abs(y2 - y1) * float(expand_ratio)))
    xmin -= dx
    xmax += dx
    ymin -= dy
    ymax += dy

    xmin = max(0, min(xmin, w - 1))
    xmax = max(0, min(xmax, w - 1))
    ymin = max(0, min(ymin, h - 1))
    ymax = max(0, min(ymax, h - 1))

    cand: List[Gene] = []
    for y in range(ymin, ymax + 1):
        for x in range(xmin, xmax + 1):
            if _is_installable(m, x, y, h, w):
                cand.append((x, y))

    if cand:
        return random.choice(cand)

    return random.choice([(x1, y1), (x2, y2)])


def crossover(parent1: Chromosome, parent2: Chromosome, installable_map) -> Chromosome:
    """
    Your intended behavior:
      - ordered chromosome is meaningful (highest contribution first)
      - crossover is allowed to induce deletion pressure (length shrinkage)

    Thus:
      - child length = min(len(p1), len(p2))  (intentional)
      - gene-wise recombination within bounding box (installable-only)
    """
    h, w = _get_shape(installable_map)
    m = min(len(parent1), len(parent2))

    if h <= 0 or w <= 0 or m <= 0:
        return [tuple(map(int, p)) for p in parent1[:m]]

    out: Chromosome = []
    for i in range(m):
        out.append(_pick_gene(parent1[i], parent2[i], installable_map, h, w))
    return out
