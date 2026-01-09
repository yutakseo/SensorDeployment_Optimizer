# InnerDeployment/GeneticAlgorithm/mutation.py
from __future__ import annotations

import random
from typing import List, Tuple, Optional

import numpy as np

from .fitnessfunction import FitnessFunc

Gene = Tuple[int, int]
Chromosome = List[Gene]


def mutation(
    chromosome: Chromosome,
    *,
    installable_map,
    jobsite_map,
    corner_positions: List[Gene],
    coverage: int,
    max_total_sensors: Optional[int] = None,
    p_add: float = 0.40,
    p_del: float = 0.30,
    p_move: float = 0.30,
    min_coverage_keep: Optional[float] = None,
) -> Chromosome:
    # -------------------------
    # _internal
    # -------------------------
    def _toInt(points) -> Chromosome:
        return [tuple(map(int, p)) for p in points]

    def _toBool(m) -> np.ndarray:
        return (np.asarray(m) > 0)

    def _pickOp() -> str:
        ops, w = [], []
        if p_add > 0:
            ops.append("add"); w.append(float(p_add))
        if p_del > 0:
            ops.append("del"); w.append(float(p_del))
        if p_move > 0:
            ops.append("move"); w.append(float(p_move))
        if not ops:
            return "none"
        return random.choices(ops, weights=w, k=1)[0]

    def _makeEval() -> FitnessFunc:
        return FitnessFunc(jobsite_map=jobsite_map, corner_positions=corners, coverage=int(coverage))

    def _getCov(inner: Chromosome) -> float:
        nonlocal evalr
        if evalr is None:
            evalr = _makeEval()
        return float(evalr.evaluate(inner)[1])

    def _pickAdd(exist, inner: Chromosome) -> Optional[Gene]:
        nonlocal evalr
        if evalr is None:
            evalr = _makeEval()

        grid = np.asarray(evalr.uncovered_map(inner))
        if grid.shape != installable.shape:
            return None

        yx = np.argwhere((grid > 0) & installable)
        if yx.size == 0:
            return None

        for _ in range(min(32, len(yx))):
            y, x = yx[random.randrange(len(yx))]
            g = (int(x), int(y))
            if g not in exist:
                return g

        cand: List[Gene] = []
        for y, x in yx:
            g = (int(x), int(y))
            if g not in exist:
                cand.append(g)
        return random.choice(cand) if cand else None

    # -------------------------
    # normalize
    # -------------------------
    inner = _toInt(chromosome)
    corners = _toInt(corner_positions)
    installable = _toBool(installable_map)

    limit = None
    if max_total_sensors is not None:
        limit = max(0, int(max_total_sensors) - len(corners))

    op = _pickOp()
    if op == "none":
        return inner

    evalr: Optional[FitnessFunc] = None  # lazy

    exist = set(inner)
    exist.update(corners)

    # -------------------------
    # ADD
    # -------------------------
    if op == "add":
        if limit is not None and len(inner) >= limit:
            return inner

        g = _pickAdd(exist, inner)
        if g is None:
            return inner

        inner.append(g)
        return inner

    # -------------------------
    # DEL
    # -------------------------
    if op == "del":
        if not inner:
            return inner

        idx = random.randrange(len(inner))
        g0 = inner.pop(idx)

        if min_coverage_keep is not None:
            cov = _getCov(inner)
            if cov < float(min_coverage_keep):
                inner.insert(idx, g0)
                return inner

        return inner

    # -------------------------
    # MOVE
    # -------------------------
    if op == "move":
        if not inner:
            if limit is not None and len(inner) >= limit:
                return inner
            g = _pickAdd(exist, inner)
            if g is None:
                return inner
            inner.append(g)
            return inner

        idx = random.randrange(len(inner))
        g0 = inner.pop(idx)

        if limit is not None and len(inner) >= limit:
            inner.insert(idx, g0)
            return inner

        exist2 = set(inner)
        exist2.update(corners)

        g1 = _pickAdd(exist2, inner)
        if g1 is None:
            inner.insert(idx, g0)
            return inner

        inner.append(g1)

        if min_coverage_keep is not None:
            cov = _getCov(inner)
            if cov < float(min_coverage_keep):
                inner.pop()
                inner.insert(idx, g0)
                return inner

        return inner

    return inner
