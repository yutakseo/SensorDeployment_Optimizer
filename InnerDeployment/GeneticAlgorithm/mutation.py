# InnerDeployment/GeneticAlgorithm/mutation.py
from __future__ import annotations

import random
from typing import List, Tuple, Optional, Dict, Any

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
    min_total_sensors: Optional[int] = None,
    # base probabilities (used only if adaptive_by_feasibility=False or target_coverage is None)
    p_add: float = 0.40,
    p_del: float = 0.30,
    p_move: float = 0.30,
    # coverage guards / objective settings
    target_coverage: Optional[float] = None,
    min_coverage_keep: Optional[float] = None,
    big_reward: float = 1_000_000.0,
    # feasibility-aware operator selection (recommended for your objective)
    adaptive_by_feasibility: bool = True,
    infeasible_weights: Tuple[float, float, float] = (0.70, 0.05, 0.25),  # add, del, move
    feasible_weights: Tuple[float, float, float] = (0.20, 0.55, 0.25),    # add, del, move
    add_candidate_tries: int = 32,
) -> Chromosome:
    """
    Mutation consistent with your goal:
      - allow deletion pressure to minimize sensors
      - BUT prevent uncontrolled collapse below coverage constraint:
        by default, min_coverage_keep := target_coverage

    Behavior:
      - If infeasible (cov < target): heavily suppress deletion, prefer add/move to recover.
      - If feasible: encourage deletion to minimize sensors while staying feasible.
    """

    def _toInt(points) -> Chromosome:
        return [tuple(map(int, p)) for p in points]

    def _toBool(m) -> np.ndarray:
        return (np.asarray(m) > 0)

    def _makeEval() -> FitnessFunc:
        kw: Dict[str, Any] = {}
        if target_coverage is not None:
            kw["target_coverage"] = float(target_coverage)
        if big_reward is not None:
            kw["big_reward"] = float(big_reward)
        return FitnessFunc(jobsite_map=jobsite_map, corner_positions=corners, coverage=int(coverage), **kw)

    def _ensure_eval() -> FitnessFunc:
        nonlocal evalr
        if evalr is None:
            evalr = _makeEval()
        return evalr

    def _getCov(inner_: Chromosome) -> float:
        ev = _ensure_eval()
        return float(ev.evaluate(inner_)[1])

    def _pickOp(add_w: float, del_w: float, move_w: float) -> str:
        ops, w = [], []
        if add_w > 0:
            ops.append("add"); w.append(float(add_w))
        if del_w > 0:
            ops.append("del"); w.append(float(del_w))
        if move_w > 0:
            ops.append("move"); w.append(float(move_w))
        if not ops:
            return "none"
        return random.choices(ops, weights=w, k=1)[0]

    def _pickAdd(exist_set, inner_: Chromosome) -> Optional[Gene]:
        ev = _ensure_eval()

        # uncovered-first
        grid = np.asarray(ev.uncovered_map(inner_))
        if grid.shape == installable.shape:
            yx = np.argwhere((grid > 0) & installable)
            if yx.size > 0:
                for _ in range(min(add_candidate_tries, len(yx))):
                    y, x = yx[random.randrange(len(yx))]
                    g = (int(x), int(y))
                    if g not in exist_set:
                        return g
                cand = [(int(x), int(y)) for (y, x) in yx if (int(x), int(y)) not in exist_set]
                if cand:
                    return random.choice(cand)

        # fallback: any installable cell
        yx2 = np.argwhere(installable)
        if yx2.size == 0:
            return None
        for _ in range(min(add_candidate_tries, len(yx2))):
            y, x = yx2[random.randrange(len(yx2))]
            g = (int(x), int(y))
            if g not in exist_set:
                return g
        cand2 = [(int(x), int(y)) for (y, x) in yx2 if (int(x), int(y)) not in exist_set]
        return random.choice(cand2) if cand2 else None

    # -------------------------
    # normalize
    # -------------------------
    inner = _toInt(chromosome)
    corners = _toInt(corner_positions)
    installable = _toBool(installable_map)

    # derive inner bounds from total bounds
    max_inner = None
    if max_total_sensors is not None:
        max_inner = max(0, int(max_total_sensors) - len(corners))
    min_inner = None
    if min_total_sensors is not None:
        min_inner = max(0, int(min_total_sensors) - len(corners))

    # evaluator is lazy (coverage checks can be expensive)
    evalr: Optional[FitnessFunc] = None

    # default safety: keep feasibility unless user explicitly wants otherwise
    if min_coverage_keep is None and target_coverage is not None:
        min_coverage_keep = float(target_coverage)

    # decide operator weights
    if adaptive_by_feasibility and target_coverage is not None:
        cur_cov = _getCov(inner)
        feasible = (cur_cov >= float(target_coverage))
        a, d, m = feasible_weights if feasible else infeasible_weights
        op = _pickOp(a, d, m)
    else:
        op = _pickOp(p_add, p_del, p_move)

    if op == "none":
        return inner

    exist = set(inner)
    exist.update(corners)

    # -------------------------
    # ADD
    # -------------------------
    if op == "add":
        if max_inner is not None and len(inner) >= max_inner:
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
        if min_inner is not None and len(inner) <= min_inner:
            return inner

        idx = random.randrange(len(inner))
        g0 = inner.pop(idx)

        if min_coverage_keep is not None:
            cov = _getCov(inner)
            if cov < float(min_coverage_keep):
                inner.insert(idx, g0)  # revert
                return inner
        return inner

    # -------------------------
    # MOVE
    # -------------------------
    if op == "move":
        # empty -> add
        if not inner:
            if max_inner is not None and len(inner) >= max_inner:
                return inner
            g = _pickAdd(exist, inner)
            if g is None:
                return inner
            inner.append(g)
            return inner

        idx = random.randrange(len(inner))
        g0 = inner.pop(idx)

        if max_inner is not None and len(inner) >= max_inner:
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
                inner.insert(idx, g0)  # revert
                return inner
        return inner

    return inner
