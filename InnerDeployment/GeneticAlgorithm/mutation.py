# InnerDeployment/GeneticAlgorithm/mutation.py
from __future__ import annotations

import random
from typing import List, Tuple, Optional, Dict, Any

import numpy as np

from .fitnessfunction import FitnessFunc
from .utils import to_bool_map, to_int_pairs

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
    prefix_minimize: bool = True,
    replace_duplicates: bool = True,
    empty_fill_ratio: float = 0.0,
    min_separation: float = 15.0,
    device: Optional[object] = None,
) -> Chromosome:
    """
    Mutation operator:
      - adaptive operator choice (add/del/move) based on feasibility
      - optional prefix minimization to keep the shortest feasible prefix
      - optional duplicate removal, min-distance pruning, and empty-area refill
    """

    def _make_eval() -> FitnessFunc:
        kw: Dict[str, Any] = {}
        if target_coverage is not None:
            kw["target_coverage"] = float(target_coverage)
        if big_reward is not None:
            kw["big_reward"] = float(big_reward)
        return FitnessFunc(
            jobsite_map=jobsite_map,
            corner_positions=corners,
            coverage=int(coverage),
            device=device,
            **kw,
        )

    def _ensure_eval() -> FitnessFunc:
        nonlocal evalr
        if evalr is None:
            evalr = _make_eval()
        return evalr

    def _get_cov(inner_: Chromosome) -> float:
        ev = _ensure_eval()
        return float(ev.evaluate(inner_)[1])

    def _minimal_prefix(inner_: Chromosome) -> Chromosome:
        if target_coverage is None:
            return inner_
        ev = _ensure_eval()
        ordered = ev.ordering_sensors(inner_, return_score=False)
        tau = float(target_coverage)
        best_k = 0
        best_cov = float("-inf")
        for k in range(len(ordered) + 1):
            cand = ordered[:k]
            cov = float(ev.computeCoverage(cand))
            if cov >= tau:
                return cand
            if cov > best_cov:
                best_cov = cov
                best_k = k
        return ordered[:best_k]

    def _filter_by_min_sep(points: Chromosome) -> Tuple[Chromosome, int]:
        if float(min_separation) <= 0.0:
            return points, 0
        min_d2 = float(min_separation) ** 2
        kept: Chromosome = []
        removed = 0
        base = list(corners)
        for p in points:
            x, y = int(p[0]), int(p[1])
            too_close = False
            for bx, by in base:
                dx = float(x - bx)
                dy = float(y - by)
                if (dx * dx + dy * dy) < min_d2:
                    too_close = True
                    break
            if too_close:
                removed += 1
                continue
            kept.append((x, y))
            base.append((x, y))
        return kept, removed

    def _pick_op(add_w: float, del_w: float, move_w: float) -> str:
        ops, w = [], []
        if add_w > 0:
            ops.append("add")
            w.append(float(add_w))
        if del_w > 0:
            ops.append("del")
            w.append(float(del_w))
        if move_w > 0:
            ops.append("move")
            w.append(float(move_w))
        if not ops:
            return "none"
        return random.choices(ops, weights=w, k=1)[0]

    def _pick_add(exist_set, inner_: Chromosome) -> Optional[Gene]:
        ev = _ensure_eval()
        min_sep = float(min_separation)

        def _far_enough(x: int, y: int) -> bool:
            if min_sep <= 0:
                return True
            min_d2 = min_sep * min_sep
            for px, py in exist_set:
                dx = float(x - px)
                dy = float(y - py)
                if (dx * dx + dy * dy) < min_d2:
                    return False
            return True

        # uncovered-first
        grid = np.asarray(ev.uncovered_map(inner_))
        if grid.shape == installable.shape:
            yx = np.argwhere((grid > 0) & installable)
            if yx.size > 0:
                for _ in range(min(add_candidate_tries, len(yx))):
                    y, x = yx[random.randrange(len(yx))]
                    xi, yi = int(x), int(y)
                    g = (xi, yi)
                    if g not in exist_set and _far_enough(xi, yi):
                        return g
                cand = [
                    (int(x), int(y))
                    for (y, x) in yx
                    if (int(x), int(y)) not in exist_set and _far_enough(int(x), int(y))
                ]
                if cand:
                    return random.choice(cand)

        # fallback: any installable cell
        yx2 = np.argwhere(installable)
        if yx2.size == 0:
            return None
        for _ in range(min(add_candidate_tries, len(yx2))):
            y, x = yx2[random.randrange(len(yx2))]
            xi, yi = int(x), int(y)
            g = (xi, yi)
            if g not in exist_set and _far_enough(xi, yi):
                return g
        cand2 = [
            (int(x), int(y))
            for (y, x) in yx2
            if (int(x), int(y)) not in exist_set and _far_enough(int(x), int(y))
        ]
        return random.choice(cand2) if cand2 else None

    # normalize inputs
    inner = to_int_pairs(chromosome)
    corners = to_int_pairs(corner_positions)
    installable = to_bool_map(installable_map)

    def _dedupe_and_fill(points: Chromosome) -> Chromosome:
        if (not replace_duplicates) and empty_fill_ratio <= 0 and float(min_separation) <= 0.0:
            return points
        seen = set()
        dup_count = 0
        out: Chromosome = []
        for p in points:
            key = (int(p[0]), int(p[1]))
            if key in seen:
                dup_count += 1
                continue
            seen.add(key)
            out.append(key)

        out, removed_close = _filter_by_min_sep(out)
        if (dup_count == 0) and (removed_close == 0) and (empty_fill_ratio <= 0):
            return out

        total_existing = set(out)
        total_existing.update(corners)

        add_count = dup_count + removed_close
        if empty_fill_ratio > 0:
            add_count += int(max(0, len(out)) * float(empty_fill_ratio))

        ev = _ensure_eval()
        grid = np.asarray(ev.uncovered_map(out))
        yx = np.argwhere((grid > 0) & installable)
        if yx.size == 0:
            return out

        tries = 0
        while add_count > 0 and tries < max(add_count * 10, 50):
            tries += 1
            y, x = yx[random.randrange(len(yx))]
            g = (int(x), int(y))
            if g in total_existing:
                continue
            out.append(g)
            total_existing.add(g)
            add_count -= 1

        return out

    # derive inner bounds from total bounds
    max_inner = None
    if max_total_sensors is not None:
        max_inner = max(0, int(max_total_sensors) - len(corners))
    min_inner = None
    if min_total_sensors is not None:
        min_inner = max(0, int(min_total_sensors) - len(corners))

    # evaluator is lazy (coverage checks can be expensive)
    evalr: Optional[FitnessFunc] = None

    # decide operator weights
    if adaptive_by_feasibility and target_coverage is not None:
        cur_cov = _get_cov(inner)
        feasible = (cur_cov >= float(target_coverage))
        a, d, m = feasible_weights if feasible else infeasible_weights
        op = _pick_op(a, d, m)
    else:
        op = _pick_op(p_add, p_del, p_move)

    if op == "none":
        out = _minimal_prefix(inner) if prefix_minimize else inner
        return _dedupe_and_fill(out)

    exist = set(inner)
    exist.update(corners)

    # -------------------------
    # ADD
    # -------------------------
    if op == "add":
        if max_inner is not None and len(inner) >= max_inner:
            return inner
        g = _pick_add(exist, inner)
        if g is None:
            return inner
        inner.append(g)
        out = _minimal_prefix(inner) if prefix_minimize else inner
        return _dedupe_and_fill(out)

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
            cov = _get_cov(inner)
            if cov < float(min_coverage_keep):
                inner.insert(idx, g0)  # revert
                return inner
        out = _minimal_prefix(inner) if prefix_minimize else inner
        return _dedupe_and_fill(out)

    # -------------------------
    # MOVE
    # -------------------------
    if op == "move":
        # empty -> add
        if not inner:
            if max_inner is not None and len(inner) >= max_inner:
                return inner
            g = _pick_add(exist, inner)
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

        g1 = _pick_add(exist2, inner)
        if g1 is None:
            inner.insert(idx, g0)
            out = _minimal_prefix(inner) if prefix_minimize else inner
            return _dedupe_and_fill(out)

        inner.append(g1)

        if min_coverage_keep is not None:
            cov = _get_cov(inner)
            if cov < float(min_coverage_keep):
                inner.pop()
                inner.insert(idx, g0)  # revert
                out = _minimal_prefix(inner) if prefix_minimize else inner
                return _dedupe_and_fill(out)
        out = _minimal_prefix(inner) if prefix_minimize else inner
        return _dedupe_and_fill(out)

    return _dedupe_and_fill(inner)
