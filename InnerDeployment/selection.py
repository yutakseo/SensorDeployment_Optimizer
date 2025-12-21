from __future__ import annotations
from typing import Callable, List, Sequence, TypeVar, Optional
import numpy as np

T = TypeVar("T")  # chromosome 타입 (예: List[List[float]])
FitnessFunc = Callable[[T], float]

# =========================================================
# Global fitness function
# =========================================================
FITNESS_FUNC: Optional[FitnessFunc] = None


def set_fitness_func(func: FitnessFunc) -> None:
    """
    외부 모듈에서 적합도 함수를 주입하기 위한 setter.
    예)
        from selection_ops import set_fitness_func
        from my_fitness import fitness_func
        set_fitness_func(fitness_func)
    """
    global FITNESS_FUNC
    FITNESS_FUNC = func


def _get_fitness_func() -> FitnessFunc:
    if FITNESS_FUNC is None:
        raise RuntimeError(
            "FITNESS_FUNC is not set. Call set_fitness_func(your_fitness_func) first."
        )
    return FITNESS_FUNC


# ---------------------------------------------------------
# 내부 유틸
# ---------------------------------------------------------
def _fitness_scores(generation: Sequence[T]) -> np.ndarray:
    if len(generation) == 0:
        raise ValueError("generation is empty.")

    fitness_func = _get_fitness_func()
    scores = np.array([fitness_func(ch) for ch in generation], dtype=float)

    if scores.ndim != 1:
        raise ValueError("fitness_func must return scalar fitness for each chromosome.")
    if len(scores) != len(generation):
        raise RuntimeError("fitness score length mismatch.")
    return scores


def _clamp_k(k: int, n: int, replace_ok: bool) -> int:
    k = int(k)
    if k <= 0:
        raise ValueError("n_parents must be >= 1.")
    if not replace_ok and k > n:
        raise ValueError("n_parents cannot exceed population size when replace=False.")
    return min(k, n) if not replace_ok else k


# =========================================================
# 1) Elite (Top-k)
# =========================================================
def elite(
    generation: List[T],
    *,
    n_parents: int,
    return_indices: bool = False,
) -> List[T] | tuple[List[T], np.ndarray, np.ndarray]:
    scores = _fitness_scores(generation)
    N = len(generation)

    k = _clamp_k(n_parents, N, replace_ok=False)
    sorted_idx = np.argsort(scores)[::-1]
    idx = sorted_idx[:k]

    parents = [generation[i] for i in idx]
    if return_indices:
        return parents, idx, scores
    return parents


# =========================================================
# 2) Roulette Wheel (Fitness-proportionate)
# =========================================================
def roulette(
    generation: List[T],
    *,
    n_parents: int,
    replace: bool = True,
    eps: float = 1e-12,
    shift_if_negative: bool = True,
    temperature: Optional[float] = None,
    return_indices: bool = False,
) -> List[T] | tuple[List[T], np.ndarray, np.ndarray, np.ndarray]:
    scores = _fitness_scores(generation)
    N = len(generation)

    k = _clamp_k(n_parents, N, replace_ok=replace)

    w = scores.copy()

    if shift_if_negative:
        min_s = float(np.min(w))
        w = w - min_s
        w = np.clip(w, 0.0, None)

    if np.all(w <= eps):
        probs = np.ones_like(w) / float(N)
    else:
        w = w + eps
        if temperature is not None:
            if temperature <= 0:
                raise ValueError("temperature must be > 0.")
            w = w ** (1.0 / float(temperature))
        probs = w / float(np.sum(w))

    idx = np.random.choice(np.arange(N), size=k, replace=replace, p=probs)
    parents = [generation[i] for i in idx]

    if return_indices:
        return parents, idx, scores, probs
    return parents


# =========================================================
# 3) Tournament
# =========================================================
def tournament(
    generation: List[T],
    *,
    n_parents: int,
    tournament_size: int = 3,
    p_win: float = 1.0,
    replace_contenders: bool = True,
    return_indices: bool = False,
) -> List[T] | tuple[List[T], np.ndarray, np.ndarray]:
    scores = _fitness_scores(generation)
    N = len(generation)

    k = int(n_parents)
    if k <= 0:
        raise ValueError("n_parents must be >= 1.")

    ts = int(tournament_size)
    if ts < 2:
        raise ValueError("tournament_size must be >= 2.")
    ts = min(ts, N)

    if not (0.0 < p_win <= 1.0):
        raise ValueError("p_win must be in (0, 1].")

    pool = np.arange(N)
    chosen_idx: List[int] = []

    for _ in range(k):
        contenders = np.random.choice(pool, size=ts, replace=replace_contenders)
        ranked = contenders[np.argsort(scores[contenders])[::-1]]

        if p_win == 1.0:
            winner = int(ranked[0])
        else:
            winner = int(ranked[-1])  # fallback
            for r in ranked:
                if np.random.rand() < p_win:
                    winner = int(r)
                    break

        chosen_idx.append(winner)

    idx = np.array(chosen_idx, dtype=int)
    parents = [generation[i] for i in idx]

    if return_indices:
        return parents, idx, scores
    return parents
