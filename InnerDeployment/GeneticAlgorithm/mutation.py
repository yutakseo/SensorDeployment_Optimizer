import random
from typing import List, Tuple, Optional

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
    max_total_sensors: Optional[int] = None,
    # --- operation probabilities (sum doesn't need to be 1.0) ---
    p_add: float = 0.40,
    p_del: float = 0.30,
    p_move: float = 0.30,
    # --- safety knobs ---
    min_coverage_keep: Optional[float] = None,  # 예: 90.0 (%). None이면 삭제 시 커버리지 체크 안함
) -> Chromosome:
    """
    Mutation operator (ADD / DELETE / MOVE).

    - corners는 고정이므로 chromosome(내부 센서)만 변형
    - ADD  : uncovered ∩ installable 중 랜덤 1개를 append
    - DEL  : 내부 센서 1개를 제거 (옵션으로 coverage 하락이 크면 롤백)
    - MOVE : 내부 센서 1개를 제거하고, uncovered ∩ installable에서 1개를 추가 (DEL+ADD)

    Args:
        chromosome: inner sensor positions [(x,y), ...]
        installable_map: (H,W) 설치 가능 마스크 (0/1 or 0/255 모두 허용)
        jobsite_map: (H,W) 커버 대상 마스크/레이어 (0/1 or 0/255 or 값>0)
        corner_positions: corner sensors (fixed)
        coverage: coverage radius
        max_total_sensors: (corner+inner) 최대 센서 수 제한
        p_add/p_del/p_move: 연산 확률
        min_coverage_keep: 삭제/이동 후 coverage%가 이 값 미만이면 롤백 (None이면 검사 생략)

    Returns:
        mutated chromosome (new list)
    """
    # -------------------------
    # normalize inputs
    # -------------------------
    mutated: Chromosome = [tuple(map(int, g)) for g in chromosome]
    corners = [tuple(map(int, c)) for c in corner_positions]

    evaluator = FitnessFunc(
        jobsite_map=jobsite_map,
        corner_positions=corners,
        coverage=coverage,
    )

    installable = np.asarray(installable_map)
    installable01 = (installable > 0)  # ✅ 0/1, 0/255 모두 안전

    # 기존 좌표 중복 방지 셋
    existing = set(mutated)
    existing.update(corners)

    # -------------------------
    # helper: get add candidates
    # -------------------------
    def _sample_add_gene(exist: set[Gene]) -> Optional[Gene]:
        uncovered = evaluator.uncovered_map(mutated)  # (H,W), uncovered=1
        uncovered = np.asarray(uncovered)

        cand_yx = np.argwhere((uncovered == 1) & installable01)  # (N,2) as (y,x)
        if cand_yx.size == 0:
            return None

        # 후보 필터링 (중복 제거)
        filtered: List[Gene] = []
        for (y, x) in cand_yx:
            g = (int(x), int(y))
            if g not in exist:
                filtered.append(g)

        if not filtered:
            return None
        return random.choice(filtered)

    # -------------------------
    # helper: coverage evaluate (%)
    # -------------------------
    def _coverage_percent(inner: Chromosome) -> float:
        _, cov, _ = evaluator.evaluate(inner)
        return float(cov)

    # -------------------------
    # operation choice
    # -------------------------
    ops = []
    if p_add > 0:
        ops.append(("add", float(p_add)))
    if p_del > 0:
        ops.append(("del", float(p_del)))
    if p_move > 0:
        ops.append(("move", float(p_move)))

    if not ops:
        return mutated  # no-op

    r = random.random() * sum(w for _, w in ops)
    acc = 0.0
    op = ops[-1][0]
    for name, w in ops:
        acc += w
        if r <= acc:
            op = name
            break

    # max_total_sensors 체크용
    corner_cnt = len(corners)
    if max_total_sensors is not None:
        max_inner_allowed = max(0, int(max_total_sensors) - corner_cnt)
    else:
        max_inner_allowed = None

    # -------------------------
    # ADD
    # -------------------------
    if op == "add":
        if max_inner_allowed is not None and len(mutated) >= max_inner_allowed:
            return mutated  # 더 못 늘림

        g = _sample_add_gene(existing)
        if g is None:
            return mutated

        mutated.append(g)
        return mutated

    # -------------------------
    # DEL
    # -------------------------
    if op == "del":
        if not mutated:
            return mutated

        # 삭제 전 coverage 저장(옵션)
        cov_before = _coverage_percent(mutated) if (min_coverage_keep is not None) else None

        idx = random.randrange(len(mutated))
        removed = mutated.pop(idx)

        if min_coverage_keep is not None:
            cov_after = _coverage_percent(mutated)
            # 기준 미달이면 롤백
            if cov_after < float(min_coverage_keep):
                mutated.insert(idx, removed)  # rollback
                return mutated

            # 또는 "삭제로 손해가 너무 크면" 기준을 추가하고 싶으면 여기서 비교 가능
            # ex) if cov_before - cov_after > 2.0: rollback ...

        return mutated

    # -------------------------
    # MOVE (DEL + ADD)
    # -------------------------
    if op == "move":
        if not mutated:
            # 이동할 게 없으면 add로 대체
            if max_inner_allowed is not None and len(mutated) >= max_inner_allowed:
                return mutated
            g = _sample_add_gene(existing)
            if g is None:
                return mutated
            mutated.append(g)
            return mutated

        # coverage 저장(옵션)
        cov_before = _coverage_percent(mutated) if (min_coverage_keep is not None) else None

        # 1) delete one
        idx = random.randrange(len(mutated))
        removed = mutated.pop(idx)

        # exist 업데이트(removed는 제거했으니 다시 추가 가능)
        exist2 = set(mutated)
        exist2.update(corners)

        # 2) add one
        if max_inner_allowed is not None and len(mutated) >= max_inner_allowed:
            # add 불가 -> 삭제 취소
            mutated.insert(idx, removed)
            return mutated

        g = _sample_add_gene(exist2)
        if g is None:
            # add 실패 -> 삭제 취소
            mutated.insert(idx, removed)
            return mutated

        mutated.append(g)

        # coverage 조건 검사
        if min_coverage_keep is not None:
            cov_after = _coverage_percent(mutated)
            if cov_after < float(min_coverage_keep):
                # rollback: 원상복구 (added 제거 + removed 복원)
                mutated.pop()              # remove g
                mutated.insert(idx, removed)
                return mutated

        return mutated

    # fallback
    return mutated
