# /workspace/Tools/Logger.py  (또는 네가 저장한 경로에 그대로 덮어쓰기)
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

Point = Tuple[int, int]


def _to_jsonable(obj: Any) -> Any:
    """tuple/list/set/numpy 등을 JSON 친화적으로 변환"""
    try:
        import numpy as np  # optional
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass

    if isinstance(obj, tuple):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, list):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, set):
        return [_to_jsonable(x) for x in sorted(obj)]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}

    if hasattr(obj, "__dataclass_fields__"):
        return _to_jsonable(asdict(obj))

    return obj


def _kst_now() -> datetime:
    """
    KST 시도 → 실패 시 local time → 최종 fallback UTC
    (tzdata 없는 Docker 환경 대응)
    """
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("Asia/Seoul"))
    except Exception:
        try:
            return datetime.now().astimezone()
        except Exception:
            return datetime.utcnow()


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _next_run_dir(base_dir: str, date_yyyymmdd: str, map_name: str) -> Tuple[str, str]:
    """
    __RESULTS__/yyyymmdd_mapname_01 형태로 디렉토리 생성.
    이미 있으면 02,03...로 증가.
    return (run_dir, run_name)
    """
    _ensure_dir(base_dir)
    i = 1
    while True:
        run_name = f"{date_yyyymmdd}_{map_name}_{i:02d}"
        run_dir = os.path.join(base_dir, run_name)
        if not os.path.exists(run_dir):
            os.makedirs(run_dir, exist_ok=False)
            return run_dir, run_name
        i += 1


def _as_points(points: Any) -> List[Point]:
    """입력 포인트를 (int,int) 리스트로 강제 변환"""
    if points is None:
        return []
    out: List[Point] = []
    for p in points:
        if p is None:
            continue
        x, y = p
        out.append((int(x), int(y)))
    return out


def _fmt_points(points: List[Point], fmt: str) -> Any:
    """
    fmt:
      - "tuple_str": ["(x,y)", ...]  (가독성 최고, 기본)
      - "list": [[x,y], ...]        (기존 호환)
      - "flat": [x,y,x,y,...]       (가장 compact)
    """
    if fmt == "list":
        return [[x, y] for x, y in points]
    if fmt == "flat":
        flat: List[int] = []
        for x, y in points:
            flat.extend([x, y])
        return flat
    # default: tuple_str
    return [f"({x},{y})" for x, y in points]


@dataclass
class GenStats:
    gen: int
    sensors_min: float
    sensors_max: float
    sensors_avg: float
    fitness_min: float
    fitness_max: float
    fitness_avg: float
    best_solution: Any
    best_fitness: float
    best_coverage: float


class GAJsonLogger:
    """
    - 세대별 통계 저장
    - 종료 시 최종 결과 저장
    - __RESULTS__/yyyymmdd_mapname_01/yyyymmdd_mapname_01.json 생성
    """

    def __init__(
        self,
        *,
        map_name: str,
        base_dir: str = "__RESULTS__",
        meta: Optional[Dict[str, Any]] = None,
        point_format: str = "tuple_str",  # ✅ 추가
        sort_points: bool = False,        # ✅ 추가 (보기용 정렬)
    ):
        self.t0 = _kst_now()
        self.map_name = map_name
        self.base_dir = base_dir
        self.meta = meta or {}

        self.point_format = point_format
        self.sort_points = bool(sort_points)

        date = self.t0.strftime("%Y%m%d")
        self.run_dir, self.run_name = _next_run_dir(self.base_dir, date, self.map_name)
        self.out_path = os.path.join(self.run_dir, f"{self.run_name}.json")

        self.generations: List[GenStats] = []

        # 최종 결과
        self.final_best_solution: Any = None
        self.final_corner_points: Optional[List[Point]] = None
        self.final_fitness: Optional[float] = None
        self.final_coverage: Optional[float] = None

    def _maybe_sort(self, pts: List[Point]) -> List[Point]:
        if not self.sort_points:
            return pts
        return sorted(pts, key=lambda p: (p[0], p[1]))

    def log_generation(
        self,
        *,
        gen: int,
        sensors_min: float,
        sensors_max: float,
        sensors_avg: float,
        fitness_min: float,
        fitness_max: float,
        fitness_avg: float,
        best_solution: Any,
        best_fitness: float,
        best_coverage: float,
    ) -> None:
        best_pts = self._maybe_sort(_as_points(best_solution))
        best_fmt = _fmt_points(best_pts, self.point_format)

        self.generations.append(
            GenStats(
                gen=int(gen),
                sensors_min=float(sensors_min),
                sensors_max=float(sensors_max),
                sensors_avg=float(sensors_avg),
                fitness_min=float(fitness_min),
                fitness_max=float(fitness_max),
                fitness_avg=float(fitness_avg),
                best_solution=best_fmt,               # ✅ 포맷 적용
                best_fitness=float(best_fitness),
                best_coverage=float(best_coverage),
            )
        )

    def finalize(
        self,
        *,
        best_solution: Any,
        corner_points: List[Point],
        fitness: float,
        coverage: float,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        t1 = _kst_now()
        elapsed = (t1 - self.t0).total_seconds()

        best_pts = self._maybe_sort(_as_points(best_solution))
        corner_pts = self._maybe_sort(_as_points(corner_points))

        self.final_best_solution = _fmt_points(best_pts, self.point_format)
        self.final_corner_points = corner_pts
        self.final_fitness = float(fitness)
        self.final_coverage = float(coverage)

        payload: Dict[str, Any] = {
            "run_name": self.run_name,
            "map_name": self.map_name,
            "created_at_kst": self.t0.isoformat(),
            "finished_at_kst": t1.isoformat(),
            "elapsed_sec": float(elapsed),
            "meta": _to_jsonable(self.meta),
            "generations": _to_jsonable([asdict(g) for g in self.generations]),
            "final": _to_jsonable(
                {
                    "best_solution": self.final_best_solution,               # ✅ 포맷 적용
                    "corner_points": _fmt_points(corner_pts, self.point_format),  # ✅ 포맷 적용
                    "fitness": self.final_fitness,
                    "coverage": self.final_coverage,
                    "elapsed_sec": float(elapsed),                           # ✅ final에도 포함
                }
            ),
        }

        if extra:
            payload["extra"] = _to_jsonable(extra)

        with open(self.out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        return self.out_path
