# /workspace/Tools/Logger.py
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


def _sanitize_dirname(name: str) -> str:
    """
    디렉토리명 안전화 (OS/FS 호환)
    - 슬래시/역슬래시 등 경로 문자 제거
    - 공백은 언더스코어로 치환
    """
    name = str(name).strip().replace(" ", "_")
    for ch in ["/", "\\", ":", "*", "?", '"', "<", ">", "|"]:
        name = name.replace(ch, "_")
    return name if name else "unknown_map"


def _next_result_path(base_dir: str, map_name: str, ts: datetime) -> Tuple[str, str, str]:
    """
    저장 규칙:
      root: /workspace/__RESULTS__
      dir : /workspace/__RESULTS__/<map_name>/
      file: yyyymmdd_HHMMSS.json
    동일 초 충돌 시:
      yyyymmdd_HHMMSS_01.json, _02.json ...
    return (map_dir, file_stem, out_path)
    """
    _ensure_dir(base_dir)

    safe_map = _sanitize_dirname(map_name)
    map_dir = os.path.join(base_dir, safe_map)
    _ensure_dir(map_dir)

    file_stem = ts.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(map_dir, f"{file_stem}.json")
    if not os.path.exists(out_path):
        return map_dir, file_stem, out_path

    i = 1
    while True:
        stem_i = f"{file_stem}_{i:02d}"
        out_i = os.path.join(map_dir, f"{stem_i}.json")
        if not os.path.exists(out_i):
            return map_dir, stem_i, out_i
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
    n_inner: int
    best_fitness: float
    best_coverage: float


class GAJsonLogger:
    """
    - 세대별 통계 저장
    - 종료 시 최종 결과 저장
    - 저장 경로:
        /workspace/__RESULTS__/<map_name>/yyyymmdd_HHMMSS.json
    """

    def __init__(
        self,
        *,
        map_name: str,
        base_dir: str = "/workspace/__RESULTS__",
        meta: Optional[Dict[str, Any]] = None,
        point_format: str = "tuple_str",
        sort_points: bool = False,
    ):
        self.t0 = _kst_now()
        self.map_name = map_name
        self.base_dir = base_dir
        self.meta = meta or {}

        self.point_format = point_format
        self.sort_points = bool(sort_points)

        self.map_dir, self.run_name, self.out_path = _next_result_path(
            self.base_dir, self.map_name, self.t0
        )

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
        n_inner = int(len(best_pts))

        self.generations.append(
            GenStats(
                gen=int(gen),
                sensors_min=float(sensors_min),
                sensors_max=float(sensors_max),
                sensors_avg=float(sensors_avg),
                fitness_min=float(fitness_min),
                fitness_max=float(fitness_max),
                fitness_avg=float(fitness_avg),
                best_solution=best_fmt,
                n_inner=n_inner,
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
            "run_name": self.run_name,  # yyyymmdd_HHMMSS(또는 _01)
            "map_name": self.map_name,
            "map_dir": self.map_dir,
            "created_at_kst": self.t0.isoformat(),
            "finished_at_kst": t1.isoformat(),
            "elapsed_sec": float(elapsed),
            "meta": _to_jsonable(self.meta),
            "generations": _to_jsonable([asdict(g) for g in self.generations]),
            "final": _to_jsonable(
                {
                    "best_solution": self.final_best_solution,
                    "corner_points": _fmt_points(corner_pts, self.point_format),
                    "fitness": self.final_fitness,
                    "coverage": self.final_coverage,
                    "n_inner": int(len(best_pts)),
                    "n_corner": int(len(corner_pts)),
                    "n_total": int(len(best_pts) + len(corner_pts)),
                    "elapsed_sec": float(elapsed),
                }
            ),
        }

        if extra:
            payload["extra"] = _to_jsonable(extra)

        # map_dir는 __init__에서 생성/확정, 여기서는 파일만 저장
        with open(self.out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        return self.out_path
