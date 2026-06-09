from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

PathLike = str | Path
RunRecord = Tuple[Path, Dict[str, Any]]


def loadJson(path: PathLike) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def isValid(run: Any) -> bool:
    return isinstance(run, dict) and isinstance(run.get("generations"), list) and isinstance(run.get("final"), dict)


def loadRecords(root_dir: PathLike) -> List[RunRecord]:
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"not found: {root}")

    files = sorted(root.rglob("*.json"))
    if not files:
        raise FileNotFoundError(f"no json files under: {root}")

    records: List[RunRecord] = []
    for path in files:
        try:
            run = loadJson(path)
        except Exception:
            continue
        if isValid(run):
            records.append((path, run))

    if not records:
        raise ValueError("no valid run json (must contain 'generations' and 'final').")
    return records


def loadRuns(root_dir: PathLike) -> List[Dict[str, Any]]:
    return [run for _, run in loadRecords(root_dir)]


def meanVal(vals: Sequence[float], *, default: float = 0.0) -> float:
    clean = [float(v) for v in vals if not math.isnan(float(v))]
    return float(sum(clean) / len(clean)) if clean else float(default)


def stdVal(vals: Sequence[float], *, sample: bool = False, default: float = 0.0) -> float:
    clean = [float(v) for v in vals if not math.isnan(float(v))]
    if not clean:
        return float(default)
    if len(clean) == 1:
        return 0.0
    avg = meanVal(clean)
    denom = len(clean) - 1 if sample else len(clean)
    return float((sum((v - avg) ** 2 for v in clean) / denom) ** 0.5)


def resultDir(
    *,
    results_root: PathLike = "__RESULTS__",
    algorithm: str = "ga",
    map_name: str = "gangjin.down",
) -> Path:
    return Path(results_root) / str(algorithm) / str(map_name)


def bandKey(band: str) -> int:
    try:
        return int(str(band).split("-")[0])
    except Exception:
        return 0


def listBands(
    *,
    results_root: PathLike = "__RESULTS__",
    algorithm: str = "ga",
    map_name: str = "gangjin.down",
) -> List[str]:
    root = resultDir(results_root=results_root, algorithm=algorithm, map_name=map_name)
    if not root.exists():
        return []
    return sorted([p.name for p in root.iterdir() if p.is_dir()], key=bandKey)


def loadAlgoRuns(
    *,
    results_root: PathLike = "__RESULTS__",
    algorithm: str = "ga",
    map_name: str = "gangjin.down",
    seed_band: str | None = None,
) -> List[RunRecord]:
    root = resultDir(results_root=results_root, algorithm=algorithm, map_name=map_name)
    if seed_band is not None:
        root = root / str(seed_band)
    return loadRecords(root)


def finalPoints(run: Dict[str, Any]) -> List[Any]:
    final = run.get("final", {})
    best = final.get("best_solution", [])
    corners = final.get("corner_points", [])
    return list(best) + list(corners)


def cornerCount(run: Dict[str, Any]) -> int:
    final = run.get("final", {}) or {}
    corners = final.get("corner_points", []) or []
    if isinstance(corners, list):
        return len(corners)
    extra = run.get("extra", {}).get("final", {}) or {}
    return int(extra.get("n_corner", 0) or 0)
