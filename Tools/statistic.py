from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def loadRuns(root_dir: str) -> List[Dict[str, Any]]:
    root_path = Path(root_dir)
    if not root_path.exists():
        raise FileNotFoundError(f"not found: {root_path}")

    run_list: List[Dict[str, Any]] = []
    file_list = sorted(root_path.rglob("*.json"))

    if len(file_list) == 0:
        raise FileNotFoundError(f"no json files under: {root_path}")

    for file_path in file_list:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                run = json.load(f)

            if isinstance(run, dict) and "generations" in run and "final" in run:
                run_list.append(run)
        except Exception:
            continue

    if len(run_list) == 0:
        raise ValueError("no valid run json (must contain 'generations' and 'final').")

    return run_list


def getGen100(run: Dict[str, Any]) -> Dict[str, Any]:
    gen_list = run.get("generations", [])
    if not isinstance(gen_list, list) or len(gen_list) < 100:
        raise ValueError(f"run has insufficient generations: {len(gen_list)}")

    gen_data = gen_list[99]
    if not isinstance(gen_data, dict) or gen_data.get("gen") != 100:
        for item in gen_list:
            if isinstance(item, dict) and item.get("gen") == 100:
                return item
        raise ValueError("cannot find gen=100 in generations.")
    return gen_data


def getCornerCnt(run: Dict[str, Any]) -> int:
    final_data = run.get("final", {})
    corner_list = final_data.get("corner_points", [])

    if isinstance(corner_list, list):
        return int(len(corner_list))

    extra_data = run.get("extra", {}).get("final", {})
    n_corner = extra_data.get("n_corner", 0)
    return int(n_corner) if isinstance(n_corner, (int, float)) else 0


def getTotalCnt(run: Dict[str, Any]) -> int:
    extra_data = run.get("extra", {}).get("final", {})
    n_total = extra_data.get("n_final_points", None)

    if isinstance(n_total, (int, float)):
        return int(n_total)

    final_data = run.get("final", {})
    best_list = final_data.get("best_solution", [])
    corner_list = final_data.get("corner_points", [])

    best_cnt = len(best_list) if isinstance(best_list, list) else 0
    corner_cnt = len(corner_list) if isinstance(corner_list, list) else 0
    return int(best_cnt + corner_cnt)


def getCornerSec(run: Dict[str, Any]) -> Optional[float]:
    """
    corner-only time (sec).
    여러 로그 버전을 고려해서 후보 키를 순차 탐색.
    없으면 None 반환 (통계에서 제외).
    """
    # 1) 최상위 키 후보
    key_list = [
        "corner_elapsed_sec",
        "corner_elapsed",
        "corner_sec",
    ]
    for key in key_list:
        val = run.get(key, None)
        if isinstance(val, (int, float)):
            return float(val)

    # 2) meta/corner 내부에 시간이 기록되는 케이스
    meta_corner = run.get("meta", {}).get("corner", {})
    if isinstance(meta_corner, dict):
        for key in ["elapsed_sec", "elapsed", "time_sec", "sec"]:
            val = meta_corner.get(key, None)
            if isinstance(val, (int, float)):
                return float(val)

    # 3) extra에 기록되는 케이스
    extra_corner = run.get("extra", {}).get("corner", {})
    if isinstance(extra_corner, dict):
        for key in ["elapsed_sec", "elapsed", "time_sec", "sec"]:
            val = extra_corner.get(key, None)
            if isinstance(val, (int, float)):
                return float(val)

    return None


def getGaSec(run: Dict[str, Any]) -> Optional[float]:
    """
    GA time (sec).
    우선순위:
      1) final.elapsed_sec
      2) top-level elapsed_sec
      3) meta.ga_run.elapsed_sec (있다면)
    """
    final_data = run.get("final", {})
    val = final_data.get("elapsed_sec", None)
    if isinstance(val, (int, float)):
        return float(val)

    val = run.get("elapsed_sec", None)
    if isinstance(val, (int, float)):
        return float(val)

    ga_run = run.get("meta", {}).get("ga_run", {})
    if isinstance(ga_run, dict):
        val = ga_run.get("elapsed_sec", None)
        if isinstance(val, (int, float)):
            return float(val)

    return None


def calcMean(vals: List[float]) -> float:
    if len(vals) == 0:
        return 0.0
    return float(sum(vals) / len(vals))


def calcStd(vals: List[float], mean: float) -> float:
    """
    population std (divide by N)
    """
    if len(vals) == 0:
        return 0.0
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return float(var ** 0.5)


def calcStats(
    root_dir: str,
) -> Tuple[int, float, float, float, float, float, float, float, float, float, float]:
    """
    returns:
      run_cnt,
      cov_mean, cov_std,
      corner_mean, corner_std,
      total_mean, total_std,
      corner_sec_mean, corner_sec_std,
      ga_sec_mean, ga_sec_std
    """
    run_list = loadRuns(root_dir)
    run_cnt = len(run_list)

    cov_list: List[float] = []
    corner_list: List[float] = []
    total_list: List[float] = []
    corner_sec_list: List[float] = []
    ga_sec_list: List[float] = []

    for run in run_list:
        gen100 = getGen100(run)

        cov = gen100.get("best_coverage", None)
        if isinstance(cov, (int, float)):
            cov_list.append(float(cov))
        else:
            fin_cov = run.get("final", {}).get("coverage", None)
            if isinstance(fin_cov, (int, float)):
                cov_list.append(float(fin_cov))

        corner_list.append(float(getCornerCnt(run)))
        total_list.append(float(getTotalCnt(run)))

        corner_sec = getCornerSec(run)
        if isinstance(corner_sec, (int, float)):
            corner_sec_list.append(float(corner_sec))

        ga_sec = getGaSec(run)
        if isinstance(ga_sec, (int, float)):
            ga_sec_list.append(float(ga_sec))

    cov_mean = calcMean(cov_list)
    corner_mean = calcMean(corner_list)
    total_mean = calcMean(total_list)

    cov_std = calcStd(cov_list, cov_mean)
    corner_std = calcStd(corner_list, corner_mean)
    total_std = calcStd(total_list, total_mean)

    corner_sec_mean = calcMean(corner_sec_list)
    corner_sec_std = calcStd(corner_sec_list, corner_sec_mean)

    ga_sec_mean = calcMean(ga_sec_list)
    ga_sec_std = calcStd(ga_sec_list, ga_sec_mean)

    return (
        run_cnt,
        cov_mean,
        cov_std,
        corner_mean,
        corner_std,
        total_mean,
        total_std,
        corner_sec_mean,
        corner_sec_std,
        ga_sec_mean,
        ga_sec_std,
    )
