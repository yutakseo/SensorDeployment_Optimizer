from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from Analysis.internal.result_io import finalPoints, loadRuns as loadRunList, meanVal, stdVal

TARGET_GENERATION = 100


def loadRuns(root_dir: str) -> List[Dict[str, Any]]:
    """Load valid run JSON payloads from a result directory."""
    return loadRunList(root_dir)


def getGen100(run: Dict[str, Any]) -> Dict[str, Any]:
    generations = run.get("generations", [])
    if not isinstance(generations, list) or len(generations) < TARGET_GENERATION:
        raise ValueError(f"run has insufficient generations: {len(generations)}")

    generation = generations[TARGET_GENERATION - 1]
    if not isinstance(generation, dict) or generation.get("gen") != TARGET_GENERATION:
        for item in generations:
            if isinstance(item, dict) and item.get("gen") == TARGET_GENERATION:
                return item
        raise ValueError(f"cannot find gen={TARGET_GENERATION} in generations.")
    return generation


def getCoverage(run: Dict[str, Any]) -> Optional[float]:
    generations = run.get("generations", [])
    if isinstance(generations, list):
        for item in generations:
            if not isinstance(item, dict) or item.get("gen") != TARGET_GENERATION:
                continue
            coverage = item.get("best_coverage", None)
            if isinstance(coverage, (int, float)):
                return float(coverage)
            break

    final_data = run.get("final", {})
    if isinstance(final_data, dict):
        coverage = final_data.get("coverage", None)
        if isinstance(coverage, (int, float)):
            return float(coverage)

    if isinstance(generations, list) and generations:
        last_generation = generations[-1]
        if isinstance(last_generation, dict):
            coverage = last_generation.get("best_coverage", None)
            if isinstance(coverage, (int, float)):
                return float(coverage)
    return None


def getCornerCnt(run: Dict[str, Any]) -> int:
    final_data = run.get("final", {})
    corner_points = final_data.get("corner_points", [])

    if isinstance(corner_points, list):
        return int(len(corner_points))

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


def getOptSec(run: Dict[str, Any]) -> Optional[float]:
    """
    Optimizer time (sec).
    우선순위:
      1) final.elapsed_sec
      2) top-level elapsed_sec
      3) meta.optimizer_run.elapsed_sec or legacy meta.ga_run.elapsed_sec (있다면)
    """
    final_data = run.get("final", {})
    val = final_data.get("elapsed_sec", None)
    if isinstance(val, (int, float)):
        return float(val)

    val = run.get("elapsed_sec", None)
    if isinstance(val, (int, float)):
        return float(val)

    meta = run.get("meta", {})
    for key in ("optimizer_run", "ga_run"):
        optimizer_run = meta.get(key, {})
        if not isinstance(optimizer_run, dict):
            continue
        val = optimizer_run.get("elapsed_sec", None)
        if isinstance(val, (int, float)):
            return float(val)

    return None


def calcMean(values: List[float]) -> float:
    return meanVal(values, default=0.0)


def calcStd(values: List[float], mean_value: float) -> float:
    """
    population std (divide by N)
    """
    del mean_value
    return stdVal(values, sample=False, default=0.0)


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
    runs = loadRuns(root_dir)
    run_count = len(runs)

    cov_list: List[float] = []
    corner_list: List[float] = []
    total_list: List[float] = []
    corner_sec_list: List[float] = []
    ga_sec_list: List[float] = []

    for run in runs:
        coverage = getCoverage(run)
        if coverage is not None:
            cov_list.append(coverage)

        corner_list.append(float(getCornerCnt(run)))
        total_list.append(float(getTotalCnt(run)))

        corner_sec = getCornerSec(run)
        if isinstance(corner_sec, (int, float)):
            corner_sec_list.append(float(corner_sec))

        optimizer_sec = getOptSec(run)
        if isinstance(optimizer_sec, (int, float)):
            ga_sec_list.append(float(optimizer_sec))

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
        run_count,
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


def getFinalPoints(run: Dict[str, Any]) -> List[Any]:
    """run 한 개에서 최종 포인트 목록(best_solution + corner_points) 반환."""
    return finalPoints(run)


def reportCluster(
    root_dir: str,
    map_name: str,
    grid_m: float = 5.0,
    verbose: bool = True,
) -> Optional[Dict[str, float]]:
    """
    해당 맵 디렉터리의 모든 run에 대해 평균 군집거리(가장 가까운 센서까지의 거리 평균)를
    run별로 구한 뒤, 그 평균·표준편차를 실제 거리(m)로 출력합니다.
    grid_m: 1그리드당 미터 (기본 5m).
    """
    from Analysis.internal.distance_metrics import meanNearest

    try:
        runs = loadRuns(root_dir)
    except FileNotFoundError as e:
        if verbose:
            print(f"[{map_name}] 결과 디렉터리가 없습니다: {e}")
            print("  → experiment.py로 해당 맵 실험을 먼저 실행하세요.")
        return None
    dist_list: List[float] = []
    for run in runs:
        pts = getFinalPoints(run)
        if len(pts) >= 2:
            dist_list.append(meanNearest(pts))

    if not dist_list:
        if verbose:
            print(f"[{map_name}] 유효한 run 없음 (최종 포인트 2개 미만)")
        return None

    n_runs = len(dist_list)
    mean_d = sum(dist_list) / n_runs
    std_d = (
        (sum((d - mean_d) ** 2 for d in dist_list) / (n_runs - 1)) ** 0.5
        if n_runs > 1
        else 0.0
    )
    min_d = min(dist_list)
    max_d = max(dist_list)
    if verbose:
        print(f"[{map_name}] total runs: {n_runs}")
        print(
            "[final] 평균 군집거리 mean ± std: "
            f"{mean_d * grid_m:.3f} ± {std_d * grid_m:.3f} m"
        )
        print(
            "[final] 군집거리 min / max: "
            f"{min_d * grid_m:.3f} / {max_d * grid_m:.3f} m"
        )
    return {
        "mean": mean_d * grid_m,
        "std": std_d * grid_m,
        "min": min_d * grid_m,
        "max": max_d * grid_m,
        "n_runs": n_runs,
    }


def printStats(root_dir: str) -> None:
    """Print calcStats output in a readable format."""
    try:
        (
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
        ) = calcStats(root_dir)
    except FileNotFoundError as e:
        print(f"결과 디렉터리가 없습니다: {e}")
        print("  → experiment.py로 해당 맵 실험을 먼저 실행하세요.")
        return
    print(f"total runs: {run_cnt}")
    print(f"[coverage] mean ± std: {cov_mean:.4f} ± {cov_std:.4f}")
    print(f"[final] corner sensors mean ± std: {corner_mean:.2f} ± {corner_std:.2f}")
    print(f"[final] total sensors mean ± std: {total_mean:.2f} ± {total_std:.2f}")
    print(f"[time] corner mean ± std (sec): {corner_sec_mean:.3f} ± {corner_sec_std:.3f}")
    print(f"[time] optimizer mean ± std (sec): {ga_sec_mean:.3f} ± {ga_sec_std:.3f}")


getGaSec = getOptSec
