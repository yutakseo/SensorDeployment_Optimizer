from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


def loadJson(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extractSensorsAvgSeries(obj: Dict[str, Any]) -> List[float]:
    """
    JSON 1개(run)에서 generation별 sensors_avg 시퀀스를 추출.
    네 샘플 구조: obj["generations"] = [{"gen":1, "sensors_avg":...}, ...]
    """
    gens = obj.get("generations", None)
    if not isinstance(gens, list) or not gens:
        return []

    series: List[float] = []
    for rec in gens:
        if not isinstance(rec, dict):
            continue
        val = rec.get("sensors_avg", None)
        if isinstance(val, (int, float)) and not (isinstance(val, float) and math.isnan(val)):
            series.append(float(val))
        else:
            # sensors_avg가 없으면 해당 run은 버리거나(권장), NaN 처리로 바꿀 수도 있음
            return []
    return series


def meanStd(vals: List[float]) -> Tuple[float, float]:
    if not vals:
        return (float("nan"), float("nan"))
    m = sum(vals) / len(vals)
    if len(vals) == 1:
        return (m, 0.0)
    v = sum((x - m) ** 2 for x in vals) / (len(vals) - 1)
    return (m, math.sqrt(v))


def buildTrend(run_series: List[List[float]]) -> Tuple[List[float], List[float]]:
    """
    여러 run의 sensors_avg 시퀀스를 min generation 길이로 맞춰(truncate)서
    세대별 평균/표준편차 산출.
    """
    run_series = [s for s in run_series if s]
    if not run_series:
        return ([], [])

    min_gen = min(len(s) for s in run_series)
    if min_gen <= 0:
        return ([], [])

    run_series = [s[:min_gen] for s in run_series]

    mean_list: List[float] = []
    std_list: List[float] = []
    for gi in range(min_gen):
        col = [s[gi] for s in run_series]
        m, sd = meanStd(col)
        mean_list.append(m)
        std_list.append(sd)
    return (mean_list, std_list)


def collectRunsSensorsAvg(root_dir: str, band: str, map_name: str) -> List[List[float]]:
    """
    {root_dir}/{band}/{map_name}/*.json 전부 로드해서 sensors_avg 시퀀스 리스트로 반환
    """
    map_dir = Path(root_dir) / band / map_name
    if not map_dir.exists():
        return []

    run_series: List[List[float]] = []
    for jp in sorted(map_dir.glob("*.json")):
        obj = loadJson(jp)
        series = extractSensorsAvgSeries(obj)
        if series:
            run_series.append(series)
    return run_series


def plotSensorsAvgTrend(
    root_dir: str,
    map_name: str,
    band_list: Optional[List[str]] = None,
    show_std: bool = True,
    title: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    오버레이 플롯: band별 '세대별 sensors_avg의 run평균' 곡선을 한 장에 겹쳐 그림.
    반환: band별 runs/gens/mean/std
    """
    root = Path(root_dir)
    if band_list is None:
        band_list = sorted([p.name for p in root.iterdir() if p.is_dir()])

    result: Dict[str, Dict[str, Any]] = {}

    plt.figure()

    for band in band_list:
        run_series = collectRunsSensorsAvg(root_dir, band, map_name)
        mean_list, std_list = buildTrend(run_series)
        if not mean_list:
            continue

        x = list(range(1, len(mean_list) + 1))
        plt.plot(x, mean_list, label=f"{band}")

        if show_std:
            low = [m - s for m, s in zip(mean_list, std_list)]
            high = [m + s for m, s in zip(mean_list, std_list)]
            plt.fill_between(x, low, high, alpha=0.2)

        result[band] = {
            "runs": len(run_series),
            "gens": len(mean_list),
            "mean": mean_list,
            "std": std_list,
        }

    plt.xlabel("Generation")
    plt.ylabel("Sensors")
    plt.title(title or f"{map_name} | sensors convergence by init band")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    return result
