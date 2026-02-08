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


def bandSortKey(band: str) -> int:
    """
    band 문자열이 '40-60' 형태일 때,
    앞 숫자(40)를 기준으로 정렬하기 위한 key
    """
    try:
        return int(band.split("-")[0])
    except Exception:
        return 0


def plotSensorsAvgTrend(
    root_dir: str,
    map_name: str,
    band_list: Optional[List[str]] = None,
    show_std: bool = True,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (6.5, 4.0),
    dpi: int = 300,
    save_path: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:

    root = Path(root_dir)

    # ✅ band 내림차순 정렬 (숫자 기준)
    if band_list is None:
        band_list = [p.name for p in root.iterdir() if p.is_dir()]
    band_list = sorted(band_list, key=bandSortKey, reverse=True)

    # ✅ 흑백 논문용 스타일 세트
    linestyle_cycle = ["-", "--", "-.", ":"]
    marker_cycle = ["o", "s", "^", "D", "x", "+"]

    result: Dict[str, Dict[str, Any]] = {}

    fig = plt.figure(figsize=figsize, dpi=dpi)

    for idx, band in enumerate(band_list):
        run_series = collectRunsSensorsAvg(root_dir, band, map_name)
        mean_list, std_list = buildTrend(run_series)
        if not mean_list:
            continue

        x = list(range(1, len(mean_list) + 1))

        linestyle = linestyle_cycle[idx % len(linestyle_cycle)]
        marker = marker_cycle[idx % len(marker_cycle)]

        plt.plot(
            x,
            mean_list,
            label=band,
            linestyle=linestyle,
            marker=marker,
            markersize=4,
            markevery=max(len(x) // 10, 1),  # 마커 과밀 방지
            linewidth=1.5,
        )

        if show_std:
            low = [m - s for m, s in zip(mean_list, std_list)]
            high = [m + s for m, s in zip(mean_list, std_list)]
            plt.fill_between(x, low, high, alpha=0.15)

        result[band] = {
            "runs": len(run_series),
            "gens": len(mean_list),
            "mean": mean_list,
            "std": std_list,
        }

    plt.xlabel("Generation")
    plt.ylabel("Number of Sensors")
    plt.title(title or f"Convergence of Sensors")
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False)

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    plt.show()
    plt.close(fig)

    return result


def analyze_change_by_generation(
    info: Dict[str, Dict[str, Any]],
    threshold: float = 0.5,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    각 초기값(band)이 변화량 threshold 미만으로 수렴한 세대를 구하고,
    **모든 band가 한번에 수렴하는 세대** = 그 중 가장 늦은 세대를 구합니다.

    변화량 = 연속 세대 간 평균값 차이의 절대값.
    threshold 미만이면 "수렴"으로 간주합니다.

    Args:
        info: plotSensorsAvgTrend() 반환값 (band -> {mean, std, runs, gens})
        threshold: 변화량이 이 값 미만이면 수렴으로 간주 (센서 수 기준)
        verbose: True면 결과를 print

    Returns:
        {
            "convergence_gen": int,  # 모든 band가 한번에 수렴하는 세대
            "by_band": { band -> {"gen_from", "max_change", "abs_diffs"} 또는 None }
        }
    """
    by_band: Dict[str, Optional[Dict[str, Any]]] = {}
    for band, d in info.items():
        mean_list = d.get("mean", [])
        if len(mean_list) < 2:
            by_band[band] = None
            continue
        abs_diffs = [
            abs(mean_list[i + 1] - mean_list[i])
            for i in range(len(mean_list) - 1)
        ]
        last_big = None
        for i, ad in enumerate(abs_diffs):
            if ad >= threshold:
                last_big = i
        if last_big is None:
            gen_from = 2
        else:
            gen_from = last_big + 2
        by_band[band] = {
            "gen_from": gen_from,
            "max_change": max(abs_diffs),
            "abs_diffs": abs_diffs,
        }

    convergence_gen = 0
    for r in by_band.values():
        if r is not None and r["gen_from"] > convergence_gen:
            convergence_gen = r["gen_from"]

    if verbose:
        print(
            f"변화량 < {threshold} 센서일 때 수렴으로 간주.\n"
            f"**모든 초기값(band)이 한번에 수렴하는 유전 세대: {convergence_gen}세대**"
        )
        def _gen_key(b: str) -> float:
            r = by_band.get(b)
            return r["gen_from"] if r is not None else 0.0

        for band in sorted(by_band, key=_gen_key):
            r = by_band[band]
            if r is None:
                print(f"  {band}: 데이터 부족")
            else:
                print(
                    f"  {band}: {r['gen_from']}세대부터 수렴 "
                    f"(최대 변화량 {r['max_change']:.3f})"
                )

    return {"convergence_gen": convergence_gen, "by_band": by_band}
