from __future__ import annotations

import json
import math
from os import PathLike
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, TypeAlias

import matplotlib.pyplot as plt
import numpy as np

from Analysis.distance_metrics import asPoints
from Analysis.result_io import (
    bandKey,
    listBands as _list_bands,
    loadAlgoRuns,
    loadJson as readJson,
    meanVal,
    resultDir,
    stdVal,
)
from Engine.map_loader import MapLoader
from InnerDeployment.geometry import circle_offsets

PathInput: TypeAlias = str | PathLike[str]


def loadJson(path: Path) -> Dict[str, Any]:
    return readJson(path)


def getAvgSeries(run: Dict[str, Any]) -> List[float]:
    generations = run.get("generations", None)
    if not isinstance(generations, list) or not generations:
        return []

    series: List[float] = []
    for generation in generations:
        if not isinstance(generation, dict):
            continue
        value = generation.get("sensors_avg", None)
        if isinstance(value, (int, float)) and not (isinstance(value, float) and math.isnan(value)):
            series.append(float(value))
        else:
            return []
    return series


def meanStd(values: List[float]) -> Tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    return meanVal(values, default=float("nan")), stdVal(values, sample=True, default=float("nan"))


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
    for generation_index in range(min_gen):
        values = [s[generation_index] for s in run_series]
        mean_value, std_value = meanStd(values)
        mean_list.append(mean_value)
        std_list.append(std_value)
    return (mean_list, std_list)


def collectAvgRuns(root_dir: str, band: str, map_name: str) -> List[List[float]]:
    map_dir = Path(root_dir) / band / map_name
    if not map_dir.exists():
        return []

    run_series: List[List[float]] = []
    for json_path in sorted(map_dir.glob("*.json")):
        run = loadJson(json_path)
        series = getAvgSeries(run)
        if series:
            run_series.append(series)
    return run_series


def _resultDir(
    *,
    results_root: str = "__RESULTS__",
    algorithm: str = "ga",
    map_name: str = "gangjin.down",
) -> Path:
    return resultDir(results_root=results_root, algorithm=algorithm, map_name=map_name)


def listBands(
    *,
    results_root: str = "__RESULTS__",
    algorithm: str = "ga",
    map_name: str = "gangjin.down",
) -> List[str]:
    """실험 결과 경로에서 초기 센서수 band 디렉터리 목록을 숫자순으로 반환."""
    return _list_bands(results_root=results_root, algorithm=algorithm, map_name=map_name)


def loadRuns(
    *,
    results_root: str = "__RESULTS__",
    algorithm: str = "ga",
    map_name: str = "gangjin.down",
    seed_band: Optional[str] = None,
) -> List[Tuple[Path, Dict[str, Any]]]:
    """`__RESULTS__/<algorithm>/<map>/<seed_band>/*.json` 결과를 로드."""
    return loadAlgoRuns(
        results_root=results_root,
        algorithm=algorithm,
        map_name=map_name,
        seed_band=seed_band,
    )


def _cornerCnt(run: Dict[str, Any]) -> int:
    final = run.get("final", {}) or {}
    corners = final.get("corner_points", []) or []
    if isinstance(corners, list):
        return len(corners)
    extra = run.get("extra", {}).get("final", {}) or {}
    return int(extra.get("n_corner", 0) or 0)


def _sensorVal(
    gen: Dict[str, Any],
    *,
    include_corners: bool,
    corner_count: int,
    metric: str,
) -> float:
    metric = str(metric).lower()
    if metric in {"best", "best_inner", "n_inner"}:
        val = gen.get("n_inner", None)
        if not isinstance(val, (int, float)):
            val = len(gen.get("best_solution", []) or [])
    elif metric in {"avg", "avg_inner", "sensors_avg"}:
        val = gen.get("sensors_avg", None)
        if not isinstance(val, (int, float)):
            return float("nan")
        # Logger records total avg for GA/PSO/DRL in current pipeline, but
        # older runs may store inner-only values. Keep the raw logged value.
        return float(val)
    elif metric in {"min", "sensors_min"}:
        val = gen.get("sensors_min", None)
    elif metric in {"max", "sensors_max"}:
        val = gen.get("sensors_max", None)
    else:
        raise ValueError("metric must be one of: best, avg, min, max")

    if not isinstance(val, (int, float)):
        return float("nan")
    val = float(val)
    return val + float(corner_count) if include_corners else val


def getSensorSeries(
    run: Dict[str, Any],
    *,
    include_corners: bool = True,
    metric: str = "best",
) -> List[float]:
    """세대별 센서 수 series 추출. 기본은 best_solution 기준."""
    gens = run.get("generations", [])
    if not isinstance(gens, list):
        return []
    corners = _cornerCnt(run)
    series = [
        _sensorVal(
            g,
            include_corners=include_corners,
            corner_count=corners,
            metric=metric,
        )
        for g in gens
        if isinstance(g, dict)
    ]
    return [v for v in series if not math.isnan(v)]


def _buildTrend(
    *,
    results_root: str,
    algorithm: str,
    map_name: str,
    seed_bands: Optional[Iterable[str]],
    include_corners: bool,
    metric: str,
) -> Dict[str, Dict[str, Any]]:
    bands = list(seed_bands) if seed_bands is not None else listBands(
        results_root=results_root,
        algorithm=algorithm,
        map_name=map_name,
    )
    info: Dict[str, Dict[str, Any]] = {}
    for band in sorted(bands, key=bandKey):
        try:
            runs = loadRuns(
                results_root=results_root,
                algorithm=algorithm,
                map_name=map_name,
                seed_band=band,
            )
        except FileNotFoundError:
            continue
        run_series = [
            getSensorSeries(run, include_corners=include_corners, metric=metric)
            for _, run in runs
        ]
        mean_list, std_list = buildTrend(run_series)
        if not mean_list:
            continue
        info[band] = {
            "runs": len(run_series),
            "gens": len(mean_list),
            "mean": mean_list,
            "std": std_list,
            "final_mean": mean_list[-1],
            "final_std": std_list[-1],
        }
    return info


def plotConverge(
    *,
    results_root: str = "__RESULTS__",
    algorithm: str = "ga",
    map_name: str = "gangjin.down",
    seed_bands: Optional[Iterable[str]] = None,
    include_corners: bool = True,
    metric: str = "best",
    show_std: bool = True,
    threshold: float = 0.5,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (8.0, 4.8),
    dpi: int = 300,
    save_path: Optional[str] = None,
    show: bool = False,
) -> Dict[str, Any]:
    """
    초기 seed 센서수 band별 세대 수렴 그래프를 생성하고 선택적으로 저장.

    Returns:
        {
            "trend": {
                "info": band별 mean/std,
                "convergence": analyzeChange 결과,
            },
            "convergence": analyzeChange 결과,  # backward compatible
        }
    """
    info = _buildTrend(
        results_root=results_root,
        algorithm=algorithm,
        map_name=map_name,
        seed_bands=seed_bands,
        include_corners=include_corners,
        metric=metric,
    )
    if not info:
        raise FileNotFoundError(
            f"No convergence data for {results_root}/{algorithm}/{map_name}"
        )

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    linestyle_cycle = ["-", "--", "-.", ":"]
    marker_cycle = ["o", "s", "^", "D", "x", "+"]
    for idx, (band, data) in enumerate(sorted(info.items(), key=lambda item: bandKey(item[0]))):
        y = data["mean"]
        x = list(range(1, len(y) + 1))
        style = linestyle_cycle[idx % len(linestyle_cycle)]
        marker = marker_cycle[idx % len(marker_cycle)]
        ax.plot(
            x,
            y,
            label=f"{band} (n={data['runs']})",
            linestyle=style,
            marker=marker,
            markersize=1.8,
            markevery=max(len(x) // 12, 1),
            linewidth=1.6,
        )
        if show_std:
            std = data["std"]
            low = [m - s for m, s in zip(y, std)]
            high = [m + s for m, s in zip(y, std)]
            ax.fill_between(x, low, high, alpha=0.12)

    ax.set_xlim(0, max(x) + 1)
    ax.set_ylim(bottom=0)
    ax.set_title(title or f"{algorithm.upper()} sensor-count convergence: {map_name}")
    ax.set_xlabel("Generation")
    ylabel = "Sensors (inner + corner)" if include_corners else "Inner sensors"
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    convergence = analyzeChange(info, threshold=threshold, verbose=False)
    return {
        "trend": {"info": info, "convergence": convergence},
        "convergence": convergence,
    }


def _coverMap(
    *,
    map_data: Any,
    points: Sequence[Tuple[float, float]],
    coverage: int = 45,
    target_values: Sequence[int] = (2, 3),
) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(map_data)
    if arr.ndim != 2:
        raise ValueError(f"map_data must be 2D. Got shape={arr.shape}")
    target_mask = np.isin(arr, list(target_values))
    counts = np.zeros(arr.shape, dtype=np.uint16)
    height, width = arr.shape
    offsets = circle_offsets(max(0, int(coverage) // 5))
    for x_f, y_f in points:
        x, y = int(round(float(x_f))), int(round(float(y_f)))
        yy = y + offsets[:, 0]
        xx = x + offsets[:, 1]
        valid = (yy >= 0) & (yy < height) & (xx >= 0) & (xx < width)
        yy = yy[valid]
        xx = xx[valid]
        inside_target = target_mask[yy, xx]
        counts[yy[inside_target], xx[inside_target]] += 1
    return counts, target_mask


def coverOverlap(
    run: Dict[str, Any],
    *,
    map_data: Optional[Any] = None,
    target_values: Sequence[int] = (2, 3),
    coverage: Optional[int] = None,
) -> Dict[str, float]:
    """최종 센서 배치의 전체 커버리지와 중첩 커버리지를 셀 단위로 계산."""
    final = run.get("final", {}) or {}
    points = asPoints(final.get("best_solution", [])) + asPoints(final.get("corner_points", []))
    if map_data is None:
        map_name = str(run.get("map_name") or run.get("meta", {}).get("map_name"))
        map_data = MapLoader().load(map_name)
    if coverage is None:
        coverage = int(
            (run.get("meta", {}).get("optimizer_init", {}) or {}).get("coverage", 45)
        )

    counts, target_mask = _coverMap(
        map_data=map_data,
        points=points,
        coverage=int(coverage),
        target_values=target_values,
    )
    target_area = int(np.count_nonzero(target_mask))
    covered_cells = int(np.count_nonzero((counts > 0) & target_mask))
    overlap_cells = int(np.count_nonzero((counts > 1) & target_mask))
    target_counts = counts[target_mask].astype(np.int64, copy=False)
    total_hits = int(target_counts.sum())
    duplicate_hits = int(np.maximum(target_counts - 1, 0).sum())

    denom_target = max(1, target_area)
    denom_covered = max(1, covered_cells)
    denom_hits = max(1, total_hits)
    return {
        "n_sensors": float(len(points)),
        "target_area_cells": float(target_area),
        "covered_cells": float(covered_cells),
        "overlap_cells": float(overlap_cells),
        "total_coverage_hits": float(total_hits),
        "duplicate_coverage_hits": float(duplicate_hits),
        "coverage_percent": 100.0 * covered_cells / denom_target,
        "overlap_percent_of_target": 100.0 * overlap_cells / denom_target,
        "overlap_percent_of_covered": 100.0 * overlap_cells / denom_covered,
        "redundant_hit_percent": 100.0 * duplicate_hits / denom_hits,
        "logged_final_coverage": float(final.get("coverage", float("nan"))),
    }


def coverSummary(
    *,
    results_root: str = "__RESULTS__",
    algorithm: str = "ga",
    map_name: str = "gangjin.down",
    seed_bands: Optional[Iterable[str]] = None,
    target_values: Sequence[int] = (2, 3),
) -> Dict[str, Dict[str, float]]:
    """seed band별 최종 커버리지/중첩 커버리지 평균과 표준편차 계산."""
    map_data = MapLoader().load(map_name)
    bands = list(seed_bands) if seed_bands is not None else listBands(
        results_root=results_root,
        algorithm=algorithm,
        map_name=map_name,
    )
    summary: Dict[str, Dict[str, float]] = {}
    keys = [
        "n_sensors",
        "coverage_percent",
        "overlap_percent_of_target",
        "overlap_percent_of_covered",
        "redundant_hit_percent",
        "logged_final_coverage",
    ]
    for band in sorted(bands, key=bandKey):
        try:
            runs = loadRuns(
                results_root=results_root,
                algorithm=algorithm,
                map_name=map_name,
                seed_band=band,
            )
        except FileNotFoundError:
            continue
        rows = [
            coverOverlap(run, map_data=map_data, target_values=target_values)
            for _, run in runs
        ]
        if not rows:
            continue
        stats: Dict[str, float] = {"runs": float(len(rows))}
        for key in keys:
            vals = [row[key] for row in rows if not math.isnan(float(row[key]))]
            stats[f"{key}_mean"] = meanVal(vals, default=float("nan"))
            stats[f"{key}_std"] = stdVal(vals, sample=True, default=0.0)
        summary[band] = stats
    return summary


def plotOverlap(
    *,
    results_root: str = "__RESULTS__",
    algorithm: str = "ga",
    map_name: str = "gangjin.down",
    seed_bands: Optional[Iterable[str]] = None,
    target_values: Sequence[int] = (2, 3),
    figsize: Tuple[float, float] = (8.0, 4.8),
    dpi: int = 300,
    save_path: Optional[str] = None,
    show: bool = False,
) -> Dict[str, Dict[str, float]]:
    """최종 전체 커버리지와 중첩 커버리지 요약 막대 그래프 생성."""
    summary = coverSummary(
        results_root=results_root,
        algorithm=algorithm,
        map_name=map_name,
        seed_bands=seed_bands,
        target_values=target_values,
    )
    if not summary:
        raise FileNotFoundError(f"No coverage data for {results_root}/{algorithm}/{map_name}")

    bands = sorted(summary, key=bandKey)
    x = np.arange(len(bands))
    width = 0.2
    cov = [summary[b]["coverage_percent_mean"] for b in bands]
    cov_std = [summary[b]["coverage_percent_std"] for b in bands]
    overlap = [summary[b]["overlap_percent_of_covered_mean"] for b in bands]
    overlap_std = [summary[b]["overlap_percent_of_covered_std"] for b in bands]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.bar(x - width / 2, cov, width, yerr=cov_std, capsize=3, label="Total coverage (%)")
    ax.bar(
        x + width / 2,
        overlap,
        width,
        yerr=overlap_std,
        capsize=3,
        label="Overlapped area within covered (%)",
    )
    ax.set_title(f"{algorithm.upper()} final coverage and overlap: {map_name}")
    ax.set_xlabel("Initial sensor seed band")
    ax.set_ylabel("Percent (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(bands)
    ax.set_ylim(0, max(100.0, max(cov + overlap) * 1.08))
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return summary


def saveReport(
    *,
    results_root: str = "__RESULTS__",
    algorithm: str = "ga",
    map_name: str = "gangjin.down",
    output_dir: PathInput = "__RESULTS__/_analysis",
    summary_dir: Optional[PathInput] = None,
    seed_bands: Optional[Iterable[str]] = None,
    include_corners: bool = True,
    metric: str = "best",
    threshold: float = 0.5,
    target_values: Sequence[int] = (2, 3),
    dpi: int = 300,
    show: bool = False,
) -> Dict[str, Any]:
    """
    논문용 분석 산출물을 한 번에 저장.

    저장 파일:
      - output_dir/sensor_convergence_<algorithm>_<map>.png
      - output_dir/coverage_overlap_<algorithm>_<map>.png
      - summary_dir/coverage_overlap_<algorithm>_<map>.json
    """
    out_dir = Path(output_dir)
    json_dir = Path(summary_dir) if summary_dir is not None else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    safe_map = str(map_name).replace("/", "_").replace(".", "_")
    stem = f"{algorithm}_{safe_map}"
    convergence_path = out_dir / f"sensor_convergence_{stem}.png"
    coverage_path = out_dir / f"coverage_overlap_{stem}.png"
    summary_path = json_dir / f"coverage_overlap_{stem}.json"

    convergence = plotConverge(
        results_root=results_root,
        algorithm=algorithm,
        map_name=map_name,
        seed_bands=seed_bands,
        include_corners=include_corners,
        metric=metric,
        threshold=threshold,
        save_path=str(convergence_path),
        dpi=dpi,
        show=show,
    )
    coverage_summary = plotOverlap(
        results_root=results_root,
        algorithm=algorithm,
        map_name=map_name,
        seed_bands=seed_bands,
        target_values=target_values,
        save_path=str(coverage_path),
        dpi=dpi,
        show=show,
    )
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(coverage_summary, f, ensure_ascii=False, indent=2)

    return {
        "convergence_plot": str(convergence_path),
        "coverage_overlap_plot": str(coverage_path),
        "coverage_overlap_summary": str(summary_path),
        "convergence": convergence["convergence"],
        "coverage_summary": coverage_summary,
    }


def seedBandKey(band: str) -> int:
    """
    band 문자열이 '40-60' 형태일 때,
    앞 숫자(40)를 기준으로 정렬하기 위한 key
    """
    return bandKey(band)


def plotAvgTrend(
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
    band_list = sorted(band_list, key=seedBandKey, reverse=True)

    # ✅ 흑백 논문용 스타일 세트
    linestyle_cycle = ["-", "--", "-.", ":"]
    marker_cycle = ["o", "s", "^", "D", "x", "+"]

    result: Dict[str, Dict[str, Any]] = {}

    fig = plt.figure(figsize=figsize, dpi=dpi)

    for idx, band in enumerate(band_list):
        run_series = collectAvgRuns(root_dir, band, map_name)
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
            markersize=1.8,
            markevery=min(len(x) // 10, 1),  # 마커 과밀 방지
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

    plt.xlim(0, max(x) + 1)
    plt.ylim(bottom=0)
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


def analyzeChange(
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
        info: plotAvgTrend() 반환값 (band -> {mean, std, runs, gens})
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
