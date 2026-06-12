from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from Analysis import plotCombinatorialFitness3d
from Engine import run_pipeline

MapNames = Tuple[str, ...]
ResultRow = Dict[str, str]

RESULTS_ROOT = "__RESULTS__"
DEFAULT_MAPS: MapNames = (
    "gangjin.full",
    "gangjin.up",
    "gangjin.down",
    "sejong.full",
    "seocho.full",
    "seocho.up",
    "seocho.down",
)
LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"


@dataclass(frozen=True, slots=True)
class SearchConfig:
    coverage: int = 45
    target_coverage: float = 90.0
    candidate_stride: int = 10
    max_candidates: int = 30
    max_combinations: int = 100_000_000
    min_separation_ratio: float = 5.0
    search_mode: str = "adaptive"
    adaptive_start_sensors: Optional[int] = 5
    adaptive_samples_per_count: int = 200_000
    adaptive_intensify_samples: int = 100_000
    adaptive_refine_samples: int = 800_000
    adaptive_refine_rounds: int = 2
    adaptive_patience: int = 2
    adaptive_min_delta: float = 1.0
    adaptive_regress_delta: float = 0.0
    adaptive_uniform_ratio: float = 0.70
    adaptive_weighted_ratio: float = 0.20
    adaptive_local_ratio: float = 0.10
    adaptive_refine_uniform_ratio: float = 0.10
    adaptive_refine_weighted_ratio: float = 0.30
    adaptive_refine_local_ratio: float = 0.60
    sample_combinations: Optional[int] = None
    sample_seed: int = 42


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    workers: int
    chunk_size: int = 4096
    fitness_trace_stride: int = 1000
    profile_every: int = 50_000
    show: bool = False


@dataclass(frozen=True, slots=True)
class CombinatorialDemoConfig:
    map_names: MapNames = DEFAULT_MAPS
    results_root: Path = Path(RESULTS_ROOT)
    search: SearchConfig = field(default_factory=SearchConfig)
    runtime: RuntimeConfig = field(
        default_factory=lambda: RuntimeConfig(workers=max(1, (os.cpu_count() or 2) - 1))
    )


def defaultConfig() -> CombinatorialDemoConfig:
    return CombinatorialDemoConfig()


def parseArgs() -> CombinatorialDemoConfig:
    defaults = defaultConfig()
    parser = argparse.ArgumentParser(
        description="Run adaptive combinatorial sensor placement.",
    )
    addMapArgs(parser, defaults)
    addSearchArgs(parser, defaults.search)
    addRuntimeArgs(parser, defaults.runtime)
    args = parser.parse_args()
    return CombinatorialDemoConfig(
        map_names=parseMapNames(args.map_names),
        results_root=Path(args.results_root),
        search=SearchConfig(
            coverage=max(1, int(args.coverage)),
            target_coverage=float(args.target_coverage),
            candidate_stride=max(1, int(args.candidate_stride)),
            max_candidates=max(1, int(args.max_candidates)),
            max_combinations=max(1, int(args.max_combinations)),
            min_separation_ratio=max(1.0e-9, float(args.min_separation_ratio)),
            search_mode=str(args.search_mode),
            adaptive_start_sensors=args.adaptive_start_sensors,
            adaptive_samples_per_count=max(1, int(args.adaptive_samples_per_count)),
            adaptive_intensify_samples=max(0, int(args.adaptive_intensify_samples)),
            adaptive_refine_samples=max(0, int(args.adaptive_refine_samples)),
            adaptive_refine_rounds=max(0, int(args.adaptive_refine_rounds)),
            adaptive_patience=max(1, int(args.adaptive_patience)),
            adaptive_min_delta=max(0.0, float(args.adaptive_min_delta)),
            adaptive_regress_delta=max(0.0, float(args.adaptive_regress_delta)),
            adaptive_uniform_ratio=max(0.0, float(args.adaptive_uniform_ratio)),
            adaptive_weighted_ratio=max(0.0, float(args.adaptive_weighted_ratio)),
            adaptive_local_ratio=max(0.0, float(args.adaptive_local_ratio)),
            adaptive_refine_uniform_ratio=max(0.0, float(args.adaptive_refine_uniform_ratio)),
            adaptive_refine_weighted_ratio=max(0.0, float(args.adaptive_refine_weighted_ratio)),
            adaptive_refine_local_ratio=max(0.0, float(args.adaptive_refine_local_ratio)),
            sample_combinations=positiveOptional(args.sample_combinations),
            sample_seed=int(args.sample_seed),
        ),
        runtime=RuntimeConfig(
            workers=max(1, int(args.workers)),
            chunk_size=max(1, int(args.chunk_size)),
            fitness_trace_stride=max(1, int(args.fitness_trace_stride)),
            profile_every=max(1, int(args.profile_every)),
            show=bool(args.show),
        ),
    )


def addMapArgs(parser: argparse.ArgumentParser, defaults: CombinatorialDemoConfig) -> None:
    parser.add_argument(
        "--map",
        dest="map_names",
        action="append",
        default=None,
        help="Map name to run. Repeat or comma-separate. Defaults to all maps.",
    )
    parser.add_argument("--results-root", default=str(defaults.results_root))
    parser.add_argument("--show", action="store_true")


def addSearchArgs(parser: argparse.ArgumentParser, defaults: SearchConfig) -> None:
    group = parser.add_argument_group("search")
    group.add_argument("--coverage", type=int, default=defaults.coverage)
    group.add_argument("--target-coverage", type=float, default=defaults.target_coverage)
    group.add_argument("--candidate-stride", type=int, default=defaults.candidate_stride)
    group.add_argument("--max-candidates", type=int, default=defaults.max_candidates)
    group.add_argument("--max-combinations", type=int, default=defaults.max_combinations)
    group.add_argument("--min-separation-ratio", type=float, default=defaults.min_separation_ratio)
    group.add_argument(
        "--search-mode",
        choices=("adaptive", "sampled", "exhaustive"),
        default=defaults.search_mode,
    )
    group.add_argument("--adaptive-start-sensors", type=int, default=defaults.adaptive_start_sensors)
    group.add_argument("--adaptive-samples-per-count", type=int, default=defaults.adaptive_samples_per_count)
    group.add_argument("--adaptive-intensify-samples", type=int, default=defaults.adaptive_intensify_samples)
    group.add_argument("--adaptive-refine-samples", type=int, default=defaults.adaptive_refine_samples)
    group.add_argument("--adaptive-refine-rounds", type=int, default=defaults.adaptive_refine_rounds)
    group.add_argument("--adaptive-patience", type=int, default=defaults.adaptive_patience)
    group.add_argument("--adaptive-min-delta", type=float, default=defaults.adaptive_min_delta)
    group.add_argument("--adaptive-regress-delta", type=float, default=defaults.adaptive_regress_delta)
    group.add_argument("--adaptive-uniform-ratio", type=float, default=defaults.adaptive_uniform_ratio)
    group.add_argument("--adaptive-weighted-ratio", type=float, default=defaults.adaptive_weighted_ratio)
    group.add_argument("--adaptive-local-ratio", type=float, default=defaults.adaptive_local_ratio)
    group.add_argument(
        "--adaptive-refine-uniform-ratio",
        type=float,
        default=defaults.adaptive_refine_uniform_ratio,
    )
    group.add_argument(
        "--adaptive-refine-weighted-ratio",
        type=float,
        default=defaults.adaptive_refine_weighted_ratio,
    )
    group.add_argument(
        "--adaptive-refine-local-ratio",
        type=float,
        default=defaults.adaptive_refine_local_ratio,
    )
    group.add_argument("--sample-combinations", type=int, default=defaults.sample_combinations)
    group.add_argument("--sample-seed", type=int, default=defaults.sample_seed)


def addRuntimeArgs(parser: argparse.ArgumentParser, defaults: RuntimeConfig) -> None:
    group = parser.add_argument_group("runtime")
    group.add_argument("--workers", type=int, default=defaults.workers)
    group.add_argument("--chunk-size", type=int, default=defaults.chunk_size)
    group.add_argument("--fitness-trace-stride", type=int, default=defaults.fitness_trace_stride)
    group.add_argument("--profile-every", type=int, default=defaults.profile_every)


def positiveOptional(value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    return max(1, int(value))


def parseMapNames(values: Optional[list[str]]) -> MapNames:
    if not values:
        return DEFAULT_MAPS
    names: list[str] = []
    for value in values:
        for item in value.split(","):
            name = item.strip()
            if name:
                names.append(name)
    if not names:
        raise ValueError("At least one map name must be provided.")
    return tuple(dict.fromkeys(names))


def runAllMaps(config: CombinatorialDemoConfig) -> None:
    validateConfig(config)
    results: list[ResultRow] = []
    total = len(config.map_names)
    for index, map_name in enumerate(config.map_names, start=1):
        logging.info("Running map %s/%s: %s", index, total, map_name)
        try:
            results.append(runMap(config, map_name))
        except Exception as exc:
            logging.exception("Map run failed: %s", map_name)
            results.append({"map_name": map_name, "error": str(exc)})

    summary_path = saveSummary(config, results)
    logging.info("Completed %s map runs.", len(results))
    logging.info("Batch summary: %s", summary_path)
    logSummary(results)


def runMap(config: CombinatorialDemoConfig, map_name: str) -> ResultRow:
    run_dir = buildRunDir(config, map_name)
    trace_path = run_dir / "fitness_trace.jsonl"
    params = buildPipelineParams(
        config,
        map_name=map_name,
        run_dir=run_dir,
        trace_path=trace_path,
    )

    logging.info("Starting combinatorial demo: map=%s run_dir=%s", map_name, run_dir)
    final_points, result_path = run_pipeline(**params)
    fitness_plot_path = saveFitnessPlot(config, run_dir, trace_path)
    logging.info("Final sensors: %s", len(final_points))
    logging.info("Result JSON: %s", result_path)
    logging.info("Fitness trace: %s", trace_path)
    logging.info("3D fitness plot: %s", fitness_plot_path or "not generated")
    logging.info("Final placement plot: %s", run_dir / "final_sensors.png")
    return {
        "map_name": map_name,
        "result_path": str(result_path),
        "trace_path": str(trace_path),
        "fitness_plot_path": "" if fitness_plot_path is None else str(fitness_plot_path),
        "final_plot_path": str(run_dir / "final_sensors.png"),
        "final_sensors": str(len(final_points)),
    }


def buildRunDir(config: CombinatorialDemoConfig, map_name: str) -> Path:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    map_dir = map_name.replace(".", "_")
    path = config.results_root / "combinatorial" / map_dir / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def buildPipelineParams(
    config: CombinatorialDemoConfig,
    *,
    map_name: str,
    run_dir: Path,
    trace_path: Path,
) -> Dict[str, Any]:
    search = config.search
    runtime = config.runtime
    min_separation = float(search.coverage) / search.min_separation_ratio
    return {
        "map_name": map_name,
        "algorithm": "combinatorial",
        "sensor_range": (0, search.max_candidates),
        "results_dir": str(run_dir),
        "map_layer_params": buildMapLayerParams(),
        "harris_params": buildHarrisParams(),
        "common_optimizer_params": {"coverage": search.coverage},
        "optimizer_params": {
            "combinatorial": {
                "candidate_stride": search.candidate_stride,
                "max_candidates": search.max_candidates,
                "max_combinations": search.max_combinations,
                "min_separation": min_separation,
                "parallel_workers": runtime.workers,
                "chunk_size": runtime.chunk_size,
                "fitness_kwargs": {"target_coverage": search.target_coverage},
            },
        },
        "optimizer_run_params": {
            "combinatorial": buildRunParams(config, trace_path),
        },
        "logger_params": buildLoggerParams(),
        "final_plot_params": buildPlotParams(config, run_dir),
    }


def buildRunParams(config: CombinatorialDemoConfig, trace_path: Path) -> Dict[str, Any]:
    search = config.search
    runtime = config.runtime
    return {
        "target_coverage": search.target_coverage,
        "return_best_only": True,
        "verbose": True,
        "profile": True,
        "profile_every": runtime.profile_every,
        "parallel_workers": runtime.workers,
        "chunk_size": runtime.chunk_size,
        "fitness_trace_stride": runtime.fitness_trace_stride,
        "sample_combinations": search.sample_combinations,
        "sample_seed": search.sample_seed,
        "fitness_log_path": str(trace_path),
        "search_mode": search.search_mode,
        "adaptive_start_sensors": search.adaptive_start_sensors,
        "adaptive_samples_per_count": search.adaptive_samples_per_count,
        "adaptive_intensify_samples": search.adaptive_intensify_samples,
        "adaptive_refine_samples": search.adaptive_refine_samples,
        "adaptive_refine_rounds": search.adaptive_refine_rounds,
        "adaptive_patience": search.adaptive_patience,
        "adaptive_min_delta": search.adaptive_min_delta,
        "adaptive_regress_delta": search.adaptive_regress_delta,
        "adaptive_uniform_ratio": search.adaptive_uniform_ratio,
        "adaptive_weighted_ratio": search.adaptive_weighted_ratio,
        "adaptive_local_ratio": search.adaptive_local_ratio,
        "adaptive_refine_uniform_ratio": search.adaptive_refine_uniform_ratio,
        "adaptive_refine_weighted_ratio": search.adaptive_refine_weighted_ratio,
        "adaptive_refine_local_ratio": search.adaptive_refine_local_ratio,
    }


def buildMapLayerParams() -> Dict[str, list[int]]:
    return {
        "installable_values": [2],
        "road_values": [3],
        "jobsite_values": [2, 3],
    }


def buildHarrisParams() -> Dict[str, float | int]:
    return {
        "blockSize": 3,
        "ksize": 3,
        "k": 0.05,
        "dilate_size": 5,
        "min_dist": 9,
    }


def buildLoggerParams() -> Dict[str, bool | str]:
    return {
        "point_format": "tuple_str",
        "sort_points": False,
        "group_by_map": False,
    }


def buildPlotParams(config: CombinatorialDemoConfig, run_dir: Path) -> Dict[str, Any]:
    return {
        "enabled": True,
        "show": config.runtime.show,
        "size": (10, 10),
        "dpi": 220,
        "title": "Final Sensor Locations after Combinatorial Optimization",
        "filename": "final_sensors",
        "save_dir": str(run_dir),
        "cmap": "gray",
    }


def saveFitnessPlot(
    config: CombinatorialDemoConfig,
    run_dir: Path,
    trace_path: Path,
) -> Optional[Path]:
    try:
        return Path(
            plotCombinatorialFitness3d(
                trace_path,
                save_path=run_dir / "fitness_landscape_3d.png",
                show=config.runtime.show,
                max_points=50_000,
            )
        )
    except ValueError as exc:
        logging.warning("Fitness plot skipped: %s", exc)
        return None


def saveSummary(config: CombinatorialDemoConfig, results: list[ResultRow]) -> Path:
    summary_dir = config.results_root / "combinatorial"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    payload = {
        "config": configPayload(config),
        "results": results,
    }
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary_path


def configPayload(config: CombinatorialDemoConfig) -> Dict[str, Any]:
    payload = asdict(config)
    payload["results_root"] = str(config.results_root)
    return payload


def logSummary(results: list[ResultRow]) -> None:
    for result in results:
        if "error" in result:
            logging.info("Summary map=%s error=%s", result["map_name"], result["error"])
            continue
        logging.info(
            "Summary map=%s sensors=%s result=%s",
            result["map_name"],
            result["final_sensors"],
            result["result_path"],
        )


def validateConfig(config: CombinatorialDemoConfig) -> None:
    if not config.map_names:
        raise ValueError("map_names must not be empty.")
    if config.search.max_candidates <= 0:
        raise ValueError("max_candidates must be positive.")
    if config.search.search_mode not in {"adaptive", "sampled", "exhaustive"}:
        raise ValueError("search_mode must be one of: adaptive, sampled, exhaustive.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    runAllMaps(parseArgs())
