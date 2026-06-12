from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

from Analysis import plotCombinatorialFitness3d
from Engine import run_pipeline

RESULTS_ROOT = "__RESULTS__"
DEFAULT_MAPS = (
    "gangjin.full",
    "gangjin.up",
    "gangjin.down",
    "sejong.full",
    "seocho.full",
    "seocho.up",
    "seocho.down",
)
DEFAULT_COVERAGE = 45
DEFAULT_CANDIDATE_STRIDE = 10
DEFAULT_MAX_CANDIDATES = 100_000_000
DEFAULT_MAX_COMBINATIONS = 100_000_000_000
DEFAULT_CHUNK_SIZE = 4096
DEFAULT_FITNESS_TRACE_STRIDE = 1000
DEFAULT_SAMPLE_COMBINATIONS = 100_000_000
DEFAULT_SAMPLE_SEED = 42
DEFAULT_TARGET_COVERAGE = 90.0
DEFAULT_MIN_SEPARATION_RATIO = 5.0
DEFAULT_PROFILE_EVERY = 5_000
LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"


@dataclass(frozen=True, slots=True)
class BruteDemoConfig:
    map_names: Tuple[str, ...]
    coverage: int
    candidate_stride: int
    max_candidates: int
    max_combinations: int
    workers: int
    chunk_size: int
    fitness_trace_stride: int
    sample_combinations: int
    sample_seed: int
    target_coverage: float
    results_root: Path
    show: bool


def parseArgs() -> BruteDemoConfig:
    parser = argparse.ArgumentParser(
        description="Run combinatorial brute-force sensor placement with sampled fitness tracing.",
    )
    parser.add_argument(
        "--map",
        dest="map_names",
        action="append",
        default=None,
        help="Map name to run. Repeat this option to run a subset. Defaults to all maps.",
    )
    parser.add_argument("--coverage", type=int, default=DEFAULT_COVERAGE)
    parser.add_argument("--candidate-stride", type=int, default=DEFAULT_CANDIDATE_STRIDE)
    parser.add_argument("--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES)
    parser.add_argument("--max-combinations", type=int, default=DEFAULT_MAX_COMBINATIONS)
    parser.add_argument("--workers", type=int, default=defaultWorkers())
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument(
        "--fitness-trace-stride",
        type=int,
        default=DEFAULT_FITNESS_TRACE_STRIDE,
    )
    parser.add_argument(
        "--sample-combinations",
        type=int,
        default=DEFAULT_SAMPLE_COMBINATIONS,
    )
    parser.add_argument("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument("--target-coverage", type=float, default=DEFAULT_TARGET_COVERAGE)
    parser.add_argument("--results-root", default=RESULTS_ROOT)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    return BruteDemoConfig(
        map_names=parseMapNames(args.map_names),
        coverage=int(args.coverage),
        candidate_stride=max(1, int(args.candidate_stride)),
        max_candidates=max(1, int(args.max_candidates)),
        max_combinations=max(1, int(args.max_combinations)),
        workers=max(1, int(args.workers)),
        chunk_size=max(1, int(args.chunk_size)),
        fitness_trace_stride=max(1, int(args.fitness_trace_stride)),
        sample_combinations=max(1, int(args.sample_combinations)),
        sample_seed=int(args.sample_seed),
        target_coverage=float(args.target_coverage),
        results_root=Path(args.results_root),
        show=bool(args.show),
    )


def parseMapNames(values: list[str] | None) -> Tuple[str, ...]:
    if not values:
        return DEFAULT_MAPS
    names: list[str] = []
    for value in values:
        for item in str(value).split(","):
            name = item.strip()
            if name:
                names.append(name)
    if not names:
        raise ValueError("At least one map name must be provided.")
    return tuple(dict.fromkeys(names))


def defaultWorkers() -> int:
    return max(1, (os.cpu_count() or 2) - 1)


def buildRunDir(config: BruteDemoConfig, map_name: str) -> Path:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    map_dir = map_name.replace(".", "_")
    path = config.results_root / "combinatorial" / map_dir / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def buildTracePath(run_dir: Path) -> Path:
    return run_dir / "fitness_trace.jsonl"


def buildPipelineParams(
    config: BruteDemoConfig,
    *,
    map_name: str,
    run_dir: Path,
    trace_path: Path,
) -> Dict[str, Any]:
    min_separation = float(config.coverage) / DEFAULT_MIN_SEPARATION_RATIO
    return {
        "map_name": map_name,
        "algorithm": "combinatorial",
        "sensor_range": (0, config.max_candidates),
        "results_dir": str(run_dir),
        "map_layer_params": {
            "installable_values": [2],
            "road_values": [3],
            "jobsite_values": [2, 3],
        },
        "harris_params": {
            "blockSize": 3,
            "ksize": 3,
            "k": 0.05,
            "dilate_size": 5,
            "min_dist": 9,
        },
        "common_optimizer_params": {
            "coverage": config.coverage,
        },
        "optimizer_params": {
            "combinatorial": {
                "candidate_stride": config.candidate_stride,
                "max_candidates": config.max_candidates,
                "max_combinations": config.max_combinations,
                "min_separation": min_separation,
                "parallel_workers": config.workers,
                "chunk_size": config.chunk_size,
                "fitness_kwargs": {
                    "target_coverage": config.target_coverage,
                },
            },
        },
        "optimizer_run_params": {
            "combinatorial": {
                "target_coverage": config.target_coverage,
                "return_best_only": True,
                "verbose": True,
                "profile": True,
                "profile_every": DEFAULT_PROFILE_EVERY,
                "parallel_workers": config.workers,
                "chunk_size": config.chunk_size,
                "fitness_trace_stride": config.fitness_trace_stride,
                "sample_combinations": config.sample_combinations,
                "sample_seed": config.sample_seed,
                "fitness_log_path": str(trace_path),
            },
        },
        "logger_params": {
            "point_format": "tuple_str",
            "sort_points": False,
            "group_by_map": False,
        },
        "final_plot_params": {
            "enabled": True,
            "show": config.show,
            "size": (10, 10),
            "dpi": 220,
            "title": "Final Sensor Locations after Brute Force Optimization",
            "filename": "final_sensors",
            "save_dir": str(run_dir),
            "cmap": "gray",
        },
    }


def runMap(config: BruteDemoConfig, map_name: str) -> Dict[str, str]:
    run_dir = buildRunDir(config, map_name)
    trace_path = buildTracePath(run_dir)
    params = buildPipelineParams(
        config,
        map_name=map_name,
        run_dir=run_dir,
        trace_path=trace_path,
    )

    logging.info("Starting brute-force demo: map=%s run_dir=%s", map_name, run_dir)
    final_points, result_path = run_pipeline(**params)
    fitness_plot_path = plotCombinatorialFitness3d(
        trace_path,
        save_path=run_dir / "fitness_landscape_3d.png",
        show=config.show,
        max_points=50_000,
    )

    logging.info("Final sensors: %s", len(final_points))
    logging.info("Result JSON: %s", result_path)
    logging.info("Fitness trace: %s", trace_path)
    logging.info("3D fitness plot: %s", fitness_plot_path)
    logging.info("Final placement plot: %s", run_dir / "final_sensors.png")
    return {
        "map_name": map_name,
        "result_path": str(result_path),
        "trace_path": str(trace_path),
        "fitness_plot_path": str(fitness_plot_path),
        "final_plot_path": str(run_dir / "final_sensors.png"),
        "final_sensors": str(len(final_points)),
    }


def runAllMaps(config: BruteDemoConfig) -> None:
    results: list[Dict[str, str]] = []
    total = len(config.map_names)
    for index, map_name in enumerate(config.map_names, start=1):
        logging.info("Running map %s/%s: %s", index, total, map_name)
        try:
            results.append(runMap(config, map_name))
        except Exception as exc:
            logging.exception("Map run failed: %s", map_name)
            results.append(
                {
                    "map_name": map_name,
                    "error": str(exc),
                }
            )

    summary_path = saveSummary(config, results)
    logging.info("Completed %s map runs.", len(results))
    logging.info("Batch summary: %s", summary_path)
    for result in results:
        if "error" in result:
            logging.info(
                "Summary map=%s error=%s",
                result["map_name"],
                result["error"],
            )
            continue
        logging.info(
            "Summary map=%s sensors=%s result=%s",
            result["map_name"],
            result["final_sensors"],
            result["result_path"],
        )


def saveSummary(config: BruteDemoConfig, results: list[Dict[str, str]]) -> Path:
    summary_dir = config.results_root / "combinatorial"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    payload = {
        "map_names": list(config.map_names),
        "sample_combinations": config.sample_combinations,
        "sample_seed": config.sample_seed,
        "results": results,
    }
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    config = parseArgs()
    validateConfig(config)
    runAllMaps(config)


def validateConfig(config: BruteDemoConfig) -> None:
    if not config.map_names:
        raise ValueError("map_names must not be empty.")
    if config.max_candidates <= 0:
        raise ValueError("max_candidates must be positive.")
    if config.sample_combinations <= 0:
        raise ValueError("sample_combinations must be positive.")


if __name__ == "__main__":
    main()
