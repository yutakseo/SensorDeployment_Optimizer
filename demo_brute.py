from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from Analysis import plotCombinatorialFitness3d
from Engine import run_pipeline

RESULTS_ROOT = "__RESULTS__"
DEFAULT_MAP = "sejong.full"
DEFAULT_COVERAGE = 45
DEFAULT_CANDIDATE_STRIDE = 10
DEFAULT_MAX_CANDIDATES = 1000
DEFAULT_MAX_COMBINATIONS = 100_000_000_000_000
DEFAULT_CHUNK_SIZE = 512
DEFAULT_TARGET_COVERAGE = 90.0
DEFAULT_MIN_SEPARATION_RATIO = 5.0
DEFAULT_PROFILE_EVERY = 5_000
LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"


@dataclass(frozen=True, slots=True)
class BruteDemoConfig:
    map_name: str
    coverage: int
    candidate_stride: int
    max_candidates: int
    max_combinations: int
    workers: int
    chunk_size: int
    target_coverage: float
    results_root: Path
    show: bool


def parseArgs() -> BruteDemoConfig:
    parser = argparse.ArgumentParser(
        description="Run combinatorial brute-force sensor placement with full fitness tracing.",
    )
    parser.add_argument("--map", dest="map_name", default=DEFAULT_MAP)
    parser.add_argument("--coverage", type=int, default=DEFAULT_COVERAGE)
    parser.add_argument("--candidate-stride", type=int, default=DEFAULT_CANDIDATE_STRIDE)
    parser.add_argument("--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES)
    parser.add_argument("--max-combinations", type=int, default=DEFAULT_MAX_COMBINATIONS)
    parser.add_argument("--workers", type=int, default=defaultWorkers())
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--target-coverage", type=float, default=DEFAULT_TARGET_COVERAGE)
    parser.add_argument("--results-root", default=RESULTS_ROOT)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    return BruteDemoConfig(
        map_name=str(args.map_name),
        coverage=int(args.coverage),
        candidate_stride=max(1, int(args.candidate_stride)),
        max_candidates=max(1, int(args.max_candidates)),
        max_combinations=max(1, int(args.max_combinations)),
        workers=max(1, int(args.workers)),
        chunk_size=max(1, int(args.chunk_size)),
        target_coverage=float(args.target_coverage),
        results_root=Path(args.results_root),
        show=bool(args.show),
    )


def defaultWorkers() -> int:
    return min(16, max(1, (os.cpu_count() or 2) - 1))


def buildRunDir(config: BruteDemoConfig) -> Path:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    map_dir = config.map_name.replace(".", "_")
    path = config.results_root / "combinatorial" / map_dir / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def buildTracePath(run_dir: Path) -> Path:
    return run_dir / "fitness_trace.jsonl"


def buildPipelineParams(
    config: BruteDemoConfig,
    *,
    run_dir: Path,
    trace_path: Path,
) -> Dict[str, Any]:
    min_separation = float(config.coverage) / DEFAULT_MIN_SEPARATION_RATIO
    return {
        "map_name": config.map_name,
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


def runDemo(config: BruteDemoConfig) -> None:
    run_dir = buildRunDir(config)
    trace_path = buildTracePath(run_dir)
    params = buildPipelineParams(config, run_dir=run_dir, trace_path=trace_path)

    logging.info("Starting brute-force demo: map=%s run_dir=%s", config.map_name, run_dir)
    final_points, result_path = run_pipeline(**params)
    fitness_plot_path = plotCombinatorialFitness3d(
        trace_path,
        save_path=run_dir / "fitness_landscape_3d.png",
        show=config.show,
    )

    logging.info("Final sensors: %s", len(final_points))
    logging.info("Result JSON: %s", result_path)
    logging.info("Fitness trace: %s", trace_path)
    logging.info("3D fitness plot: %s", fitness_plot_path)
    logging.info("Final placement plot: %s", run_dir / "final_sensors.png")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    config = parseArgs()
    validateConfig(config)
    runDemo(config)


def validateConfig(config: BruteDemoConfig) -> None:
    if config.max_candidates <= 0:
        raise ValueError("max_candidates must be positive.")


if __name__ == "__main__":
    main()
