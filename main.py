from __future__ import annotations
import os
from copy import deepcopy
from typing import Any
from Engine import run_pipeline


RESULTS_ROOT = "__RESULTS__/convergence_exp"
MAP_NAMES = [
    "gangjin.full",
    "gangjin.up",
    "gangjin.down",
    "sejong.full",
    "seocho.full",
    "seocho.up",
    "seocho.down",
]
SENSOR_RANGES = [
    (100, 100),
]
ALGORITHMS = (
    "ga",
    #"greedy", 
    "drl", 
    "pso"
    )

ITERATIONS = 100
GENERATIONS = 100
EPISODES = 1000
EARLY_STOP = False
TARGET_COVERAGE = 99.0
# 1) Map loader / layer parameters
MAP_LAYER_PARAMS = {
    "installable_values": [2],
    "road_values": [3],
    "jobsite_values": [2, 3],
}

# 2) Outermost sensor placement parameters
HARRIS_PARAMS = {
    "blockSize": 3,
    "ksize": 3,
    "k": 0.05,
    "dilate_size": 5,
    "min_dist": 9,
}

# 3) Inner sensor optimizer parameters
COMMON_OPTIMIZER_PARAMS = {
    "coverage": 45,
}
GA_CPU_WORKERS = min(16, max(1, (os.cpu_count() or 2) - 4))
OPTIMIZER_PARAMS: dict[str, dict[str, Any]] = {
    "ga": {
        "initial_size": 100,
        "selection_size": 50,
        "child_chromo_size": 100,
        "min_sensors": 0,
        "generations": GENERATIONS,
        "fitness_kwargs": {"target_coverage": TARGET_COVERAGE},
        "mutation_kwargs": {"min_separation": COMMON_OPTIMIZER_PARAMS["coverage"] / 5},
    },
    "pso": {
        "swarm_size": 100,
        "min_sensors": 0,
        "generations": GENERATIONS,
        "fitness_kwargs": {"target_coverage": TARGET_COVERAGE},
    },
    "greedy": {
        "min_sensors": 0,
        "candidate_stride": 5,
        "min_separation": COMMON_OPTIMIZER_PARAMS["coverage"] / 5,
        "fitness_kwargs": {"target_coverage": TARGET_COVERAGE},
    },
    "combinatorial": {
        "min_sensors": 0,
        "max_sensors": 24,
        "candidate_stride": 10,
        "max_candidates": 24,
        "max_combinations": 5_000_000,
        "min_separation": COMMON_OPTIMIZER_PARAMS["coverage"] / 5,
        "parallel_workers": GA_CPU_WORKERS,
        "chunk_size": 4096,
        "fitness_kwargs": {"target_coverage": TARGET_COVERAGE},
    },
    "drl": {
        "min_sensors": 0,
        "generations": EPISODES,
        "candidate_stride": 5,
        "max_candidates": 512,
        "min_separation": COMMON_OPTIMIZER_PARAMS["coverage"] / 5,
        "fitness_kwargs": {"target_coverage": TARGET_COVERAGE},
    },
}

OPTIMIZER_RUN_PARAMS: dict[str, dict[str, Any]] = {
    "ga": {
        "selection_method": "elite",
        "tournament_size": 3,
        "mutation_rate": 0.7,
        "early_stop": EARLY_STOP,
        "early_stop_coverage": TARGET_COVERAGE,
        "early_stop_patience": 10,
        "return_best_only": True,
        "verbose": True,
        "profile": True,
        "profile_every": 1,
        "profile_fitness_breakdown": True,
        "parallel_workers": GA_CPU_WORKERS,
    },
    "pso": {
        "inertia": 0.72,
        "cognitive": 2.0,
        "social": 2.0,
        "count_add_rate": 0.4,
        "count_del_rate": 0.3,
        "count_change_rate": 0.7,
        "early_stop": EARLY_STOP,
        "early_stop_coverage": TARGET_COVERAGE,
        "early_stop_patience": 10,
        "return_best_only": True,
        "verbose": True,
        "profile": True,
        "profile_every": 1,
    },
    "greedy": {
        "target_coverage": TARGET_COVERAGE,
        "return_best_only": True,
        "verbose": True,
        "profile": True,
        "profile_every": 1,
    },
    "combinatorial": {
        "target_coverage": TARGET_COVERAGE,
        "return_best_only": True,
        "verbose": True,
        "profile": True,
        "profile_every": 100_000,
        "parallel_workers": GA_CPU_WORKERS,
        "chunk_size": 4096,
    },
    "drl": {
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay": 0.985,
        "heuristic_warmup_episodes": 10,
        "return_best_only": True,
        "verbose": True,
        "profile": True,
        "profile_every": 1,
    },
}

# 4) Result logger parameters
LOGGER_PARAMS = {
    "point_format": "tuple_str",
    "sort_points": False,
    "group_by_map": False,
}

# 5) Final result plot parameters
FINAL_PLOT_PARAMS = {
    "enabled": True,
    "show": False,
    "size": (10, 10),
    "dpi": 200,
    "title": "Final Sensor Locations after Optimization",
    "filename": None,  # None -> <run_name>_final_sensors.png
    "save_dir": None,  # None -> same directory as result JSON
    "cmap": "gray",
}


def algorithmKey(algorithm: str) -> str:
    return str(algorithm).lower()


def selectedAlgorithms() -> tuple[str, ...]:
    selected = tuple(algorithmKey(algorithm) for algorithm in ALGORITHMS)
    configured = set(OPTIMIZER_PARAMS) & set(OPTIMIZER_RUN_PARAMS)
    unknown = sorted(set(selected) - configured)
    if unknown:
        available = sorted(configured)
        raise ValueError(
            f"Algorithms are not configured: {unknown}. "
            f"Available algorithms: {available}"
        )
    return selected


def resultsDir(*, algorithm: str, map_name: str, sensor_range: tuple[int, int]) -> str:
    algorithm_key = algorithmKey(algorithm)
    range_label = f"{sensor_range[0]}-{sensor_range[1]}"
    return f"{RESULTS_ROOT}/{algorithm_key}/{map_name}/{range_label}"


def optimizerParams() -> dict[str, dict[str, Any]]:
    return deepcopy(OPTIMIZER_PARAMS)


def optimizerRunParams() -> dict[str, dict[str, Any]]:
    return deepcopy(OPTIMIZER_RUN_PARAMS)


if __name__ == "__main__":
    run_optimizer_params = optimizerParams()
    run_optimizer_params_by_algorithm = optimizerRunParams()
    for algorithm in selectedAlgorithms():
        for sensor_range in SENSOR_RANGES:
            for map_name in MAP_NAMES:
                output_dir = resultsDir(
                    algorithm=algorithm,
                    map_name=map_name,
                    sensor_range=sensor_range,
                )
                for _ in range(ITERATIONS):
                    final_points, out_path = run_pipeline(
                        map_name=map_name,
                        algorithm=algorithm,
                        sensor_range=sensor_range,
                        results_dir=output_dir,
                        map_layer_params=MAP_LAYER_PARAMS,
                        harris_params=HARRIS_PARAMS,
                        common_optimizer_params=COMMON_OPTIMIZER_PARAMS,
                        optimizer_params=run_optimizer_params,
                        optimizer_run_params=run_optimizer_params_by_algorithm,
                        logger_params=LOGGER_PARAMS,
                        final_plot_params=FINAL_PLOT_PARAMS,
                    )
                    print(
                        f"[Done] map={map_name} algorithm={algorithm} "
                        f"range={sensor_range} sensors={len(final_points)} result={out_path}"
                    )
