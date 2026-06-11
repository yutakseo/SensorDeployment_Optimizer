from __future__ import annotations
import os

from Engine import run_pipeline


RESULTS_ROOT = "__RESULTS__"
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
    (40, 60),
    (60, 80),
    (80, 100),
    (100, 120),
    (120, 140),
]
RANGE_ALGORITHMS = {"ga", "pso"}
DEFAULT_SENSOR_RANGE = (0, 140)
ITERATIONS = 100

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
GA_CPU_WORKERS = min(16, max(1, (os.cpu_count() or 2) - 1))
ALGORITHMS = ("ga", "greedy", "drl", "pso")
OPTIMIZER_PARAMS = {
    "ga": {
        "initial_size": 100,
        "selection_size": 50,
        "child_chromo_size": 100,
        "min_sensors": 0,
        "generations": 1000,
        "mutation_kwargs": {"min_separation": COMMON_OPTIMIZER_PARAMS["coverage"] / 5},
    },
    "pso": {
        "swarm_size": 100,
        "min_sensors": 0,
        "generations": 1000,
    },
    "greedy": {
        "min_sensors": 0,
        "candidate_stride": 5,
        "min_separation": COMMON_OPTIMIZER_PARAMS["coverage"] / 5,
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
    },
    "drl": {
        "min_sensors": 0,
        "generations": 1000,
        "candidate_stride": 5,
        "max_candidates": 512,
        "min_separation": COMMON_OPTIMIZER_PARAMS["coverage"] / 5,
        "fitness_kwargs": {"target_coverage": 100.0},
    },
}

OPTIMIZER_RUN_PARAMS = {
    "ga": {
        "selection_method": "elite",
        "tournament_size": 3,
        "mutation_rate": 0.7,
        "early_stop": False,
        "early_stop_coverage": 90.0,
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
        "count_change_rate": 0.7,
        "early_stop": False,
        "early_stop_coverage": 90.0,
        "early_stop_patience": 10,
        "return_best_only": True,
        "verbose": True,
        "profile": True,
        "profile_every": 1,
    },
    "greedy": {
        "target_coverage": 100.0,
        "return_best_only": True,
        "verbose": True,
        "profile": True,
        "profile_every": 1,
    },
    "combinatorial": {
        "target_coverage": 100.0,
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
        "heuristic_warmup_episodes": 1,
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
    "dpi": 300,
    "title": "Final Sensor Locations after Optimization",
    "filename": None,  # None -> <run_name>_final_sensors.png
    "save_dir": None,  # None -> same directory as result JSON
    "cmap": "gray",
}


if __name__ == "__main__":
    for algorithm in ALGORITHMS:
        algorithm_key = str(algorithm).lower()
        sensor_ranges = (
            SENSOR_RANGES
            if algorithm_key in RANGE_ALGORITHMS
            else [DEFAULT_SENSOR_RANGE]
        )
        for sensor_range in sensor_ranges:
            range_label = f"{sensor_range[0]}-{sensor_range[1]}"
            for map_name in MAP_NAMES:
                if algorithm_key in RANGE_ALGORITHMS:
                    results_dir = f"{RESULTS_ROOT}/{algorithm}/{map_name}/{range_label}"
                else:
                    results_dir = f"{RESULTS_ROOT}/{algorithm}/{map_name}"
                for _ in range(ITERATIONS):
                    final_points, out_path = run_pipeline(
                        map_name=map_name,
                        algorithm=algorithm,
                        sensor_range=sensor_range,
                        results_dir=results_dir,
                        map_layer_params=MAP_LAYER_PARAMS,
                        harris_params=HARRIS_PARAMS,
                        common_optimizer_params=COMMON_OPTIMIZER_PARAMS,
                        optimizer_params=OPTIMIZER_PARAMS,
                        optimizer_run_params=OPTIMIZER_RUN_PARAMS,
                        logger_params=LOGGER_PARAMS,
                        final_plot_params=FINAL_PLOT_PARAMS,
                    )
                    range_text = (
                        f"range={sensor_range}"
                        if algorithm_key in RANGE_ALGORITHMS
                        else "range=excluded"
                    )
                    print(
                        f"[Done] map={map_name} algorithm={algorithm} "
                        f"{range_text} sensors={len(final_points)} result={out_path}"
                    )
