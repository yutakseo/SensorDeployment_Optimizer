from __future__ import annotations
from Engine import run_pipeline


RESULTS_ROOT = "__RESULTS__"
MAP_NAMES = [
    "gangjin.full",
   # "gangjin.up",
   # "gangjin.down",
   # "sejong.full",
   # "seocho.full",
   # "seocho.up",
   # "seocho.down",
]
SENSOR_RANGES = [
    (40, 60),
    #(60, 80),
    #(80, 100),
    #(100, 120),
    #(120, 140),
]
ITERATIONS = 1

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
ALGORITHM = "pso"
OPTIMIZER_PARAMS = {
    "ga": {
        "initial_size": 100,
        "selection_size": 50,
        "child_chromo_size": 100,
        "min_sensors": 0,
        "generations": 1000,
    },
    "pso": {
        "swarm_size": 100,
        "min_sensors": 0,
        "generations": 1000,
    },
    "greedy": {
        "min_sensors": 0,
        "candidate_stride": 5,
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
    },
    "pso": {
        "inertia": 0.72,
        "cognitive": 1.49,
        "social": 1.49,
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
    for sensor_range in SENSOR_RANGES:
        for map_name in MAP_NAMES:
            results_dir = f"{RESULTS_ROOT}/{ALGORITHM}/{map_name}/{sensor_range[0]}-{sensor_range[1]}"
            for _ in range(ITERATIONS):
                final_points, out_path = run_pipeline(
                    map_name=map_name,
                    algorithm=ALGORITHM,
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
                print(
                    f"[Done] map={map_name} algorithm={ALGORITHM} "
                    f"range={sensor_range} sensors={len(final_points)} result={out_path}"
                )
