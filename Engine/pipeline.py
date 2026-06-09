from __future__ import annotations

import os
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Sequence, Tuple

from cpuinfo import get_cpu_info

from Analysis.visualization import VisualTool
from Engine.api import make_optimizer_configs
from Engine.logger import GAJsonLogger
from Engine.map_loader import MapLoader
from Engine.masks import layer_map
from Engine.optimizers import make_inner_optimizer
from OuterDeployment.HarrisCorner import HarrisCorner

Point = Tuple[int, int]
SensorRange = Tuple[int, int]


def load_map_layers(
    map_name: str,
    *,
    installable_values: Sequence[int],
    road_values: Sequence[int],
    jobsite_values: Sequence[int],
) -> Tuple[Any, Any, Any, Any]:
    """MapLoader -> installable / road / jobsite layers."""
    map_data = MapLoader().load(map_name)
    installable_layer = layer_map(map_data, keep_values=installable_values)
    road_layer = layer_map(map_data, keep_values=road_values)
    jobsite_layer = layer_map(map_data, keep_values=jobsite_values)
    return map_data, installable_layer, road_layer, jobsite_layer


def place_outer_sensors(
    *,
    jobsite_layer,
    installable_layer,
    harris_params: Dict[str, Any],
) -> List[Point]:
    """HarrisCorner -> outermost sensor positions."""
    return HarrisCorner(jobsite_layer).run(
        grid=jobsite_layer,
        installable_layer=installable_layer,
        **harris_params,
    )


def configure_inner_optimizer(
    *,
    algorithm: str,
    sensor_range: SensorRange,
    common_params: Dict[str, Any],
    optimizer_params: Dict[str, Dict[str, Any]],
    optimizer_run_params: Dict[str, Dict[str, Any]],
):
    """Create inner-optimizer config objects from one algorithm switch."""
    key = str(algorithm).lower()
    if key not in optimizer_params or key not in optimizer_run_params:
        raise ValueError(f"Unsupported algorithm={algorithm!r}. Use 'ga', 'pso', 'greedy', or 'drl'.")

    selected_optimizer_params = optimizer_params[key]
    generations = selected_optimizer_params.get(
        "generations",
        common_params.get("generations", 100),
    )

    return make_optimizer_configs(
        key,
        sensor_range=sensor_range,
        coverage=int(common_params["coverage"]),
        generations=int(generations),
        optimizer=selected_optimizer_params,
        run=optimizer_run_params[key],
    )


def _system_meta() -> Dict[str, Any]:
    try:
        info = get_cpu_info()
        cpu = info.get("brand_raw") if info else None
    except Exception:
        cpu = None
    return {"cpu": cpu}


def build_logger(
    *,
    map_name: str,
    results_dir: str,
    optimizer_init,
    optimizer_run,
    harris_params: Dict[str, Any],
    corner_count: int,
    logger_params: Dict[str, Any],
) -> GAJsonLogger:
    """GAJsonLogger -> result output."""
    return GAJsonLogger(
        map_name=map_name,
        base_dir=results_dir,
        meta={
            "map_name": map_name,
            "created_at": datetime.now().isoformat(),
            "system": _system_meta(),
            "optimizer_init": asdict(optimizer_init),
            "optimizer_run": asdict(optimizer_run),
            "harris": {**harris_params, "n_corner_candidates": int(corner_count)},
        },
        point_format=str(logger_params.get("point_format", "tuple_str")),
        sort_points=bool(logger_params.get("sort_points", False)),
        group_by_map=bool(logger_params.get("group_by_map", True)),
    )


def optimize_inner_sensors(
    *,
    algorithm: str,
    installable_layer,
    jobsite_layer,
    corner_positions: List[Point],
    optimizer_init,
    optimizer_run,
    logger: GAJsonLogger,
) -> Tuple[List[Point], Any]:
    """Inner optimizer -> inner sensor positions."""
    optimizer = make_inner_optimizer(
        algorithm=algorithm,
        installable_map=installable_layer,
        jobsite_map=jobsite_layer,
        corner_positions=corner_positions,
        init_cfg=optimizer_init,
        run_cfg=optimizer_run,
        logger=logger,
    )
    inner_points = optimizer.run()
    return list(inner_points), optimizer


def save_result(
    *,
    logger: GAJsonLogger,
    optimizer,
    inner_points: List[Point],
    corner_points: List[Point],
    final_plot_path: str | None = None,
) -> str:
    """Result logger -> JSON output path."""
    final_points = list(inner_points) + list(corner_points)
    extra_final = {
        "n_final_points": int(len(final_points)),
        "n_inner": int(len(inner_points)),
        "n_corner": int(len(corner_points)),
    }
    if final_plot_path is not None:
        extra_final["final_plot_path"] = final_plot_path

    return logger.finalize(
        best_solution=optimizer.best_solution or inner_points,
        corner_points=optimizer.corner_points,
        fitness=optimizer.best_fitness,
        coverage=optimizer.best_coverage,
        extra={"final": extra_final},
    )


def plot_final_sensor_placement(
    *,
    map_data,
    logger: GAJsonLogger,
    final_points: List[Point],
    plot_params: Dict[str, Any],
) -> str | None:
    """Final sensor placement plot -> PNG output path."""
    if not bool(plot_params.get("enabled", True)):
        return None

    save_dir = os.path.abspath(str(plot_params.get("save_dir") or logger.map_dir))
    filename = str(plot_params.get("filename") or f"{logger.run_name}_final_sensors")

    vis = VisualTool(
        save_dir=save_dir,
        show=bool(plot_params.get("show", False)),
        save=True,
        size=tuple(plot_params.get("size", (10, 10))),
        dpi=int(plot_params.get("dpi", 300)),
        stamp_filename=bool(plot_params.get("stamp_filename", False)),
        tight=bool(plot_params.get("tight", True)),
        pad_inches=float(plot_params.get("pad_inches", 0.0)),
        facecolor=plot_params.get("facecolor", None),
    )
    vis.showMapCircle(
        map_data=map_data,
        sensor_positions=final_points,
        title=str(plot_params.get("title", "Final Sensor Locations after Optimization")),
        radius=float(plot_params.get("radius", 45)),
        cmap=plot_params.get("cmap", "gray"),
        filename=filename,
        save_path=save_dir,
        vmin=plot_params.get("vmin", None),
        vmax=plot_params.get("vmax", None),
    )
    return os.path.join(save_dir, f"{filename}.png")


def run_pipeline(
    *,
    map_name: str,
    algorithm: str,
    sensor_range: SensorRange,
    results_dir: str,
    map_layer_params: Dict[str, Any],
    harris_params: Dict[str, Any],
    common_optimizer_params: Dict[str, Any],
    optimizer_params: Dict[str, Dict[str, Any]],
    optimizer_run_params: Dict[str, Dict[str, Any]],
    logger_params: Dict[str, Any],
    final_plot_params: Dict[str, Any],
) -> Tuple[List[Point], str]:
    # 1) MapLoader
    map_data, installable_layer, _, jobsite_layer = load_map_layers(
        map_name,
        **map_layer_params,
    )

    # 2) Harris outer sensor placement
    corner_points = place_outer_sensors(
        jobsite_layer=jobsite_layer,
        installable_layer=installable_layer,
        harris_params=harris_params,
    )

    # 3) Inner sensor placement
    optimizer_init, optimizer_run = configure_inner_optimizer(
        algorithm=algorithm,
        sensor_range=sensor_range,
        common_params=common_optimizer_params,
        optimizer_params=optimizer_params,
        optimizer_run_params=optimizer_run_params,
    )
    logger = build_logger(
        map_name=map_name,
        results_dir=results_dir,
        optimizer_init=optimizer_init,
        optimizer_run=optimizer_run,
        harris_params=harris_params,
        corner_count=len(corner_points),
        logger_params=logger_params,
    )
    inner_points, optimizer = optimize_inner_sensors(
        algorithm=algorithm,
        installable_layer=installable_layer,
        jobsite_layer=jobsite_layer,
        corner_positions=corner_points,
        optimizer_init=optimizer_init,
        optimizer_run=optimizer_run,
        logger=logger,
    )

    # 4) Final plot + result log
    final_points = inner_points + corner_points
    plot_params = {
        **final_plot_params,
        "radius": final_plot_params.get(
            "radius",
            common_optimizer_params.get("coverage", 45),
        ),
    }
    final_plot_path = plot_final_sensor_placement(
        map_data=map_data,
        logger=logger,
        final_points=final_points,
        plot_params=plot_params,
    )
    out_path = save_result(
        logger=logger,
        optimizer=optimizer,
        inner_points=inner_points,
        corner_points=corner_points,
        final_plot_path=final_plot_path,
    )
    return final_points, out_path
