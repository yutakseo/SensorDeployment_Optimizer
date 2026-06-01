from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

MapList = Sequence[str]
SensorRange = Tuple[int, int]


@dataclass
class CornerConfig:
    blockSize: int = 3
    ksize: int = 3
    k: float = 0.05
    dilate_size: int = 5
    min_dist: int = 9


@dataclass
class GAConfig:
    algorithm: str = "ga"
    coverage: int = 45
    generations: int = 100
    initial_size: int = 100
    selection_size: int = 50
    child_chromo_size: int = 100
    min_sensors: int = 0
    max_sensors: int = 140
    init_min_sensors: int = 40
    init_max_sensors: int = 140
    fitness_kwargs: Optional[dict] = None
    mutation_kwargs: Optional[dict] = None


@dataclass
class GARunConfig:
    selection_method: str = "elite"
    tournament_size: int = 3
    mutation_rate: float = 0.7
    early_stop: bool = False
    early_stop_coverage: float = 90.0
    early_stop_patience: int = 5
    return_best_only: bool = True
    verbose: bool = True
    profile: bool = True
    profile_every: int = 1
    profile_fitness_breakdown: bool = True
    ordering_top_k: int = 0
    parallel_workers: Optional[int] = None
    mutation_kwargs: Optional[dict] = None


@dataclass
class PSOConfig:
    algorithm: str = "pso"
    coverage: int = 45
    generations: int = 100
    swarm_size: int = 100
    min_sensors: int = 0
    max_sensors: int = 140
    initial_min_sensors: int = 40
    initial_max_sensors: int = 140
    fitness_kwargs: Optional[dict] = None


@dataclass
class PSORunConfig:
    inertia: float = 0.72
    cognitive: float = 2.0
    social: float = 2.0
    velocity_clip: Optional[float] = None
    count_add_rate: float = 0.40
    count_del_rate: float = 0.30
    count_change_rate: float = 0.7
    early_stop: bool = False
    early_stop_coverage: float = 90.0
    early_stop_patience: int = 5
    return_best_only: bool = True
    verbose: bool = True
    profile: bool = True
    profile_every: int = 1


@dataclass
class GreedyConfig:
    algorithm: str = "greedy"
    coverage: int = 45
    min_sensors: int = 0
    max_sensors: int = 140
    candidate_stride: int = 1
    fitness_kwargs: Optional[dict] = None


@dataclass
class GreedyRunConfig:
    target_coverage: float = 100.0
    max_sensors: Optional[int] = None
    return_best_only: bool = True
    verbose: bool = True
    profile: bool = False
    profile_every: int = 1


def _with_overrides(cfg: Any, overrides: Optional[Dict[str, Any]]) -> Any:
    if not overrides:
        return cfg
    valid = cfg.__dataclass_fields__.keys()
    unknown = sorted(set(overrides) - set(valid))
    if unknown:
        raise ValueError(f"Unknown config fields for {type(cfg).__name__}: {unknown}")
    return replace(cfg, **overrides)


def make_optimizer_configs(
    algorithm: str = "ga",
    *,
    coverage: int = 45,
    generations: int = 100,
    sensor_range: SensorRange = (40, 140),
    optimizer: Optional[Dict[str, Any]] = None,
    run: Optional[Dict[str, Any]] = None,
):
    low, high = int(sensor_range[0]), int(sensor_range[1])
    key = str(algorithm).lower()

    if key in {"pso", "swarm", "particle_swarm"}:
        init_cfg = PSOConfig(
            coverage=int(coverage),
            generations=int(generations),
            min_sensors=0,
            max_sensors=int(high),
            initial_min_sensors=int(low),
            initial_max_sensors=int(high),
        )
        run_cfg = PSORunConfig()
    elif key in {"ga", "genetic", "genetic_algorithm"}:
        init_cfg = GAConfig(
            coverage=int(coverage),
            generations=int(generations),
            min_sensors=0,
            max_sensors=int(high),
            init_min_sensors=int(low),
            init_max_sensors=int(high),
        )
        run_cfg = GARunConfig()
    elif key in {"greedy", "greedy_search", "recursive_greedy"}:
        init_cfg = GreedyConfig(
            coverage=int(coverage),
            min_sensors=0,
            max_sensors=int(high),
        )
        run_cfg = GreedyRunConfig(max_sensors=int(high))
    else:
        raise ValueError(f"Unsupported optimizer algorithm: {algorithm}")

    return _with_overrides(init_cfg, optimizer), _with_overrides(run_cfg, run)


def run_experiment(
    *,
    maps: MapList,
    algorithm: str = "ga",
    sensor_range: SensorRange = (40, 140),
    iterations: int = 1,
    coverage: int = 45,
    generations: int = 100,
    results_dir: Optional[str] = None,
    optimizer: Optional[Dict[str, Any]] = None,
    run: Optional[Dict[str, Any]] = None,
    corner: Optional[Dict[str, Any]] = None,
    logger_point_format: str = "tuple_str",
    logger_sort_points: bool = False,
) -> List[Tuple[List[Tuple[int, int]], str]]:
    from Engine.experiment import Experiment

    init_cfg, run_cfg = make_optimizer_configs(
        algorithm=algorithm,
        coverage=coverage,
        generations=generations,
        sensor_range=sensor_range,
        optimizer=optimizer,
        run=run,
    )
    corner_cfg = _with_overrides(CornerConfig(), corner)
    out_dir = results_dir or f"__RESULTS__/{algorithm}/{sensor_range[0]}-{sensor_range[1]}"

    results: List[Tuple[List[Tuple[int, int]], str]] = []
    for map_name in maps:
        for _ in range(int(iterations)):
            results.append(
                Experiment(
                    map_name=map_name,
                    optimizer_init=init_cfg,
                    optimizer_run=run_cfg,
                    corner_cfg=corner_cfg,
                    results_dir=out_dir,
                    logger_point_format=logger_point_format,
                    logger_sort_points=logger_sort_points,
                ).run()
            )
    return results


def run_batch(
    *,
    maps: MapList,
    algorithm: str = "ga",
    sensor_ranges: Iterable[SensorRange],
    iterations: int = 1,
    coverage: int = 45,
    generations: int = 100,
    results_root: Optional[str] = None,
    optimizer: Optional[Dict[str, Any]] = None,
    run: Optional[Dict[str, Any]] = None,
    corner: Optional[Dict[str, Any]] = None,
) -> Dict[SensorRange, List[Tuple[List[Tuple[int, int]], str]]]:
    root = results_root or f"__RESULTS__/{algorithm}"
    results: Dict[SensorRange, List[Tuple[List[Tuple[int, int]], str]]] = {}

    for sensor_range in sensor_ranges:
        low, high = int(sensor_range[0]), int(sensor_range[1])
        key = (low, high)
        results[key] = run_experiment(
            maps=maps,
            algorithm=algorithm,
            sensor_range=key,
            iterations=iterations,
            coverage=coverage,
            generations=generations,
            results_dir=f"{root}/{low}-{high}",
            optimizer=optimizer,
            run=run,
            corner=corner,
        )
    return results
