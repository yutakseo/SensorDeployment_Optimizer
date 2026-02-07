# experiment.py
from Tools.engine import Experiment
from dataclasses import dataclass

@dataclass
class CornerConfig:
    blockSize: int = 3
    ksize: int = 3
    k: float = 0.05
    dilate_size: int = 5
    min_dist: int = 9

@dataclass
class GAInitConfig:
    coverage: int = 45
    generations: int = 100
    initial_size: int = 100
    selection_size: int = 50
    child_chromo_size: int = 100
    min_sensors: int = 40
    max_sensors: int = 140

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


def run_experiments(Iter, DIR):
    for i in range(Iter):
        Experiment(
            map_name="gangjin.full",
            ga_init=GAInitConfig(),
            ga_run=GARunConfig(),
            corner_cfg=CornerConfig(),
            results_dir=DIR,
        ).run()
        
    for i in range(Iter):
        Experiment(
            map_name="gangjin.up",
            ga_init=GAInitConfig(),
            ga_run=GARunConfig(),
            corner_cfg=CornerConfig(),
            results_dir=DIR,
        ).run()

    for i in range(Iter):
        Experiment(
            map_name="gangjin.down",
            ga_init=GAInitConfig(),
            ga_run=GARunConfig(),
            corner_cfg=CornerConfig(),
            results_dir=DIR,
        ).run()

    for i in range(Iter):
        Experiment(
            map_name="sejong.full",
            ga_init=GAInitConfig(),
            ga_run=GARunConfig(),
            corner_cfg=CornerConfig(),
            results_dir=DIR,
        ).run()
        
    for i in range(Iter):
        Experiment(
            map_name="seocho.full",
            ga_init=GAInitConfig(),
            ga_run=GARunConfig(),
            corner_cfg=CornerConfig(),
            results_dir=DIR,
        ).run()

    for i in range(Iter):
        Experiment(
            map_name="seocho.up",
            ga_init=GAInitConfig(),
            ga_run=GARunConfig(),
            corner_cfg=CornerConfig(),
            results_dir=DIR,
        ).run()

    for i in range(Iter):
        Experiment(
            map_name="seocho.down",
            ga_init=GAInitConfig(),
            ga_run=GARunConfig(),
            corner_cfg=CornerConfig(),
            results_dir=DIR,
        ).run()
        
        
        
if __name__ == "__main__":
    Iter=100
    run_experiments(Iter, DIR="__RESULTS__/100_runs/")