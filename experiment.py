# experiment.py
from Tools.engine import Experiment
from dataclasses import dataclass

@dataclass
class CornerConfig:
    blockSize: int = 3
    ksize: int = 3
    k: float = 0.01
    dilate_size: int = 5
    min_dist: int = 5

@dataclass
class GAInitConfig:
    coverage: int = 45
    generations: int = 100
    initial_size: int = 100
    selection_size: int = 50
    child_chromo_size: int = 100
    min_sensors: int = 50
    max_sensors: int = 100

@dataclass
class GARunConfig:
    selection_method: str = "tournament"
    tournament_size: int = 3
    mutation_rate: float = 0.7
    early_stop: bool = True
    early_stop_coverage: float = 90.0
    early_stop_patience: int = 5
    return_best_only: bool = True
    verbose: bool = True
    profile: bool = True
    profile_every: int = 1
    profile_fitness_breakdown: bool = True




for i in range(10):
    Experiment(
        map_name="gangjin.full",
        ga_init=GAInitConfig(),
        ga_run=GARunConfig(),
        corner_cfg=CornerConfig(),
    ).run()
    
for i in range(10):
    Experiment(
        map_name="gangjin.crop1",
        ga_init=GAInitConfig(),
        ga_run=GARunConfig(),
        corner_cfg=CornerConfig(),
    ).run()

for i in range(10):
    Experiment(
        map_name="gangjin.crop2",
        ga_init=GAInitConfig(),
        ga_run=GARunConfig(),
        corner_cfg=CornerConfig(),
    ).run()

for i in range(10):
    Experiment(
        map_name="sejong.full",
        ga_init=GAInitConfig(),
        ga_run=GARunConfig(),
        corner_cfg=CornerConfig(),
    ).run()
    
for i in range(10):
    Experiment(
        map_name="seocho.full",
        ga_init=GAInitConfig(),
        ga_run=GARunConfig(),
        corner_cfg=CornerConfig(),
    ).run()

for i in range(10):
    Experiment(
        map_name="seocho.crop1",
        ga_init=GAInitConfig(),
        ga_run=GARunConfig(),
        corner_cfg=CornerConfig(),
    ).run()

for i in range(10):
    Experiment(
        map_name="seocho.crop2",
        ga_init=GAInitConfig(),
        ga_run=GARunConfig(),
        corner_cfg=CornerConfig(),
    ).run()