# experiment.py
from dataclasses import dataclass

from Tools.engine import Experiment


def run_experiments(maps, iter_cnt, results_dir, ga_init, ga_run, corner_cfg):
    for map_name in maps:
        for _ in range(iter_cnt):
            Experiment(
                map_name=map_name,
                ga_init=ga_init,
                ga_run=ga_run,
                corner_cfg=corner_cfg,
                results_dir=results_dir,
            ).run()


if __name__ == "__main__":
    MAPS = [
        "gangjin.full",
        "gangjin.up",
        "gangjin.down",
        "sejong.full",
        "seocho.full",
        "seocho.up",
        "seocho.down",
    ]
    
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
        min_sensors: int = 0
        max_sensors: int = 140
        init_min_sensors: int = 40
        init_max_sensors: int = 140

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


    ga_init = GAInitConfig()
    ga_run = GARunConfig()
    corner_cfg = CornerConfig()
    Iter = 10
    SENSOR_RANGES = [
        (40, 140),
    ]
    
    
    for min_s, max_s in SENSOR_RANGES:
        run_experiments(
            maps=MAPS,
            iter_cnt=Iter,
            results_dir=f"__RESULTS__/{Iter}_rounds/",
            ga_init=ga_init,
            ga_run=ga_run,
            corner_cfg=corner_cfg,
        )
