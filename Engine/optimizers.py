from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Union

from InnerDeployment.GeneticAlgorithm.main import SensorGA, default_parallel_workers
from InnerDeployment.Greedy.main import SensorGreedy
from InnerDeployment.PSO.main import SensorPSO

Gene = Tuple[int, int]
Chromosome = List[Gene]
Generation = List[Chromosome]


class InnerOptimizerStrategy(ABC):
    def __init__(
        self,
        *,
        installable_map,
        jobsite_map,
        corner_positions: List[Gene],
        init_cfg: Any,
        run_cfg: Any,
        logger,
    ):
        self.installable_map = installable_map
        self.jobsite_map = jobsite_map
        self.corner_positions = corner_positions
        self.init_cfg = init_cfg
        self.run_cfg = run_cfg
        self.logger = logger
        self.optimizer = None

    @abstractmethod
    def build(self):
        raise NotImplementedError

    @abstractmethod
    def run(self) -> Union[Generation, Chromosome]:
        raise NotImplementedError

    @property
    def best_solution(self):
        return getattr(self.optimizer, "best_solution", None)

    @property
    def corner_points(self):
        return getattr(self.optimizer, "corner_points", self.corner_positions)

    @property
    def best_fitness(self) -> float:
        return float(getattr(self.optimizer, "best_fitness", float("nan")))

    @property
    def best_coverage(self) -> float:
        return float(getattr(self.optimizer, "best_coverage", float("nan")))


class GAOptimizerStrategy(InnerOptimizerStrategy):
    def build(self) -> SensorGA:
        gi = self.init_cfg
        self.optimizer = SensorGA(
            installable_map=self.installable_map,
            jobsite_map=self.jobsite_map,
            coverage=gi.coverage,
            generations=gi.generations,
            corner_positions=self.corner_positions,
            initial_size=gi.initial_size,
            selection_size=gi.selection_size,
            child_chromo_size=gi.child_chromo_size,
            min_sensors=gi.min_sensors,
            max_sensors=gi.max_sensors,
            init_min_sensors=getattr(gi, "init_min_sensors", None),
            init_max_sensors=getattr(gi, "init_max_sensors", None),
            fitness_kwargs=getattr(gi, "fitness_kwargs", None),
            mutation_kwargs=getattr(gi, "mutation_kwargs", None),
        )
        return self.optimizer

    def run(self) -> Union[Generation, Chromosome]:
        if self.optimizer is None:
            self.build()
        gr = self.run_cfg
        parallel_workers = getattr(gr, "parallel_workers", None)
        if parallel_workers is None:
            parallel_workers = default_parallel_workers()

        return self.optimizer.run(
            selection_method=gr.selection_method,
            tournament_size=gr.tournament_size,
            mutation_rate=gr.mutation_rate,
            verbose=gr.verbose,
            profile=gr.profile,
            profile_every=gr.profile_every,
            profile_fitness_breakdown=gr.profile_fitness_breakdown,
            early_stop=gr.early_stop,
            early_stop_coverage=gr.early_stop_coverage,
            early_stop_patience=gr.early_stop_patience,
            return_best_only=gr.return_best_only,
            ordering_top_k=getattr(gr, "ordering_top_k", 0),
            mutation_kwargs=getattr(gr, "mutation_kwargs", None),
            parallel_workers=parallel_workers,
            logger=self.logger,
        )


class PSOOptimizerStrategy(InnerOptimizerStrategy):
    def build(self) -> SensorPSO:
        gi = self.init_cfg
        self.optimizer = SensorPSO(
            installable_map=self.installable_map,
            jobsite_map=self.jobsite_map,
            coverage=gi.coverage,
            generations=gi.generations,
            corner_positions=self.corner_positions,
            swarm_size=getattr(gi, "swarm_size", getattr(gi, "initial_size", 100)),
            min_sensors=gi.min_sensors,
            max_sensors=gi.max_sensors,
            initial_min_sensors=getattr(
                gi,
                "initial_min_sensors",
                getattr(gi, "init_min_sensors", None),
            ),
            initial_max_sensors=getattr(
                gi,
                "initial_max_sensors",
                getattr(gi, "init_max_sensors", None),
            ),
            fitness_kwargs=getattr(gi, "fitness_kwargs", None),
        )
        return self.optimizer

    def run(self) -> Union[Generation, Chromosome]:
        if self.optimizer is None:
            self.build()
        gr = self.run_cfg
        return self.optimizer.run(
            inertia=getattr(gr, "inertia", 0.72),
            cognitive=getattr(gr, "cognitive", 1.49),
            social=getattr(gr, "social", 1.49),
            velocity_clip=getattr(gr, "velocity_clip", None),
            count_add_rate=getattr(gr, "count_add_rate", 0.40),
            count_del_rate=getattr(gr, "count_del_rate", 0.30),
            count_change_rate=getattr(
                gr,
                "count_change_rate",
                getattr(gr, "mutation_rate", 0.7),
            ),
            early_stop=getattr(gr, "early_stop", True),
            early_stop_coverage=getattr(gr, "early_stop_coverage", 90.0),
            early_stop_patience=getattr(gr, "early_stop_patience", 10),
            return_best_only=getattr(gr, "return_best_only", True),
            verbose=getattr(gr, "verbose", True),
            profile=getattr(gr, "profile", True),
            profile_every=getattr(gr, "profile_every", 1),
            logger=self.logger,
        )


class GreedyOptimizerStrategy(InnerOptimizerStrategy):
    def build(self) -> SensorGreedy:
        gi = self.init_cfg
        self.optimizer = SensorGreedy(
            installable_map=self.installable_map,
            jobsite_map=self.jobsite_map,
            coverage=gi.coverage,
            corner_positions=self.corner_positions,
            min_sensors=getattr(gi, "min_sensors", 0),
            max_sensors=getattr(gi, "max_sensors", None),
            candidate_stride=getattr(gi, "candidate_stride", 1),
            fitness_kwargs=getattr(gi, "fitness_kwargs", None),
        )
        return self.optimizer

    def run(self) -> Chromosome:
        if self.optimizer is None:
            self.build()
        gr = self.run_cfg
        return self.optimizer.run(
            target_coverage=getattr(gr, "target_coverage", 100.0),
            max_sensors=getattr(gr, "max_sensors", getattr(self.init_cfg, "max_sensors", None)),
            return_best_only=getattr(gr, "return_best_only", True),
            verbose=getattr(gr, "verbose", True),
            profile=getattr(gr, "profile", False),
            profile_every=getattr(gr, "profile_every", 1),
            logger=self.logger,
        )


def make_inner_optimizer(
    *,
    algorithm: str,
    installable_map,
    jobsite_map,
    corner_positions: List[Gene],
    init_cfg: Any,
    run_cfg: Any,
    logger,
) -> InnerOptimizerStrategy:
    key = str(algorithm or "ga").lower()
    if key in {"pso", "swarm", "particle_swarm"}:
        cls = PSOOptimizerStrategy
    elif key in {"greedy", "greedy_search", "recursive_greedy"}:
        cls = GreedyOptimizerStrategy
    else:
        cls = GAOptimizerStrategy
    return cls(
        installable_map=installable_map,
        jobsite_map=jobsite_map,
        corner_positions=corner_positions,
        init_cfg=init_cfg,
        run_cfg=run_cfg,
        logger=logger,
    )
