from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Union

from InnerDeployment.Combinatorial.main import SensorCombinatorial
from InnerDeployment.GeneticAlgorithm.main import SensorGA, default_parallel_workers
from InnerDeployment.Greedy.main import SensorGreedy
from InnerDeployment.PSO.main import SensorPSO
from InnerDeployment.DRL.main import SensorDRL

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

    def close(self) -> None:
        if self.optimizer is not None and hasattr(self.optimizer, "close"):
            self.optimizer.close()
        self.optimizer = None
        self.installable_map = None
        self.jobsite_map = None
        self.corner_positions = []


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
            cognitive=getattr(gr, "cognitive", 2.0),
            social=getattr(gr, "social", 2.0),
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
            min_separation=getattr(gi, "min_separation", None),
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


class CombinatorialOptimizerStrategy(InnerOptimizerStrategy):
    def build(self) -> SensorCombinatorial:
        gi = self.init_cfg
        self.optimizer = SensorCombinatorial(
            installable_map=self.installable_map,
            jobsite_map=self.jobsite_map,
            coverage=gi.coverage,
            corner_positions=self.corner_positions,
            min_sensors=getattr(gi, "min_sensors", 0),
            max_sensors=getattr(gi, "max_sensors", None),
            candidate_stride=getattr(gi, "candidate_stride", 1),
            max_candidates=getattr(gi, "max_candidates", 24),
            max_combinations=getattr(gi, "max_combinations", 5_000_000),
            min_separation=getattr(gi, "min_separation", None),
            parallel_workers=getattr(gi, "parallel_workers", None),
            chunk_size=getattr(gi, "chunk_size", 4096),
            fitness_kwargs=getattr(gi, "fitness_kwargs", None),
        )
        return self.optimizer

    def run(self) -> Chromosome:
        if self.optimizer is None:
            self.build()
        gr = self.run_cfg
        return self.optimizer.run(
            target_coverage=getattr(gr, "target_coverage", 100.0),
            return_best_only=getattr(gr, "return_best_only", True),
            verbose=getattr(gr, "verbose", True),
            profile=getattr(gr, "profile", False),
            profile_every=getattr(gr, "profile_every", 100_000),
            parallel_workers=getattr(gr, "parallel_workers", None),
            chunk_size=getattr(gr, "chunk_size", None),
            fitness_log_path=getattr(gr, "fitness_log_path", None),
            fitness_trace_stride=getattr(gr, "fitness_trace_stride", 1),
            sample_combinations=getattr(gr, "sample_combinations", None),
            sample_seed=getattr(gr, "sample_seed", 42),
            logger=self.logger,
        )


class DRLOptimizerStrategy(InnerOptimizerStrategy):
    def build(self) -> SensorDRL:
        gi = self.init_cfg
        self.optimizer = SensorDRL(
            installable_map=self.installable_map,
            jobsite_map=self.jobsite_map,
            coverage=gi.coverage,
            generations=gi.generations,
            corner_positions=self.corner_positions,
            min_sensors=getattr(gi, "min_sensors", 0),
            max_sensors=getattr(gi, "max_sensors", 140),
            candidate_stride=getattr(gi, "candidate_stride", 5),
            max_candidates=getattr(gi, "max_candidates", 512),
            min_separation=getattr(gi, "min_separation", None),
            hidden_dim=getattr(gi, "hidden_dim", 128),
            replay_capacity=getattr(gi, "replay_capacity", 5000),
            batch_size=getattr(gi, "batch_size", 64),
            learning_rate=getattr(gi, "learning_rate", 1e-3),
            gamma=getattr(gi, "gamma", 0.95),
            target_sync_interval=getattr(gi, "target_sync_interval", 100),
            warmup_steps=getattr(gi, "warmup_steps", 64),
            train_steps_per_action=getattr(gi, "train_steps_per_action", 1),
            backup_actions=getattr(gi, "backup_actions", 64),
            seed=getattr(gi, "seed", 42),
            device=getattr(gi, "device", None),
            fitness_kwargs=getattr(gi, "fitness_kwargs", None),
        )
        return self.optimizer

    def run(self) -> Chromosome:
        if self.optimizer is None:
            self.build()
        gr = self.run_cfg
        return self.optimizer.run(
            epsilon_start=getattr(gr, "epsilon_start", 1.0),
            epsilon_end=getattr(gr, "epsilon_end", 0.05),
            epsilon_decay=getattr(gr, "epsilon_decay", 0.985),
            heuristic_warmup_episodes=getattr(gr, "heuristic_warmup_episodes", 1),
            return_best_only=getattr(gr, "return_best_only", True),
            verbose=getattr(gr, "verbose", True),
            profile=getattr(gr, "profile", True),
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
    elif key in {"combinatorial", "exact", "bruteforce", "brute_force"}:
        cls = CombinatorialOptimizerStrategy
    elif key in {"drl", "dqn", "deep_q_learning"}:
        cls = DRLOptimizerStrategy
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
