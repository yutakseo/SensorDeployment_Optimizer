from __future__ import annotations

import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..fitnessfunction import FitnessFunc
from ..utils import to_int_pairs

Gene = Tuple[int, int]
Chromosome = List[Gene]


@dataclass
class Transition:
    action_features: np.ndarray
    reward: float
    next_action_features: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int):
        self._items: Deque[Transition] = deque(maxlen=max(1, int(capacity)))

    def append(self, transition: Transition) -> None:
        self._items.append(transition)

    def sample(self, size: int) -> List[Transition]:
        return random.sample(self._items, min(int(size), len(self._items)))

    def __len__(self) -> int:
        return len(self._items)


class CandidateQNetwork(nn.Module):
    """Score one candidate action from the current deployment context."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        hidden = max(8, int(hidden_dim))
        self.layers = nn.Sequential(
            nn.Linear(int(input_dim), hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features).squeeze(-1)


class SensorPlacementEnv:
    """
    Lightweight environment for sequential sensor placement.

    Actions are candidate indices. Features are recomputed after every action,
    allowing one Q-network to handle maps with different candidate counts.
    """

    FEATURE_DIM = 7

    def __init__(
        self,
        *,
        installable_map,
        jobsite_map,
        corner_positions: List[Gene],
        coverage: int,
        min_sensors: int,
        max_sensors: int,
        target_coverage: float,
        candidate_stride: int,
        max_candidates: Optional[int],
        reward_coverage: float,
        sensor_penalty: float,
        target_bonus: float,
        deficit_penalty: float,
    ):
        self.installable_map = np.asarray(installable_map) > 0
        self.jobsite_map = np.asarray(jobsite_map) > 0
        if self.installable_map.shape != self.jobsite_map.shape:
            raise ValueError("installable_map and jobsite_map must have the same shape.")

        self.height, self.width = self.jobsite_map.shape
        self.corner_positions = to_int_pairs(corner_positions)
        self.coverage_cells = max(0, int(coverage) // 5)
        self.min_sensors = max(0, int(min_sensors))
        self.max_sensors = max(self.min_sensors, int(max_sensors))
        self.target_coverage = max(0.0, min(100.0, float(target_coverage)))
        self.reward_coverage = float(reward_coverage)
        self.sensor_penalty = float(sensor_penalty)
        self.target_bonus = float(target_bonus)
        self.deficit_penalty = float(deficit_penalty)
        self.target_flat = self.jobsite_map.reshape(-1)
        self.target_area = max(1, int(np.count_nonzero(self.target_flat)))
        self.offsets = self._circle_offsets(self.coverage_cells)

        stride = max(1, int(candidate_stride))
        mask = self.installable_map.copy()
        for x, y in self.corner_positions:
            if 0 <= x < self.width and 0 <= y < self.height:
                mask[y, x] = False
        ys, xs = np.where(mask)
        ys = ys[::stride]
        xs = xs[::stride]
        points = [(int(x), int(y)) for y, x in zip(ys.tolist(), xs.tolist())]
        indices = [self._covered_indices(point) for point in points]

        if max_candidates is not None and len(points) > int(max_candidates):
            limit = max(1, int(max_candidates))
            static_rank = np.argsort([-len(item) for item in indices])
            spatial = np.linspace(0, len(points) - 1, num=limit, dtype=np.int64)
            order = []
            seen = set()
            for ranked, spread in zip(static_rank.tolist(), spatial.tolist()):
                for index in (ranked, spread):
                    if index not in seen:
                        seen.add(index)
                        order.append(index)
                        if len(order) >= limit:
                            break
                if len(order) >= limit:
                    break
            points = [points[int(i)] for i in order]
            indices = [indices[int(i)] for i in order]
        if not points:
            raise ValueError("installable_map has no candidates for DRL sensor placement.")

        self.candidates = points
        self.candidate_indices = indices
        self.static_gains = np.asarray([len(item) for item in indices], dtype=np.float32)
        self.covered = np.zeros(self.height * self.width, dtype=bool)
        self.selected = np.zeros(len(points), dtype=bool)
        self.solution: Chromosome = []
        self.coverage_percent = 0.0

    @staticmethod
    def _circle_offsets(radius: int) -> np.ndarray:
        offsets = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    offsets.append((dy, dx))
        return np.asarray(offsets, dtype=np.int32)

    def _covered_indices(self, point: Gene) -> np.ndarray:
        x, y = point
        yy = int(y) + self.offsets[:, 0]
        xx = int(x) + self.offsets[:, 1]
        valid = (yy >= 0) & (yy < self.height) & (xx >= 0) & (xx < self.width)
        lin = (yy[valid] * self.width + xx[valid]).astype(np.int64, copy=False)
        return lin[self.target_flat[lin]]

    def _coverage(self) -> float:
        return float(100.0 * np.count_nonzero(self.covered & self.target_flat) / self.target_area)

    def reset(self) -> np.ndarray:
        self.covered.fill(False)
        self.selected.fill(False)
        self.solution = []
        for point in self.corner_positions:
            self.covered[self._covered_indices(point)] = True
        self.coverage_percent = self._coverage()
        return self.action_features()

    def available_indices(self) -> np.ndarray:
        return np.flatnonzero(~self.selected)

    def marginal_gains(self, available: Optional[np.ndarray] = None) -> np.ndarray:
        actions = self.available_indices() if available is None else available
        return np.asarray(
            [np.count_nonzero(~self.covered[self.candidate_indices[int(i)]]) for i in actions],
            dtype=np.float32,
        )

    def action_features(
        self,
        *,
        max_actions: Optional[int] = None,
    ) -> np.ndarray:
        actions = self.available_indices()
        if actions.size == 0:
            return np.empty((0, self.FEATURE_DIM + 1), dtype=np.float32)
        gains = self.marginal_gains(actions)
        if max_actions is not None and actions.size > int(max_actions):
            keep = np.argsort(-gains)[: int(max_actions)]
            actions = actions[keep]
            gains = gains[keep]

        points = np.asarray([self.candidates[int(i)] for i in actions], dtype=np.float32)
        n = len(actions)
        coverage = self.coverage_percent / 100.0
        step = len(self.solution) / max(1, self.max_sensors)
        features = np.column_stack(
            [
                actions.astype(np.float32),
                np.full(n, coverage, dtype=np.float32),
                np.full(n, step, dtype=np.float32),
                points[:, 0] / max(1, self.width - 1),
                points[:, 1] / max(1, self.height - 1),
                gains / self.target_area,
                self.static_gains[actions] / self.target_area,
                np.full(n, max(0.0, self.target_coverage / 100.0 - coverage), dtype=np.float32),
            ]
        )
        return features.astype(np.float32, copy=False)

    def step(self, action_index: int) -> Tuple[float, bool]:
        action = int(action_index)
        if action < 0 or action >= len(self.candidates) or self.selected[action]:
            raise ValueError(f"Invalid or already selected DRL action: {action}")

        previous = self.coverage_percent
        self.selected[action] = True
        self.solution.append(self.candidates[action])
        self.covered[self.candidate_indices[action]] = True
        self.coverage_percent = self._coverage()

        reached_target = (
            self.coverage_percent >= self.target_coverage
            and len(self.solution) >= self.min_sensors
        )
        exhausted = len(self.solution) >= self.max_sensors or not np.any(~self.selected)
        done = bool(reached_target or exhausted)
        reward = (
            self.reward_coverage * (self.coverage_percent - previous)
            - self.sensor_penalty
        )
        if reached_target:
            reward += self.target_bonus
        elif exhausted:
            reward -= self.deficit_penalty * max(0.0, self.target_coverage - self.coverage_percent)
        return float(reward), done


class SensorDRL:
    """DQN-based optimizer compatible with the existing InnerDeployment API."""

    def __init__(
        self,
        installable_map,
        jobsite_map,
        coverage: int,
        generations: int,
        corner_positions: List[Gene],
        min_sensors: int = 0,
        max_sensors: int = 140,
        candidate_stride: int = 5,
        max_candidates: Optional[int] = 512,
        hidden_dim: int = 128,
        replay_capacity: int = 5000,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        gamma: float = 0.95,
        target_sync_interval: int = 100,
        warmup_steps: int = 64,
        train_steps_per_action: int = 1,
        backup_actions: int = 64,
        reward_coverage: float = 1.0,
        sensor_penalty: float = 0.2,
        target_bonus: float = 10.0,
        deficit_penalty: float = 1.0,
        seed: int = 42,
        device: Optional[str] = None,
        fitness_kwargs: Optional[dict] = None,
    ):
        random.seed(int(seed))
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.generations = max(1, int(generations))
        self.corner_positions = to_int_pairs(corner_positions)
        self.corner_points = list(self.corner_positions)
        self.coverage = int(coverage)
        self.min_sensors = max(0, int(min_sensors))
        self.max_sensors = max(self.min_sensors, int(max_sensors))
        self.fitness_kwargs = dict(fitness_kwargs or {})
        target = float(self.fitness_kwargs.get("target_coverage", 90.0))

        self.env = SensorPlacementEnv(
            installable_map=installable_map,
            jobsite_map=jobsite_map,
            corner_positions=self.corner_positions,
            coverage=self.coverage,
            min_sensors=self.min_sensors,
            max_sensors=self.max_sensors,
            target_coverage=target,
            candidate_stride=candidate_stride,
            max_candidates=max_candidates,
            reward_coverage=reward_coverage,
            sensor_penalty=sensor_penalty,
            target_bonus=target_bonus,
            deficit_penalty=deficit_penalty,
        )
        self.q_network = CandidateQNetwork(self.env.FEATURE_DIM, hidden_dim).to(self.device)
        self.target_network = CandidateQNetwork(self.env.FEATURE_DIM, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=float(learning_rate))
        self.loss_fn = nn.SmoothL1Loss()
        self.replay = ReplayBuffer(replay_capacity)
        self.batch_size = max(1, int(batch_size))
        self.gamma = float(gamma)
        self.target_sync_interval = max(1, int(target_sync_interval))
        self.warmup_steps = max(1, int(warmup_steps))
        self.train_steps_per_action = max(1, int(train_steps_per_action))
        self.backup_actions = max(1, int(backup_actions))
        self.train_steps = 0

        self.best_solution: Chromosome = []
        self.best_fitness = float("-inf")
        self.best_coverage = float("nan")

    def _choose_action(self, features: np.ndarray, epsilon: float, *, heuristic: bool = False) -> int:
        if features.size == 0:
            raise RuntimeError("No DRL action is available.")
        if heuristic:
            return int(features[int(np.argmax(features[:, 5])), 0])
        if random.random() < float(epsilon):
            return int(features[random.randrange(len(features)), 0])
        with torch.no_grad():
            inputs = torch.as_tensor(features[:, 1:], dtype=torch.float32, device=self.device)
            return int(features[int(torch.argmax(self.q_network(inputs)).item()), 0])

    def _optimize(self) -> Optional[float]:
        if len(self.replay) < max(self.batch_size, self.warmup_steps):
            return None
        batch = self.replay.sample(self.batch_size)
        action_features = torch.as_tensor(
            np.stack([item.action_features for item in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        rewards = torch.as_tensor([item.reward for item in batch], dtype=torch.float32, device=self.device)
        done = torch.as_tensor([item.done for item in batch], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_values = []
            for item in batch:
                if item.done or item.next_action_features.size == 0:
                    next_values.append(0.0)
                    continue
                next_inputs = torch.as_tensor(
                    item.next_action_features[:, 1:],
                    dtype=torch.float32,
                    device=self.device,
                )
                next_values.append(float(self.target_network(next_inputs).max().item()))
            next_q = torch.as_tensor(next_values, dtype=torch.float32, device=self.device)
            target = rewards + self.gamma * (1.0 - done) * next_q

        predicted = self.q_network(action_features)
        loss = self.loss_fn(predicted, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=5.0)
        self.optimizer.step()
        self.train_steps += 1
        if self.train_steps % self.target_sync_interval == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        return float(loss.item())

    def _run_episode(self, epsilon: float, *, heuristic: bool, train: bool) -> Tuple[Chromosome, float, float]:
        features = self.env.reset()
        total_reward = 0.0
        losses = []
        done = False
        while not done and features.size:
            action = self._choose_action(features, epsilon, heuristic=heuristic)
            selected = features[features[:, 0] == action][0, 1:].copy()
            reward, done = self.env.step(action)
            next_features = self.env.action_features(max_actions=self.backup_actions)
            total_reward += reward
            if train:
                self.replay.append(
                    Transition(
                        action_features=selected,
                        reward=reward,
                        next_action_features=next_features.copy(),
                        done=done,
                    )
                )
                for _ in range(self.train_steps_per_action):
                    loss = self._optimize()
                    if loss is not None:
                        losses.append(loss)
            features = self.env.action_features()
        avg_loss = float(np.mean(losses)) if losses else float("nan")
        return list(self.env.solution), float(total_reward), avg_loss

    def _prune_solution(
        self,
        evaluator: FitnessFunc,
        chromosome: Chromosome,
        *,
        target_coverage: float,
    ) -> Tuple[Chromosome, float, float]:
        solution = list(dict.fromkeys(to_int_pairs(chromosome)))
        fitness, coverage, _ = evaluator.evaluate(solution)
        if coverage < target_coverage:
            return solution, float(fitness), float(coverage)
        while len(solution) > self.min_sensors:
            candidates = []
            for index in range(len(solution)):
                candidate = solution[:index] + solution[index + 1 :]
                fit, cov, _ = evaluator.evaluate(candidate)
                if cov >= target_coverage:
                    candidates.append((float(fit), float(cov), candidate))
            if not candidates:
                break
            fitness, coverage, solution = max(candidates, key=lambda item: item[0])
        return solution, float(fitness), float(coverage)

    def _fitness_from_coverage(
        self,
        evaluator: FitnessFunc,
        solution: Chromosome,
        coverage: float,
    ) -> float:
        deficit = max(0.0, float(evaluator.target_coverage) - float(coverage))
        return float(
            evaluator.coverage_weight * min(float(coverage), float(evaluator.target_coverage))
            - evaluator.sensor_weight * (len(solution) + len(self.corner_positions))
            - evaluator.deficit_penalty * deficit
            - evaluator._overlap_cost(solution)
        )

    def run(
        self,
        *,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.985,
        heuristic_warmup_episodes: int = 1,
        verbose: bool = True,
        profile: bool = True,
        profile_every: int = 1,
        return_best_only: bool = True,
        logger=None,
    ) -> Chromosome:
        del return_best_only
        t0 = time.perf_counter()
        evaluator = FitnessFunc(
            jobsite_map=self.env.jobsite_map,
            corner_positions=self.corner_positions,
            coverage=self.coverage,
            **self.fitness_kwargs,
        )
        target = float(evaluator.target_coverage)
        epsilon = float(epsilon_start)
        candidate_solutions: List[Chromosome] = []

        for episode in range(1, self.generations + 1):
            ep_t0 = time.perf_counter()
            solution, reward, loss = self._run_episode(
                epsilon,
                heuristic=episode <= int(heuristic_warmup_episodes),
                train=True,
            )
            candidate_solutions.append(solution)
            coverage = float(self.env.coverage_percent)
            fitness = self._fitness_from_coverage(evaluator, solution, coverage)
            if fitness > self.best_fitness:
                self.best_solution = list(solution)
                self.best_fitness = float(fitness)
                self.best_coverage = coverage

            if logger is not None:
                logger.log_generation(
                    gen=episode,
                    sensors_min=float(len(solution)),
                    sensors_max=float(len(solution)),
                    sensors_avg=float(len(solution) + len(self.corner_positions)),
                    fitness_min=float(fitness),
                    fitness_max=float(fitness),
                    fitness_avg=float(fitness),
                    best_solution=self.best_solution,
                    best_fitness=float(self.best_fitness),
                    best_coverage=float(self.best_coverage),
                )
            if verbose and (episode == 1 or episode % max(1, int(profile_every)) == 0):
                time_part = f" / time={time.perf_counter() - ep_t0:.3f}s" if profile else ""
                print(
                    f"[DRL {episode:03d}/{self.generations:03d}] "
                    f"inner={len(solution)} / coverage={coverage:.2f}% / "
                    f"reward={reward:.3f} / loss={loss:.5f} / epsilon={epsilon:.3f}"
                    f"{time_part}"
                )
            epsilon = max(float(epsilon_end), epsilon * float(epsilon_decay))

        policy_solution, _, _ = self._run_episode(0.0, heuristic=False, train=False)
        candidate_solutions.extend([self.best_solution, policy_solution])
        evaluated = [(*evaluator.evaluate(solution)[:2], solution) for solution in candidate_solutions]
        _, _, best = max(evaluated, key=lambda item: item[0])
        self.best_solution, self.best_fitness, self.best_coverage = self._prune_solution(
            evaluator,
            best,
            target_coverage=target,
        )
        self.corner_points = list(self.corner_positions)
        if verbose:
            print(
                f"[DRL Final] inner={len(self.best_solution)} / corner={len(self.corner_positions)} / "
                f"coverage={self.best_coverage:.2f}% / time={time.perf_counter() - t0:.3f}s"
            )
        return list(self.best_solution)
