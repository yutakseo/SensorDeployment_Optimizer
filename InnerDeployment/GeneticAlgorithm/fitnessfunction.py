# InnerDeployment/GeneticAlgorithm/fitnessfunction.py
from __future__ import annotations

import inspect
import random
import time
from contextlib import contextmanager
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from SensorModule.Sensor import Sensor
from .utils import to_int_pairs

Gene = Tuple[int, int]


class MeanConv(nn.Module):
    def __init__(
        self,
        mapU8: np.ndarray,
        kernels=(3, 5, 7, 9, 11, 13, 15),
        *,
        device: Optional[object] = None,
    ):
        super().__init__()
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.map = torch.as_tensor(mapU8, device=self.device)
        self.mapHalf = self.map.to(torch.float16)

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, 1, k, padding=k // 2, bias=False, padding_mode="replicate") for k in kernels]
        ).to(self.device)

        with torch.no_grad():
            for conv, k in zip(self.convs, kernels):
                conv.weight.fill_(1.0 / (k * k))
                conv.weight.requires_grad_(False)

        self.half()

    def forward(self, x):
        t = torch.as_tensor(x, device=self.device) if isinstance(x, np.ndarray) else x.to(self.device)
        if t.ndim == 2:
            t = t.unsqueeze(0).unsqueeze(0)

        t = t.to(torch.float16)
        out = sum(conv(t) for conv in self.convs) / len(self.convs)
        return out * self.mapHalf.unsqueeze(0).unsqueeze(0)


class FitnessFunc:
    """
    Multi-objective scalar fitness (maximize):
      - reward coverage up to target_coverage
      - penalize sensor count
      - penalize coverage deficit and close overlaps
    """

    def __init__(
        self,
        jobsite_map,
        corner_positions: List[Gene],
        coverage: int,  # sensor radius for Sensor.deploy
        *,
        target_coverage: float = 90.0,
        big_reward: float = 1_000_000.0,
        coverage_weight: float = 1.0,
        sensor_weight: float = 1.0,
        deficit_penalty: float = 20.0,
        overlap_min_dist: Optional[float] = 8.0,
        overlap_penalty: float = 5.0,
        useCache: bool = True,
        sampleCount: Optional[int] = None,
        gainLimit: Optional[float] = None,
        profile_acc: Optional[Dict[str, float]] = None,
        profile_cuda_sync: bool = True,
        device: Optional[object] = None,
    ):
        arr = np.asarray(jobsite_map)
        self.map: np.ndarray = (arr > 0).astype(np.uint8)

        self.coverage = int(coverage)
        self.corners: List[Gene] = to_int_pairs(corner_positions)
        self.cornerKey = tuple(self.corners)

        self.target_coverage = float(target_coverage)
        self.big_reward = float(big_reward)
        self.coverage_weight = float(coverage_weight)
        self.sensor_weight = float(sensor_weight)
        self.deficit_penalty = float(deficit_penalty)
        self.overlap_min_dist = float(overlap_min_dist) if overlap_min_dist is not None else None
        self.overlap_penalty = float(overlap_penalty)

        self.useCache = bool(useCache)
        self.sampleCount = sampleCount
        self.gainLimit = None if gainLimit is None else float(gainLimit)
        self.profile_acc = profile_acc
        self.profile_cuda_sync = bool(profile_cuda_sync)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.mapTensor = torch.as_tensor(self.map, dtype=torch.float16, device=self.device)
        self.mapArea = float(self.mapTensor.sum().item())

        self.cornerMask: Optional[torch.Tensor] = None
        self.singleMask: Dict[Tuple[int, int, int], torch.Tensor] = {}

        self.deployKeys = set(inspect.signature(Sensor.deploy).parameters.keys())
        self.model = MeanConv(self.map, device=self.device)

    @contextmanager
    def _timer(self, name: str, count_key: Optional[str] = None):
        if self.profile_acc is None:
            yield
            return

        if self.profile_cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self.profile_cuda_sync and torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            self.profile_acc[name] = self.profile_acc.get(name, 0.0) + dt
            if count_key:
                self.profile_acc[count_key] = self.profile_acc.get(count_key, 0.0) + 1.0

    def _deploy_many(self, sensor: Sensor, points: Iterable[Sequence[int]]) -> None:
        cov = int(self.coverage)
        pts = to_int_pairs(points)

        if "sensor_position" in self.deployKeys:
            kw = {"sensor_position": pts}
            if "coverage" in self.deployKeys:
                kw["coverage"] = cov
            sensor.deploy(**kw)
            return

        for key in ("sensor_positions", "positions", "corner_positions"):
            if key in self.deployKeys:
                kw = {key: pts}
                if "coverage" in self.deployKeys:
                    kw["coverage"] = cov
                sensor.deploy(**kw)
                return

        try:
            sensor.deploy(pts, coverage=cov)
        except TypeError:
            sensor.deploy(pts)

    def _makeMask(self, points: List[Gene]) -> torch.Tensor:
        with self._timer("fitness_make_mask_total", "fitness_make_mask_calls"):
            key = tuple(points)

            if key == self.cornerKey and self.cornerMask is not None:
                return self.cornerMask

            ck = None
            if self.useCache and len(points) == 1:
                x, y = map(int, points[0])
                ck = (x, y, self.coverage)
                cached = self.singleMask.get(ck)
                if cached is not None:
                    return cached

            sensor = Sensor(self.mapTensor)
            if points:
                self._deploy_many(sensor, points)

            mask = sensor.extract_only_sensor_mask().to(dtype=torch.float16)

            if key == self.cornerKey:
                self.cornerMask = mask
            elif ck is not None:
                self.singleMask[ck] = mask

            return mask

    def _computeCoverage(self, mask: torch.Tensor) -> float:
        with self._timer("fitness_compute_coverage_total", "fitness_compute_coverage_calls"):
            if self.mapArea <= 0:
                return 0.0
            return float(100.0 * (self.mapTensor * mask).sum().item() / self.mapArea)

    def _n_total(self, inner_positions: List[Gene]) -> int:
        return int(len(self.corners) + len(inner_positions))

    def _overlap_cost(self, inner_positions: List[Gene]) -> float:
        if self.overlap_min_dist is None or self.overlap_penalty <= 0:
            return 0.0
        pts = self.corners + to_int_pairs(inner_positions)
        if len(pts) < 2:
            return 0.0
        dmin2 = float(self.overlap_min_dist) ** 2
        cost = 0.0
        for i in range(len(pts)):
            xi, yi = pts[i]
            for j in range(i + 1, len(pts)):
                xj, yj = pts[j]
                dx = float(xi - xj)
                dy = float(yi - yj)
                if (dx * dx + dy * dy) < dmin2:
                    cost += 1.0
        return cost * self.overlap_penalty

    def computeCoverage(self, inner: List[Gene]) -> float:
        pts = self.corners + to_int_pairs(inner)
        return self._computeCoverage(self._makeMask(pts))

    def computeFitness(self, inner: List[Gene]) -> float:
        return self.computeCoverage(inner)

    def fitness_score(self, inner_positions: List[Gene]) -> float:
        return self.computeCoverage(inner_positions)

    def fitness_min_sensors(self, inner_positions: List[Gene]) -> float:
        cov = self.computeCoverage(inner_positions)
        n = self._n_total(inner_positions)
        tau = float(self.target_coverage)
        cov_f = float(cov)
        n_f = float(n)

        deficit = max(0.0, tau - cov_f)
        capped_cov = min(cov_f, tau)
        score = (
            self.coverage_weight * capped_cov
            - self.sensor_weight * n_f
            - self.deficit_penalty * deficit
            - self._overlap_cost(inner_positions)
        )
        return float(score)

    def evaluate(self, inner_positions: List[Gene]):
        cov = self.computeCoverage(inner_positions)
        n = self._n_total(inner_positions)
        fit = self.fitness_min_sensors(inner_positions)
        return float(fit), float(cov), int(n)

    def rankSensor(self, points: List[Gene]):
        with torch.no_grad():
            with self._timer("fitness_mean_conv_total", "fitness_mean_conv_calls"):
                fmap = self.model(self.map.astype(np.float16)).detach()

        result = []
        for p in to_int_pairs(points):
            mask = self._makeMask([p]).unsqueeze(0).unsqueeze(0)
            score = (fmap * mask).sum().item()
            result.append((p, float(score)))

        result.sort(key=lambda x: x[1], reverse=True)
        return result

    def orderSensor(self, chromosome: List[Gene], returnScore: bool = True):
        """
        Greedy marginal-gain ordering: produces a meaningful sequence
        (g1, g2, ...) in descending contribution order for THIS candidate.
        This matches your assumption that chromosome order is meaningful.
        """
        remain = to_int_pairs(chromosome)
        ordered: List[Tuple[Gene, float, float]] = []

        baseMask = self._makeMask(self.corners)
        baseCov = self._computeCoverage(baseMask)

        while remain:
            cand = remain if self.sampleCount is None else random.sample(remain, min(self.sampleCount, len(remain)))

            bestP: Optional[Gene] = None
            bestG = -1e18
            bestC = baseCov
            bestM = baseMask

            for p in cand:
                m = torch.clamp(baseMask + self._makeMask([p]), 0, 1)
                c = self._computeCoverage(m)
                g = c - baseCov
                if g > bestG:
                    bestP, bestG, bestC, bestM = p, g, c, m

            if bestP is None:
                break

            ordered.append((bestP, float(bestG), float(bestC)))
            remain.remove(bestP)

            baseMask, baseCov = bestM, bestC

            if self.gainLimit is not None and bestG < self.gainLimit:
                break

        return ordered if returnScore else [p for (p, _, _) in ordered]

    def ordering_sensors(self, chromosome: List[Gene], return_score: bool = True):
        return self.orderSensor(chromosome, returnScore=return_score)

    def extractUncovered(self, inner: List[Gene]) -> List[Gene]:
        pts = self.corners + to_int_pairs(inner)
        mask = self._makeMask(pts)
        if mask.device != self.mapTensor.device:
            mask = mask.to(self.mapTensor.device)
        u = (self.mapTensor * (1 - mask)).detach().cpu().numpy()
        yx = np.argwhere(u > 0.5)
        return [(int(x), int(y)) for (y, x) in yx]

    def uncovered_map(self, inner_positions: List[Gene]) -> np.ndarray:
        grid = np.zeros_like(self.map, dtype=np.uint8)
        for x, y in self.extractUncovered(inner_positions):
            if 0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]:
                grid[y, x] = 1
        return grid
