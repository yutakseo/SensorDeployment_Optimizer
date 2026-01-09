# InnerDeployment/GeneticAlgorithm/fitnessfunction.py
from __future__ import annotations

import inspect
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from SensorModule.Sensor import Sensor

Gene = Tuple[int, int]


def toInt(points) -> List[Gene]:
    return [tuple(map(int, p)) for p in points]


# ==================================================
# Mean Convolution (coverage potential map)
# ==================================================
class MeanConv(nn.Module):
    def __init__(self, mapU8: np.ndarray, kernels=(3, 5, 7, 9, 11, 13, 15)):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


# ==================================================
# Fitness Function
# ==================================================
class FitnessFunc:
    """
    coverage% fitness + corner-first greedy ordering
    """

    def __init__(
        self,
        jobsite_map,
        corner_positions: List[Gene],
        coverage: int,
        *,
        useCache: bool = True,
        sampleCount: Optional[int] = None,
        gainLimit: Optional[float] = None,
    ):
        arr = np.asarray(jobsite_map)
        self.map: np.ndarray = (arr > 0).astype(np.uint8)

        self.coverage = int(coverage)
        self.corners: List[Gene] = toInt(corner_positions)
        self.cornerKey = tuple(self.corners)

        self.useCache = bool(useCache)
        self.sampleCount = sampleCount
        self.gainLimit = None if gainLimit is None else float(gainLimit)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mapTensor = torch.as_tensor(self.map, dtype=torch.float16, device=self.device)
        self.mapArea = float(self.mapTensor.sum().item())

        self.cornerMask: Optional[torch.Tensor] = None
        self.singleMask: Dict[Tuple[int, int, int], torch.Tensor] = {}

        self.deployKeys = set(inspect.signature(Sensor.deploy).parameters.keys())
        self.model = MeanConv(self.map)

    # -------------------------
    # internal
    # -------------------------
    def _deploy(self, sensor: Sensor, point: Gene):
        x, y = int(point[0]), int(point[1])
        cov = int(self.coverage)

        if "sensor_position" in self.deployKeys:
            kw = {"sensor_position": (x, y)}
            if "coverage" in self.deployKeys:
                kw["coverage"] = cov
            sensor.deploy(**kw)
            return

        for key in ("sensor_positions", "positions", "corner_positions"):
            if key in self.deployKeys:
                kw = {key: [(x, y)]}
                if "coverage" in self.deployKeys:
                    kw["coverage"] = cov
                sensor.deploy(**kw)
                return

        try:
            sensor.deploy((x, y), coverage=cov)
        except TypeError:
            sensor.deploy((x, y))

    def _makeMask(self, points: List[Gene]) -> torch.Tensor:
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

        sensor = Sensor(self.map)
        for p in points:
            self._deploy(sensor, p)

        raw = torch.as_tensor(sensor.extract_only_sensor(), dtype=torch.float16, device=self.device)
        mask = (raw > 0).float()

        if key == self.cornerKey:
            self.cornerMask = mask
        elif ck is not None:
            self.singleMask[ck] = mask

        return mask

    def _computeCoverage(self, mask: torch.Tensor) -> float:
        if self.mapArea <= 0:
            return 0.0
        return float(100.0 * (self.mapTensor * mask).sum().item() / self.mapArea)

    # -------------------------
    # public API
    # -------------------------
    def computeFitness(self, inner: List[Gene]) -> float:
        pts = self.corners + toInt(inner)
        return self._computeCoverage(self._makeMask(pts))

    def fitness_score(self, inner_positions: List[Gene]) -> float:
        return self.computeFitness(inner_positions)

    def evaluate(self, inner_positions: List[Gene]):
        cov = self.computeFitness(inner_positions)
        return float(cov), float(cov), len(self.corners) + len(inner_positions)

    def rankSensor(self, points: List[Gene]):
        with torch.no_grad():
            fmap = self.model(self.map.astype(np.float16)).detach()

        result = []
        for p in toInt(points):
            mask = self._makeMask([p]).unsqueeze(0).unsqueeze(0)
            score = (fmap * mask).sum().item()
            result.append((p, float(score)))

        result.sort(key=lambda x: x[1], reverse=True)
        return result

    def orderSensor(self, chromosome: List[Gene], returnScore: bool = True):
        remain = toInt(chromosome)
        ordered = []

        baseMask = self._makeMask(self.corners)
        baseCov = self._computeCoverage(baseMask)

        while remain:
            cand = remain if self.sampleCount is None else random.sample(remain, min(self.sampleCount, len(remain)))

            bestP, bestG, bestC, bestM = None, -1e18, None, None

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

    # ---- backward compatible wrapper ----
    def ordering_sensors(self, chromosome: List[Gene], return_score: bool = True):
        return self.orderSensor(chromosome, returnScore=return_score)

    def extractUncovered(self, inner: List[Gene]) -> List[Gene]:
        pts = self.corners + toInt(inner)
        mask = self._makeMask(pts)
        u = (self.mapTensor * (1 - mask)).detach().cpu().numpy()
        yx = np.argwhere(u > 0.5)
        return [(int(x), int(y)) for (y, x) in yx]

    def uncovered_map(self, inner_positions: List[Gene]) -> np.ndarray:
        grid = np.zeros_like(self.map, dtype=np.uint8)
        for x, y in self.extractUncovered(inner_positions):
            if 0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]:
                grid[y, x] = 1
        return grid
