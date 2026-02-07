# InnerDeployment/GeneticAlgorithm/initializer.py
from __future__ import annotations

import random
from typing import List, Tuple, Optional, Union

import numpy as np
from SensorModule.Sensor import Sensor
from .utils import to_int_pairs

Gene = Tuple[int, int]
Chromosome = List[Gene]
Generation = List[Chromosome]
MapType = Union[np.ndarray, List[List[int]]]


def initialize_population(
    *,
    input_map: MapType,
    corner_positions: Chromosome,
    coverage: int,
    population_size: int,
    min_sensors: int,
    max_sensors: int,
    seed: Optional[int] = None,
) -> Generation:
    if seed is not None:
        random.seed(seed)

    arr = np.asarray(input_map)
    mask = (arr > 0).astype(np.uint8)  # Sensor 입력 안정화
    corners = to_int_pairs(corner_positions)

    sensor = Sensor(mask)

    # corner deploy: 시그니처 불일치 방지(가장 안전)
    for p in corners:
        try:
            sensor.deploy(sensor_position=p, coverage=int(coverage))
        except TypeError:
            try:
                sensor.deploy(p, coverage=int(coverage))
            except TypeError:
                sensor.deploy(p)

    uncovered = to_int_pairs(sensor.uncovered(roi_mask=(mask > 0), points=True))
    if not uncovered:
        raise ValueError("No uncovered points after deploying corner sensors.")

    low = max(1, int(min_sensors))
    high = min(max(low, int(max_sensors)), len(uncovered))

    return [
        random.sample(uncovered, k=random.randint(low, high))
        for _ in range(int(population_size))
    ]
