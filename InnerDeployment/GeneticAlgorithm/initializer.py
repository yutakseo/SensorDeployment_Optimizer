# InnerDeployment/GeneticAlgorithm/initializer.py
from __future__ import annotations

import random
from typing import List, Tuple, Sequence, Optional, Union

import numpy as np

from SensorModule.Sensor import Sensor  # 프로젝트 경로 기준

XY = Tuple[int, int]
Chromosome = List[XY]
Population = List[Chromosome]

MapLike = Union[np.ndarray, List[List[int]], List[List[float]]]


def initialize_population(
    *,
    input_map: MapLike,
    corner_positions: Sequence[XY],
    coverage: int,
    population_size: int,
    min_sensors: int,
    max_sensors: int,
    seed: Optional[int] = None,
) -> Population:
    """
    초기 population 생성 (initializer 내부에서 uncovered 영역 추출까지 수행)

    Args:
        input_map        : installable_layer(0/1) 또는 ROI 마스크 (2D)
        corner_positions : 최외곽 센서 설치 좌표 리스트 [(x,y), ...]
        coverage         : 센서 커버리지(프로젝트 기준 meters; Sensor.deploy 내부에서 /5 처리)
        population_size  : 한 세대 염색체 개수
        min_sensors      : 염색체 당 유전자 최소 개수
        max_sensors      : 염색체 당 유전자 최대 개수
        seed             : 랜덤 시드

    Returns:
        population: List[chromosome]
            chromosome: List[(x,y)]
    """
    if seed is not None:
        random.seed(seed)

    # 1) Sensor 준비 + corner deploy
    sensor = Sensor(input_map)
    sensor.deploy(corner_positions, coverage=coverage)

    # 2) corner로 커버되지 않은 좌표 풀 추출
    #    - ROI는 input_map(installable_layer)을 그대로 넣는게 가장 직관적
    uncovered_points: List[XY] = sensor.uncovered(
        roi_mask=input_map,
        points=True,
    )

    if len(uncovered_points) == 0:
        raise ValueError("No uncovered points after deploying corner sensors.")

    # 3) 유전자 개수 범위 보정
    min_k = max(1, int(min_sensors))
    max_k = max(min_k, int(max_sensors))
    max_k = min(max_k, len(uncovered_points))

    # 4) population 구성
    population: Population = []
    for _ in range(int(population_size)):
        k = random.randint(min_k, max_k)
        chromo = random.sample(uncovered_points, k=k)  # 중복 없는 샘플
        population.append(chromo)

    return population



