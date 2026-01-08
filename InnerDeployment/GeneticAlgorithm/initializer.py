# InnerDeployment/GeneticAlgorithm/initializer.py
import random
from typing import List, Tuple, Optional

from SensorModule.Sensor import Sensor

XY = Tuple[int, int]


def initialize_population(
    input_map,
    population_size: int,
    corner_positions: List[XY],
    coverage: int,
    min_sensors: int = 1,
    max_sensors: Optional[int] = None,
) -> List[List[XY]]:
    """
    1) corner_positions에 센서를 먼저 배치
    2) uncovered(points=True)로 미커버 지점 후보를 추출
    3) 후보 중 랜덤하게 뽑아 population 생성

    염색체 구조:
      - chromosome = [(x1,y1), (x2,y2), ...]  (추가 센서 위치들)
      - corner 센서들은 '고정'이므로 chromosome에는 포함하지 않음
    """
    # 1) corner 센서 배치
    sensor = Sensor(input_map)
    sensor.deploy(corner_positions, coverage=coverage)

    # 2) 미커버 후보점(=추가 센서 설치 후보)
    uncovered_points: List[XY] = sensor.uncovered(points=True)

    if not uncovered_points:
        # 이미 corner로 모두 커버된 경우
        return [[] for _ in range(population_size)]

    # 3) max_sensors 기본값
    if max_sensors is None:
        # 이론상 후보점 개수까지 가능하지만, 너무 커지면 GA가 비효율적
        # 우선 후보점 개수로 상한
        max_sensors = len(uncovered_points)

    max_sensors = max(min_sensors, max_sensors)

    population: List[List[XY]] = []

    for _ in range(population_size):
        gene_len = random.randint(min_sensors, max_sensors)

        # 후보점이 적을 수 있으니 replace 허용(중복 방지 원하면 random.sample로 바꾸면 됨)
        if gene_len <= len(uncovered_points):
            chrom = random.sample(uncovered_points, gene_len)
        else:
            chrom = [random.choice(uncovered_points) for _ in range(gene_len)]

        population.append(chrom)

    return population
