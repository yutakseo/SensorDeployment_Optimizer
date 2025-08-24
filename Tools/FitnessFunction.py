import torch
import torch.nn as nn
import numpy as np
from .SensorModule import Sensor


class Convolution(nn.Module):
    def __init__(self, MAP: np.ndarray):
        super(Convolution, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Base_map = torch.as_tensor(MAP, dtype=torch.float32, device=self.device)

        # 사용할 커널 크기 리스트
        kernel_sizes = [3, 5, 7, 9, 11, 13, 15]

        # 평균 필터 Conv2d 레이어들을 ModuleList로 관리
        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=k,
                padding=k // 2,
                bias=False,
                padding_mode="replicate"  # 경계 처리 개선
            ).to(self.device) for k in kernel_sizes
        ])

        # 평균 필터로 초기화
        with torch.no_grad():
            for conv, k in zip(self.convs, kernel_sizes):
                conv.weight.fill_(1.0 / (k * k))
                conv.weight.requires_grad_(False)

    def forward(self, x):
        if isinstance(x, list):
            x = np.array(x, dtype=np.float32)
        if isinstance(x, np.ndarray):
            x = torch.as_tensor(x, dtype=torch.float32)

        if x.ndim == 2:   # [H,W]
            x = x.unsqueeze(0).unsqueeze(0)   # → [1,1,H,W]
        elif x.ndim == 3: # [C,H,W]
            x = x.unsqueeze(0)                # → [1,C,H,W]

        x = x.to(self.device)

        # 모든 필터의 출력을 평균
        outs = [conv(x) for conv in self.convs]
        out = sum(outs) / len(outs)

        # 설치 가능 영역만 반영
        return out * self.Base_map.unsqueeze(0).unsqueeze(0)


class SensorEvaluator:
    def __init__(self, map: np.ndarray, corner_points: list, sensor_points: list, coverage: int = 45) -> None:
        self.map = np.array(map, dtype=np.float32)
        self.coverage = coverage
        self.corners = corner_points
        self.sensors = sensor_points

        self.model = Convolution(self.map)

        # 실행 후 값이 채워질 변수들
        self.base_map: np.ndarray = None
        self.sorted_sensors: list = []
        self.fitness_score: float = 0.0
        self.uncovered_area: np.ndarray = None

    def _activation_map_(self, as_numpy: bool = True):
        with torch.no_grad():
            act = self.model.forward(self.map)
        if as_numpy:
            arr = act.squeeze().detach().cpu().numpy()
            self.base_map = arr                # ✅ base_map 저장
            return arr
        self.base_map = act.detach()
        return self.base_map

    def rankSensors(self, MAP: np.ndarray, sensor_points: list, coverage: int = 45) -> list:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            fitness_map = self.model(MAP).detach()

        ranking = []
        shifted_sensor_points = np.array(sensor_points)
        for pos in shifted_sensor_points:
            sensor = Sensor(MAP)
            sensor.deploy(sensor_position=(int(pos[0]), int(pos[1])), coverage=coverage)
            sensor_map = sensor.extract_only_sensor()
            sensor_tensor = torch.as_tensor(sensor_map, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            score = (fitness_map * (sensor_tensor > 0)).sum().item()
            ranking.append((pos, score))

        ranking.sort(key=lambda x: x[1], reverse=True)
        self.sorted_sensors = ranking         # ✅ sorted_sensors 저장
        return ranking

    def fitnessFunc(self, MAP, sensor_list: list, coverage: int) -> float:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tensor_map = torch.as_tensor(MAP, dtype=torch.float32, device=device)
        map_sum = tensor_map.sum().item()

        sensor = Sensor(MAP)
        for pos in sensor_list:
            sensor.deploy(sensor_position=(int(pos[0]), int(pos[1])), coverage=coverage)
        sensor_map = sensor.extract_only_sensor()
        sensor_tensor = torch.as_tensor(sensor_map, dtype=torch.float32, device=device)

        # 바이너리화
        sensor_tensor = (sensor_tensor > 0).float()

        uncovered = (tensor_map * (1 - sensor_tensor)).sum().item()
        covered = map_sum - uncovered

        fitness_score = covered / map_sum if map_sum > 0 else 0.0
        self.fitness_score = fitness_score    # ✅ fitness_score 저장
        return fitness_score * 100

    def extractUncovered(self, MAP, sensor_list: list, coverage: int) -> list:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tensor_map = torch.as_tensor(MAP, dtype=torch.float32, device=device)
        sensor = Sensor(MAP)
        for pos in sensor_list:
            sensor.deploy(sensor_position=(int(pos[0]), int(pos[1])), coverage=coverage)
        sensor_map = sensor.extract_only_sensor()
        sensor_tensor = torch.as_tensor(sensor_map, dtype=torch.float32, device=device)

        uncovered = (tensor_map * (1 - (sensor_tensor > 0).float())).cpu().numpy()
        uncovered_positions = list(map(tuple, np.argwhere(uncovered == 1)))

        self.uncovered_area = uncovered_positions   # ✅ uncovered_area 저장
        return uncovered_positions

    def uncoveredMAP(self, MAP: np.ndarray, sensor_list: list, coverage: int) -> np.ndarray:
        """
        설치 가능한 영역 중 센서로 커버되지 않은 셀을 2D numpy array로 반환
        - 1: uncovered (설치 가능하지만 커버 안 된 셀)
        - 0: covered 또는 설치 불가 셀
        """
        uncovered_positions = self.extractUncovered(MAP=MAP, sensor_list=sensor_list, coverage=coverage)
        grid = np.zeros_like(MAP, dtype=np.uint8)
        for (y, x) in uncovered_positions:
            grid[y, x] = 1
        return grid
