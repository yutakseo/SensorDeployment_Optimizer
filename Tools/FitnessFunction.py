import torch
import torch.nn as nn
import numpy as np
from .Sensor_cuda import Sensor

class Convolution(nn.Module):
    def __init__(self, MAP):
        super(Convolution, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Base_map = torch.tensor(np.array(MAP)).float().to(self.device)

        self.conv3 = nn.Conv2d(1, 1, 3, padding=1).to(self.device)
        self.conv5 = nn.Conv2d(1, 1, 5, padding=2).to(self.device)
        self.conv7 = nn.Conv2d(1, 1, 7, padding=3).to(self.device)
        self.conv9 = nn.Conv2d(1, 1, 9, padding=4).to(self.device)

        with torch.no_grad():
            for conv, k in zip([self.conv3, self.conv5, self.conv7, self.conv9], [3, 5, 7, 9]):
                conv.weight.fill_(1 / (k * k))
                if conv.bias is not None:
                    conv.bias.zero_()

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)

        if isinstance(x, torch.Tensor):
            if x.ndim == 2:
                x = x.unsqueeze(0).unsqueeze(0)
            elif x.ndim == 3:
                x = x.unsqueeze(0)

        x = x.to(self.device)
        out = (self.conv3(x) + self.conv5(x) + self.conv7(x) + self.conv9(x)) / 4.0
        return out * self.Base_map

class F


def rankSensors(MAP, sensor_points, coverage=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNFitness(MAP)
    with torch.no_grad():
        fitness_map = model(np.array(MAP))

    ranking = []
    for pos in sensor_points:
        sensor = Sensor(MAP)
        sensor.deploy(sensor_position=(int(pos[0]), int(pos[1])), coverage=coverage)
        sensor_map = sensor.extract_only_sensor()
        sensor_tensor = torch.tensor(sensor_map, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        score = (fitness_map * sensor_tensor).sum().item()
        ranking.append((pos, score))

    ranking.sort(key=lambda x: x[1], reverse=True)
    return ranking



def fitnessFunc(MAP, sensor_list: list, coverage: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNFitness(MAP)
    tensor_map = torch.tensor(np.array(MAP), dtype=torch.float32, device=device)
    map_sum = tensor_map.sum().item()

    sensor = Sensor(MAP)
    for pos in sensor_list:
        sensor.deploy(sensor_position=(int(pos[0]), int(pos[1])), coverage=coverage)
    sensor_map = sensor.extract_only_sensor()
    sensor_tensor = torch.tensor(sensor_map, dtype=torch.float32, device=device)

    # 바이너리화
    sensor_tensor = (sensor_tensor > 0).float()

    # 커버 안 된 위치 계산 (1 - 센서 커버) * 설치가능
    uncovered = (tensor_map * (1 - sensor_tensor)).sum().item()
    covered = map_sum - uncovered

    fitness_score = covered / map_sum if map_sum > 0 else 0.0
    return fitness_score * 100



def extractUncovered(MAP, sensor_list: list, coverage: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor_map = torch.tensor(np.array(MAP), dtype=torch.float32, device=device)
    sensor = Sensor(MAP)
    # 모든 센서 배치
    for pos in sensor_list:
        sensor.deploy(sensor_position=(int(pos[0]), int(pos[1])), coverage=coverage)
    sensor_map = sensor.extract_only_sensor()
    sensor_tensor = torch.tensor(sensor_map, dtype=torch.float32, device=device)
    # 아직 커버되지 않은 위치: tensor_map은 배치 가능 영역이 1, 아니면 0
    # sensor_tensor는 센서가 커버한 곳은 1, 아니면 0
    uncovered = (tensor_map * (1 - sensor_tensor)).cpu().numpy()
    # 좌표 추출 (값이 1인 곳 = 설치 가능하지만 아직 커버 안 된 곳)
    uncovered_positions = list(map(tuple, np.argwhere(uncovered == 1)))
    
    return uncovered_positions
