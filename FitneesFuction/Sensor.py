import torch
import torch.nn.functional as F

class Sensor:
    def __init__(self, MAP):
        """
        :param MAP: 설치 가능 영역을 나타내는 2D 리스트 or numpy 배열 (0/1)
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        map_tensor = torch.tensor(MAP, dtype=torch.float32)
        self.map_tensor = map_tensor.to(self.device)
        self.cover_map = self.map_tensor.clone().float()  # 영향을 누적할 맵

    def create_circle(self, radius: int) -> torch.Tensor:
        diameter = 2 * radius + 1
        y, x = torch.meshgrid(torch.arange(diameter), torch.arange(diameter), indexing='ij')
        center = radius
        dist = torch.sqrt((x - center)**2 + (y - center)**2)
        circle = (dist <= radius).float().to(self.device)
        return circle

    def deploy(self, sensor_position: tuple, coverage: int, strength: float = 10.0):
        y, x = sensor_position
        H, W = self.map_tensor.shape
        radius = coverage
        circle = self.create_circle(radius)
        D = 2 * radius + 1

        # 실제 map 범위 계산
        y1, y2 = max(0, y - radius), min(H, y + radius + 1)
        x1, x2 = max(0, x - radius), min(W, x + radius + 1)

        # 마스크에서 사용할 영역 계산
        cy1, cy2 = max(0, radius - y), D - max(0, y + radius + 1 - H)
        cx1, cx2 = max(0, radius - x), D - max(0, x + radius + 1 - W)

        self.cover_map[y1:y2, x1:x2] += strength * circle[cy1:cy2, cx1:cx2]

    def result(self):
        return self.cover_map



from top import MAP
sensor = Sensor(MAP)  # 자동으로 GPU or CPU 선택
sensor.deploy((20, 30), coverage=45)
sensor.deploy((15, 10), coverage=45)

result = sensor.result().cpu().numpy()

import matplotlib.pyplot as plt
plt.imshow(result, cmap='hot')
plt.colorbar()
plt.title("Torch Sensor (Auto Device)")
plt.savefig("torch_auto_device_result.png")
