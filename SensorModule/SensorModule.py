import torch

class Sensor:
    FIXED_STRENGTH = 10.0
    def __init__(self, MAP, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        if not isinstance(MAP, torch.Tensor):
            map_tensor = torch.tensor(MAP, dtype=torch.float32, device=self.device)
        else:
            map_tensor = MAP.to(dtype=torch.float32, device=self.device)

        self.map_tensor = map_tensor
        self.MAP = map_tensor.clone()
        self.sensor_log = set()
        self.radius = None


    def _create_circle(self, radius: int) -> torch.Tensor:
        diameter = 2 * radius + 1
        y, x = torch.meshgrid(torch.arange(diameter), torch.arange(diameter), indexing='ij')
        center = radius
        dist = torch.sqrt((x - center)**2 + (y - center)**2)
        return (dist <= radius).float().to(self.device)


    def _apply(self, sensor_position, coverage, strength):
        x, y = sensor_position
        H, W = self.MAP.shape
        D = 2 * coverage + 1

        # 센터를 기준으로 한 원형 커버리지 생성
        if (coverage != self.radius) or (self.circle is None):
            self.circle = self._create_circle(coverage)  # 필요시 갱신
            self.radius = coverage

        circle = self.circle  # shape: [D, D]

        # 맵 영역에서 유효한 인덱스 슬라이싱 계산
        y1 = max(0, y - coverage)
        y2 = min(H, y + coverage + 1)
        x1 = max(0, x - coverage)
        x2 = min(W, x + coverage + 1)

        # circle 상 대응하는 부분 (out-of-bounds 영역 제거)
        cy1 = y1 - (y - coverage)  # 원 내부에서 시작 index
        cy2 = D - ((y + coverage + 1) - y2)
        cx1 = x1 - (x - coverage)
        cx2 = D - ((x + coverage + 1) - x2)

        # 실제 연산
        self.MAP[y1:y2, x1:x2] += strength * circle[cy1:cy2, cx1:cx2]


    def deploy(self, sensor_position:tuple, coverage:int=45):
        x, y = sensor_position
        coverage = int(coverage/5)
        if any(key[0] == x and key[1] == y for key in self.sensor_log):
            #print(f"[Info] Sensor already deployed at {sensor_position}, skipped.")
            return self.MAP
        
        key = (x, y, coverage)
        self.sensor_log.add(key)
        self._apply(sensor_position, coverage, self.FIXED_STRENGTH)
        return self.MAP


    def remove(self, sensor_position:tuple):
        x, y = sensor_position
        matching_keys = [key for key in self.sensor_log if key[0] == x and key[1] == y]
        if not matching_keys:
            print(f"No sensor found at position {sensor_position} to remove")
            return self.MAP
        
        key = matching_keys[0]
        _, _, coverage = key  
        self.sensor_log.remove(key)
        self._apply(sensor_position, coverage, -self.FIXED_STRENGTH)
        self.MAP = torch.clamp(self.MAP, min=0.0)
        return self.MAP

    def extract_only_sensor(self):
        mask = (self.MAP > self.FIXED_STRENGTH).float()  # 커버된 영역만 1
        return self.MAP * mask
