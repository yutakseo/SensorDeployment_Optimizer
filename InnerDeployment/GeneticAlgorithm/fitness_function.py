import torch
import torch.nn as nn
import numpy as np
from SensorModule.Sensor import Sensor


class Convolution(nn.Module):
    def __init__(self, MAP: np.ndarray):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 입력/맵을 FP16으로 통일
        self.base_map = torch.as_tensor(MAP, dtype=torch.float16, device=self.device)

        kernel_sizes = [3, 5, 7, 9, 11, 13, 15]
        self.convs = nn.ModuleList([
            # Conv weight도 FP16으로 통일 (중요!)
            nn.Conv2d(1, 1, k, padding=k // 2, bias=False, padding_mode="replicate")
              .to(self.device)
              .half()
            for k in kernel_sizes
        ])

        # 평균 필터로 초기화 (FP16 weight에 채움)
        with torch.no_grad():
            for conv, k in zip(self.convs, kernel_sizes):
                conv.weight.fill_(1.0 / (k * k))
                conv.weight.requires_grad_(False)

    def forward(self, x):
        # numpy 입력
        if isinstance(x, np.ndarray):
            x = torch.as_tensor(x, dtype=torch.float16)
        else:
            # torch Tensor 입력 방어: dtype을 FP16으로 통일
            x = x.to(dtype=torch.float16)

        if x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)

        x = x.to(self.device)
        out = sum(conv(x) for conv in self.convs) / len(self.convs)

        return out * self.base_map.unsqueeze(0).unsqueeze(0)


class FitnessFunc:
    def __init__(self, jobsite_map: np.ndarray, corner_positions: list[tuple[int, int]], coverage):
        self.map = np.array(jobsite_map, dtype=np.float16)
        self.coverage = int(coverage)
        self.corners = [tuple(map(int, p)) for p in corner_positions]

        # device / map tensor를 미리 캐시 (속도 + 일관성)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tensor_map = torch.as_tensor(self.map, dtype=torch.float16, device=self.device)
        self.map_sum = float(self.tensor_map.sum().item())

        self.model = Convolution(self.map)
        with torch.no_grad():
            self.activation_map = self.model(self.map).detach().cpu().numpy()

    # -------------------------
    # 🔒 internal
    # -------------------------
    def _deploy_and_get_mask(self, sensor_positions: list[tuple[int, int]]) -> torch.Tensor:
        """센서 배치 후 binary coverage mask 반환 (H,W) float32 {0,1}"""
        sensor = Sensor(self.map)
        for (x, y) in sensor_positions:
            sensor.deploy(sensor_position=(int(x), int(y)), coverage=self.coverage)

        mask = torch.as_tensor(sensor.extract_only_sensor(), dtype=torch.float16, device=self.device)
        return (mask > 0).float()

    def _fitness_from_mask(self, mask01: torch.Tensor) -> float:
        """coverage mask(0/1)에서 바로 fitness 계산"""
        if self.map_sum <= 0:
            return 0.0
        covered = (self.tensor_map * mask01).sum().item()
        return float(100.0 * covered / self.map_sum)

    def _fitness_given(self, sensor_positions: list[tuple[int, int]]) -> float:
        """corner+inner 전체를 배치해서 fitness(0~100)"""
        mask = self._deploy_and_get_mask(sensor_positions)
        return self._fitness_from_mask(mask)

    def _extract_uncovered(self, sensor_positions: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """uncovered 좌표 리스트: (y,x) 반환 (argwhere 규약 유지)"""
        mask = self._deploy_and_get_mask(sensor_positions)
        uncovered = (self.tensor_map * (1 - mask)).cpu().numpy()
        return list(map(tuple, np.argwhere(uncovered == 1)))

    # -------------------------
    # 🔓 public
    # -------------------------
    def fitness_score(self, inner_positions: list[tuple[int, int]]) -> float:
        return self._fitness_given(self.corners + [tuple(map(int, p)) for p in inner_positions])

    def rank_single_sensor(self, sensor_points: list[tuple[int, int]]) -> list:
        """
        단일 센서를 단독으로 설치했을 때의 잠재력 랭킹 (interaction 미고려)
        """
        with torch.no_grad():
            fitness_map = self.model(self.map).detach()  # [1,1,H,W]

        ranking = []
        for pos in [tuple(map(int, p)) for p in sensor_points]:
            mask = self._deploy_and_get_mask([pos]).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
            score = (fitness_map * mask).sum().item()
            ranking.append((pos, float(score)))

        ranking.sort(key=lambda x: x[1], reverse=True)
        return ranking

    def ordering_sensors(self, chromosome: list[tuple[int, int]], return_score: bool = True):
        """
        corner 선설치 후, 염색체 내부 센서들을 greedy marginal-gain 방식으로 정렬
        - 개선: base mask를 누적하여 반복 재배치 비용 감소
        """
        remaining = [tuple(map(int, p)) for p in chromosome]
        selected: list[tuple[int, int]] = []
        ordered = []

        # base: corners mask/fitness
        base_mask = self._deploy_and_get_mask(self.corners)  # (H,W) 0/1
        base_fit = self._fitness_from_mask(base_mask)

        while remaining:
            best_pos, best_gain, best_fit, best_mask = None, -1e18, None, None

            for cand in remaining:
                cand_mask = self._deploy_and_get_mask([cand])
                merged_mask = torch.clamp(base_mask + cand_mask, 0, 1)  # OR
                fit_after = self._fitness_from_mask(merged_mask)
                gain = fit_after - base_fit

                if gain > best_gain:
                    best_pos, best_gain, best_fit, best_mask = cand, gain, fit_after, merged_mask

            selected.append(best_pos)
            remaining.remove(best_pos)
            ordered.append((best_pos, float(best_gain), float(best_fit)))

            # update base
            base_mask = best_mask
            base_fit = best_fit

        return ordered if return_score else [p for p, _, _ in ordered]

    def uncovered_map(self, inner_positions: list[tuple[int, int]]) -> np.ndarray:
        uncovered = self._extract_uncovered(self.corners + [tuple(map(int, p)) for p in inner_positions])
        grid = np.zeros_like(self.map, dtype=np.uint8)
        for y, x in uncovered:
            grid[y, x] = 1
        return grid
