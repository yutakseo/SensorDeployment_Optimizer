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
            nn.Conv2d(1, 1, k, padding=k // 2, bias=False, padding_mode="replicate")
              .to(self.device)
              .half()
            for k in kernel_sizes
        ])

        with torch.no_grad():
            for conv, k in zip(self.convs, kernel_sizes):
                conv.weight.fill_(1.0 / (k * k))
                conv.weight.requires_grad_(False)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.as_tensor(x, dtype=torch.float16)
        else:
            x = x.to(dtype=torch.float16)

        if x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)

        x = x.to(self.device)
        out = sum(conv(x) for conv in self.convs) / len(self.convs)

        return out * self.base_map.unsqueeze(0).unsqueeze(0)


class FitnessFunc:
    def __init__(self, jobsite_map: np.ndarray, corner_positions: list[tuple[int, int]], coverage):
        self.map = np.array(jobsite_map, dtype=np.uint8)  # 0/1 가정이면 uint8이 유리
        self.coverage = int(coverage)
        self.corners = [tuple(map(int, p)) for p in corner_positions]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # map을 bool로 캐시 (AND/sum 최적화)
        self.map_bool = torch.as_tensor(self.map > 0, dtype=torch.bool, device=self.device)
        self.map_sum = int(self.map_bool.sum().item())

        # 단일 센서 마스크 캐시: (x,y) -> torch.bool (H,W)
        self._single_mask_cache: dict[tuple[int, int], torch.Tensor] = {}

        # corners mask를 한 번만 계산해서 캐시
        self.corner_mask = self._deploy_and_get_mask(self.corners)

        # (옵션) convolution 기반 맵은 현 코드에선 ordering에 사용 안하므로 유지/최소화
        self.model = Convolution(self.map.astype(np.float16))
        with torch.no_grad():
            self.fitness_map = self.model(self.map.astype(np.float16)).detach()  # [1,1,H,W]

    # -------------------------
    # 🔒 internal
    # -------------------------
    def _get_single_mask_cached(self, pos: tuple[int, int]) -> torch.Tensor:
        """
        단일 센서 커버 마스크를 (x,y)별로 1회만 생성/캐시.
        반환: torch.bool (H,W)
        """
        pos = (int(pos[0]), int(pos[1]))
        cached = self._single_mask_cache.get(pos, None)
        if cached is not None:
            return cached

        # NOTE: Sensor가 numpy 기반이면 여기만큼은 CPU 비용이 들어가지만,
        #       좌표당 1회만 수행되도록 캐싱한다.
        sensor = Sensor(self.map)  # map은 uint8(0/1)로 유지
        sensor.deploy(sensor_position=pos, coverage=self.coverage)

        m = sensor.extract_only_sensor()  # numpy (H,W), >0이면 커버
        mask_bool = torch.as_tensor(m > 0, dtype=torch.bool, device=self.device)

        self._single_mask_cache[pos] = mask_bool
        return mask_bool

    def _deploy_and_get_mask(self, sensor_positions: list[tuple[int, int]]) -> torch.Tensor:
        """
        여러 센서의 커버 마스크를 OR로 합성.
        반환: torch.bool (H,W)
        """
        if not sensor_positions:
            # 빈 배치면 all-false
            return torch.zeros_like(self.map_bool, dtype=torch.bool)

        # OR 누적
        acc = None
        for (x, y) in sensor_positions:
            m = self._get_single_mask_cached((x, y))
            acc = m if acc is None else (acc | m)

        return acc

    def _fitness_from_mask(self, mask_bool: torch.Tensor) -> float:
        """
        coverage mask(bool)에서 fitness 계산
        fitness = covered(jobsite) / total(jobsite) * 100
        """
        if self.map_sum <= 0:
            return 0.0
        covered = (self.map_bool & mask_bool).sum().item()
        return float(100.0 * float(covered) / float(self.map_sum))

    def _fitness_given(self, sensor_positions: list[tuple[int, int]]) -> float:
        mask = self._deploy_and_get_mask(sensor_positions)
        return self._fitness_from_mask(mask)

    # -------------------------
    # 🔓 public
    # -------------------------
    def fitness_score(self, inner_positions: list[tuple[int, int]]) -> float:
        inner = [tuple(map(int, p)) for p in inner_positions]
        # corners mask를 재사용하면 더 빠름: corners + inner OR
        mask = self.corner_mask
        for p in inner:
            mask = mask | self._get_single_mask_cached(p)
        return self._fitness_from_mask(mask)

    def rank_single_sensor(self, sensor_points: list[tuple[int, int]]) -> list:
        """
        단일 센서를 단독으로 설치했을 때의 잠재력 랭킹 (interaction 미고려)
        """
        ranking = []
        with torch.no_grad():
            fm = self.fitness_map  # [1,1,H,W] FP16

            for pos in [tuple(map(int, p)) for p in sensor_points]:
                mask = self._get_single_mask_cached(pos)  # (H,W) bool
                score = (fm[0, 0] * mask.to(dtype=fm.dtype)).sum().item()
                ranking.append((pos, float(score)))

        ranking.sort(key=lambda x: x[1], reverse=True)
        return ranking

    def ordering_sensors(self, chromosome: list[tuple[int, int]], return_score: bool = True):
        """
        corner 선설치 후, 염색체 내부 센서들을 greedy marginal-gain 방식으로 정렬

        핵심 최적화:
        - 후보 단일 마스크는 좌표별 1회만 생성(캐시)
        - base_mask는 bool OR로 누적
        - gain 계산은 fitness_from_mask 호출 (AND + sum)
        """
        remaining = [tuple(map(int, p)) for p in chromosome]
        ordered = []

        # base: corners mask/fitness (캐시 사용)
        base_mask = self.corner_mask
        base_fit = self._fitness_from_mask(base_mask)

        while remaining:
            best_pos = None
            best_gain = -1e18
            best_fit = None
            best_mask = None

            # NOTE: 여기서도 mask는 캐시에서 O(1)로 가져옴
            for cand in remaining:
                cand_mask = self._get_single_mask_cached(cand)
                merged = base_mask | cand_mask
                fit_after = self._fitness_from_mask(merged)
                gain = fit_after - base_fit

                if gain > best_gain:
                    best_pos = cand
                    best_gain = gain
                    best_fit = fit_after
                    best_mask = merged

            ordered.append((best_pos, float(best_gain), float(best_fit)))

            # update base
            base_mask = best_mask
            base_fit = best_fit
            remaining.remove(best_pos)

        return ordered if return_score else [p for p, _, _ in ordered]

    def uncovered_map(self, inner_positions: list[tuple[int, int]]) -> np.ndarray:
        """
        uncovered grid 반환 (H,W) uint8 with 1 for uncovered (jobsite==1 AND not covered)
        """
        inner = [tuple(map(int, p)) for p in inner_positions]
        mask = self.corner_mask
        for p in inner:
            mask = mask | self._get_single_mask_cached(p)

        # uncovered = jobsite AND not covered
        uncovered_bool = self.map_bool & (~mask)
        return uncovered_bool.detach().cpu().numpy().astype(np.uint8)
