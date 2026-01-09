import random
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
    """
    (복구된 원리 유지) Sensor 기반 커버리지 평가 + Greedy ordering

    - fitness_score: coverage% (0~100) 반환 (비용항 없음)  ✅원래 구현 유지
    - ordering_sensors: corners 선배치 후 greedy marginal-gain 정렬 ✅원래 구현 유지

    최적화/안전성 추가:
    - jobsite_map을 0/1 마스크로 정규화 (0/255, 2/3 등 들어와도 안전)
    - corners mask 캐시
    - 단일 센서 mask 캐시
    - (선택) ordering 후보 샘플링
    - (선택) gain 미미 시 조기 종료
    """

    def __init__(
        self,
        jobsite_map,
        corner_positions: list[tuple[int, int]],
        coverage,
        *,
        # ---- perf knobs (optional) ----
        cache_single_masks: bool = True,
        candidate_sample_k: int | None = None,  # 예: 20. None이면 전수조사(원래 방식)
        min_gain_stop: float | None = None,     # 예: 0.02 (coverage% 단위). None이면 중단 안함
    ):
        # 1) jobsite_map 정규화: "작업영역이면 1" 마스크로 통일
        #    - 0/1, 0/255, 2/3 등 어떤 입력이 와도 >0이면 1로 처리
        js = np.asarray(jobsite_map)
        self.map = (js > 0).astype(np.float16)  # ✅중요: 면적 기반 coverage가 되도록 고정

        self.coverage = int(coverage)
        self.corners = [tuple(map(int, p)) for p in corner_positions]

        # perf knobs
        self.cache_single_masks = bool(cache_single_masks)
        self.candidate_sample_k = candidate_sample_k
        self.min_gain_stop = None if min_gain_stop is None else float(min_gain_stop)

        # device / map tensor 캐시
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tensor_map = torch.as_tensor(self.map, dtype=torch.float16, device=self.device)
        self.map_sum = float(self.tensor_map.sum().item())

        # ---- caches ----
        self._single_mask_cache: dict[tuple[int, int, int], torch.Tensor] = {}
        self._corners_mask_cache: torch.Tensor | None = None

        # (기존 유지: activation_map 계산)
        self.model = Convolution(self.map)
        with torch.no_grad():
            self.activation_map = self.model(self.map).detach().cpu().numpy()

    # -------------------------
    # internal (optimized)
    # -------------------------
    def _deploy_and_get_mask(self, sensor_positions: list[tuple[int, int]]) -> torch.Tensor:
        """센서 배치 후 binary coverage mask 반환 (H,W) float32 {0,1}"""

        # corners mask 캐시 (동일 corners 리스트 요청 시)
        if sensor_positions == self.corners and self._corners_mask_cache is not None:
            return self._corners_mask_cache

        # 단일 센서 캐시
        if self.cache_single_masks and len(sensor_positions) == 1:
            x, y = map(int, sensor_positions[0])
            key = (x, y, int(self.coverage))
            cached = self._single_mask_cache.get(key, None)
            if cached is not None:
                return cached

        sensor = Sensor(self.map)
        for (x, y) in sensor_positions:
            sensor.deploy(sensor_position=(int(x), int(y)), coverage=self.coverage)

        mask = torch.as_tensor(sensor.extract_only_sensor(), dtype=torch.float16, device=self.device)
        mask01 = (mask > 0).float()

        # 캐시 저장
        if sensor_positions == self.corners:
            self._corners_mask_cache = mask01
        elif self.cache_single_masks and len(sensor_positions) == 1:
            self._single_mask_cache[key] = mask01

        return mask01

    def _fitness_from_mask(self, mask01: torch.Tensor) -> float:
        """coverage mask(0/1)에서 coverage% 계산 (0~100)"""
        if self.map_sum <= 0:
            return 0.0
        covered = (self.tensor_map * mask01).sum().item()
        return float(100.0 * covered / self.map_sum)

    def _fitness_given(self, sensor_positions: list[tuple[int, int]]) -> float:
        """corner+inner 전체를 배치해서 coverage% (0~100)"""
        mask = self._deploy_and_get_mask(sensor_positions)
        return self._fitness_from_mask(mask)

    def _extract_uncovered(self, sensor_positions: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """uncovered 좌표 리스트: (y,x) 반환 (argwhere 규약 유지)"""
        mask = self._deploy_and_get_mask(sensor_positions)
        uncovered = (self.tensor_map * (1 - mask)).cpu().numpy()
        return list(map(tuple, np.argwhere(uncovered == 1)))

    # -------------------------
    # public (same behavior)
    # -------------------------
    def fitness_score(self, inner_positions: list[tuple[int, int]]) -> float:
        """✅원래 구현 유지: coverage%만 반환"""
        inner = [tuple(map(int, p)) for p in inner_positions]
        return self._fitness_given(self.corners + inner)

    def evaluate(self, inner_positions: list[tuple[int, int]]):
        """logging용: (fitness=coverage%, coverage%, total_sensors)"""
        inner = [tuple(map(int, p)) for p in inner_positions]
        cov = self._fitness_given(self.corners + inner)
        total = len(self.corners) + len(inner)
        return float(cov), float(cov), int(total)

    def rank_single_sensor(self, sensor_points: list[tuple[int, int]]) -> list:
        """단일 센서를 단독으로 설치했을 때의 잠재력 랭킹 (interaction 미고려)"""
        with torch.no_grad():
            fitness_map = self.model(self.map).detach()  # [1,1,H,W]

        ranking = []
        for pos in [tuple(map(int, p)) for p in sensor_points]:
            mask = self._deploy_and_get_mask([pos]).unsqueeze(0).unsqueeze(0)
            score = (fitness_map * mask).sum().item()
            ranking.append((pos, float(score)))

        ranking.sort(key=lambda x: x[1], reverse=True)
        return ranking

    def ordering_sensors(self, chromosome: list[tuple[int, int]], return_score: bool = True):
        """
        corners 선설치 후, chromosome 내부 센서들을 greedy marginal-gain 방식으로 정렬
        - base_mask 누적 OR 유지 (원래 방식)
        - 최적화: 단일센서 mask 캐시 / corners mask 캐시
        - (옵션) 후보 샘플링 / gain 미미 시 조기 종료
        """
        remaining = [tuple(map(int, p)) for p in chromosome]
        ordered = []

        base_mask = self._deploy_and_get_mask(self.corners)  # cached
        base_fit = self._fitness_from_mask(base_mask)

        while remaining:
            # 후보 샘플링 (원래대로 전수조사하려면 candidate_sample_k=None)
            if self.candidate_sample_k is None or self.candidate_sample_k >= len(remaining):
                candidates = remaining
            else:
                candidates = random.sample(remaining, int(self.candidate_sample_k))

            best_pos, best_gain, best_fit, best_mask = None, -1e18, None, None

            for cand in candidates:
                cand_mask = self._deploy_and_get_mask([cand])  # cached if repeated
                merged_mask = torch.clamp(base_mask + cand_mask, 0, 1)  # OR
                fit_after = self._fitness_from_mask(merged_mask)
                gain = fit_after - base_fit

                if gain > best_gain:
                    best_pos, best_gain, best_fit, best_mask = cand, gain, fit_after, merged_mask

            if best_pos is None:
                break

            ordered.append((best_pos, float(best_gain), float(best_fit)))
            remaining.remove(best_pos)

            # update base
            base_mask = best_mask
            base_fit = best_fit

            # gain이 너무 작으면 중단 (옵션)
            if self.min_gain_stop is not None and float(best_gain) < float(self.min_gain_stop):
                break

        return ordered if return_score else [p for (p, _, _) in ordered]

    def uncovered_map(self, inner_positions: list[tuple[int, int]]) -> np.ndarray:
        uncovered = self._extract_uncovered(self.corners + [tuple(map(int, p)) for p in inner_positions])
        grid = np.zeros_like(self.map, dtype=np.uint8)
        for y, x in uncovered:
            grid[y, x] = 1
        return grid
