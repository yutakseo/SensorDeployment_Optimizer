import math
import torch
import torch.nn as nn
import numpy as np
from SensorModule.Sensor import Sensor


class Convolution(nn.Module):
    def __init__(self, MAP: np.ndarray):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    목표(임계값 없이 일반화):
    - coverage(반경)은 고정
    - installable 위에서 jobsite를 최대한 많이 덮기 (1순위)
    - coverage가 충분히 확보되면 한계효용이 감소하도록(포화) 만들어
      자연스럽게 센서 수를 최소화하도록 유도

    개선 반영 사항:
    1) cov_util 정규화 옵션(normalize_cov_util)
       - exp(-k*(1-c))는 c=0에서도 0이 아니므로(=exp(-k)),
         필요 시 0~1로 정규화해 스케일을 안정화할 수 있음.

    2) ordering_sensors()가 "fitness 개선이 더 이상 없으면" 즉시 중단
       - best_gain <= 0 이면 break
       - 이게 없으면 음수 gain 센서도 끝까지 다 붙어서 "센서 최소화"가 구조적으로 깨짐.

    3) ordering에서 n_after를 base_n+1이 아니라,
       "현재까지 선택된 내부 센서 수"에 기반해 정확히 계산
    """

    def __init__(
        self,
        jobsite_map: np.ndarray,
        corner_positions: list[tuple[int, int]],
        coverage,
        *,
        # ---------- coverage utility ----------
        k: float = 8.0,                # uncovered 감소 민감도 (6~12)
        normalize_cov_util: bool = True,  # NEW: cov_util 0~1 정규화 옵션
        # ---------- sensor cost ----------
        lam: float = 0.45,             # 센서 비용 가중치 (0.3~0.8)
        n_ref: float | None = None,
        n_ref_mode: str = "sqrt_jobsite",  # ["fixed", "sqrt_jobsite"]
        n_ref_fixed: float = 20.0,
        sensor_cost_mode: str = "log",     # ["linear", "log"]
        # ---------- ordering ----------
        stop_when_no_gain: bool = True,    # NEW: gain<=0이면 ordering 중단
        gain_eps: float = 0.0,             # NEW: 0 대신 아주 작은 양수로 설정 가능(예: 1e-9)
    ):
        self.map = np.array(jobsite_map, dtype=np.uint8)
        self.coverage = int(coverage)
        self.corners = [tuple(map(int, p)) for p in corner_positions]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.map_bool = torch.as_tensor(self.map > 0, dtype=torch.bool, device=self.device)
        self.map_sum = int(self.map_bool.sum().item())

        self._single_mask_cache: dict[tuple[int, int], torch.Tensor] = {}
        self.corner_mask = self._deploy_and_get_mask(self.corners)

        # (옵션) convolution 기반 맵
        self.model = Convolution(self.map.astype(np.float16))
        with torch.no_grad():
            self.fitness_map = self.model(self.map.astype(np.float16)).detach()

        self.k = float(k)
        self.normalize_cov_util = bool(normalize_cov_util)

        self.lam = float(lam)
        self.sensor_cost_mode = str(sensor_cost_mode)

        self.stop_when_no_gain = bool(stop_when_no_gain)
        self.gain_eps = float(gain_eps)

        if n_ref is not None:
            self.n_ref = float(n_ref)
        else:
            if n_ref_mode == "fixed":
                self.n_ref = float(n_ref_fixed)
            elif n_ref_mode == "sqrt_jobsite":
                self.n_ref = float(max(10.0, math.sqrt(max(1, self.map_sum))))
            else:
                raise ValueError(f"Unknown n_ref_mode: {n_ref_mode}")

        if self.n_ref <= 0:
            raise ValueError(f"n_ref must be positive. Got {self.n_ref}")

        if self.sensor_cost_mode not in ("linear", "log"):
            raise ValueError(f"sensor_cost_mode must be one of ['linear','log']. Got {self.sensor_cost_mode}")

    # -------------------------
    # internal
    # -------------------------
    def _get_single_mask_cached(self, pos: tuple[int, int]) -> torch.Tensor:
        pos = (int(pos[0]), int(pos[1]))
        cached = self._single_mask_cache.get(pos, None)
        if cached is not None:
            return cached

        sensor = Sensor(self.map)
        sensor.deploy(sensor_position=pos, coverage=self.coverage)
        m = sensor.extract_only_sensor()
        mask_bool = torch.as_tensor(m > 0, dtype=torch.bool, device=self.device)

        self._single_mask_cache[pos] = mask_bool
        return mask_bool

    def _deploy_and_get_mask(self, sensor_positions: list[tuple[int, int]]) -> torch.Tensor:
        if not sensor_positions:
            return torch.zeros_like(self.map_bool, dtype=torch.bool)

        acc = None
        for (x, y) in sensor_positions:
            m = self._get_single_mask_cached((x, y))
            acc = m if acc is None else (acc | m)
        return acc

    def _coverage_pct_from_mask(self, mask_bool: torch.Tensor) -> float:
        if self.map_sum <= 0:
            return 0.0
        covered = (self.map_bool & mask_bool).sum().item()
        return float(100.0 * float(covered) / float(self.map_sum))

    def _cov_util(self, coverage_pct: float) -> float:
        """
        uncovered ratio 기반 utility:
            raw = exp(-k * (1-c)), c in [0,1]
        normalize_cov_util=True이면:
            util = (raw - exp(-k)) / (1 - exp(-k))  -> 0~1
        """
        c = max(0.0, min(1.0, coverage_pct / 100.0))
        u = 1.0 - c
        raw = math.exp(-self.k * u)

        if not self.normalize_cov_util:
            return float(raw)

        lo = math.exp(-self.k)  # c=0
        hi = 1.0                # c=1
        # 수치 안전
        if hi - lo <= 1e-12:
            return 0.0
        return float((raw - lo) / (hi - lo))

    def _sensor_cost(self, total_sensors: int) -> float:
        n = float(max(0, int(total_sensors)))
        if self.sensor_cost_mode == "linear":
            return float(self.lam * (n / self.n_ref))
        # log-normalized
        return float(self.lam * (math.log1p(n) / math.log1p(self.n_ref)))

    def _fitness_from_cov_and_n(self, coverage_pct: float, total_sensors: int) -> float:
        return float(self._cov_util(coverage_pct) - self._sensor_cost(total_sensors))

    # -------------------------
    # public
    # -------------------------
    def fitness_score(self, inner_positions: list[tuple[int, int]]) -> float:
        inner = [tuple(map(int, p)) for p in inner_positions]

        mask = self.corner_mask
        for p in inner:
            mask = mask | self._get_single_mask_cached(p)

        coverage_pct = self._coverage_pct_from_mask(mask)
        total_sensors = len(self.corners) + len(inner)

        return self._fitness_from_cov_and_n(coverage_pct, total_sensors)

    def evaluate(self, inner_positions: list[tuple[int, int]]) -> tuple[float, float, int]:
        inner = [tuple(map(int, p)) for p in inner_positions]

        mask = self.corner_mask
        for p in inner:
            mask = mask | self._get_single_mask_cached(p)

        coverage_pct = self._coverage_pct_from_mask(mask)
        total_sensors = len(self.corners) + len(inner)

        fitness = self._fitness_from_cov_and_n(coverage_pct, total_sensors)
        return float(fitness), float(coverage_pct), int(total_sensors)

    def rank_single_sensor(self, sensor_points: list[tuple[int, int]]) -> list:
        ranking = []
        with torch.no_grad():
            fm = self.fitness_map
            for pos in [tuple(map(int, p)) for p in sensor_points]:
                mask = self._get_single_mask_cached(pos)
                score = (fm[0, 0] * mask.to(dtype=fm.dtype)).sum().item()
                ranking.append((pos, float(score)))
        ranking.sort(key=lambda x: x[1], reverse=True)
        return ranking

    def ordering_sensors(self, chromosome: list[tuple[int, int]], return_score: bool = True):
        """
        fitness gain 기반 greedy ordering + (중요) 더 이상 이득 없으면 중단

        반환:
        - return_score=True : [(pos, gain, cov_after), ...]
        - return_score=False: [pos, pos, ...] (prefix만)
        """
        remaining = [tuple(map(int, p)) for p in chromosome]
        ordered = []

        base_mask = self.corner_mask
        base_cov = self._coverage_pct_from_mask(base_mask)

        base_n = len(self.corners)  # corners 포함 센서 수
        base_fit = self._fitness_from_cov_and_n(base_cov, base_n)

        while remaining:
            best_pos = None
            best_gain = -1e18
            best_fit = None
            best_cov = None
            best_mask = None

            # 현재까지 선택된 내부 센서 수 = len(ordered)
            cur_total_n = len(self.corners) + len(ordered)

            for cand in remaining:
                cand_mask = self._get_single_mask_cached(cand)
                merged = base_mask | cand_mask

                cov_after = self._coverage_pct_from_mask(merged)
                n_after = cur_total_n + 1  # ★정확한 total sensors after adding this cand
                fit_after = self._fitness_from_cov_and_n(cov_after, n_after)

                gain = fit_after - base_fit
                if gain > best_gain:
                    best_pos = cand
                    best_gain = gain
                    best_fit = fit_after
                    best_cov = cov_after
                    best_mask = merged

            # ★ 핵심: 더 이상 fitness가 좋아지지 않으면 stop (센서 최소화 유도)
            if self.stop_when_no_gain and best_gain <= self.gain_eps:
                break

            ordered.append((best_pos, float(best_gain), float(best_cov)))

            base_mask = best_mask
            base_cov = best_cov
            base_fit = best_fit
            remaining.remove(best_pos)

        return ordered if return_score else [p for p, _, _ in ordered]

    def uncovered_map(self, inner_positions: list[tuple[int, int]]) -> np.ndarray:
        inner = [tuple(map(int, p)) for p in inner_positions]
        mask = self.corner_mask
        for p in inner:
            mask = mask | self._get_single_mask_cached(p)

        uncovered_bool = self.map_bool & (~mask)
        return uncovered_bool.detach().cpu().numpy().astype(np.uint8)
