# SensorModule/Sensor.py
from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union, List, overload, Literal

import numpy as np
import torch

MaskLike = Union[
    torch.Tensor,
    np.ndarray,  # ✅ 추가
    Sequence[Sequence[Union[int, float]]],
]
PosLike = Union[Tuple[int, int], Sequence[Tuple[int, int]], torch.Tensor]
XY = Tuple[int, int]


class Sensor:
    """
    Public API (only 4 methods):
      - deploy(...)
      - remove(...)
      - covered(...)
      - uncovered(...)

    Design:
      - self.map_tensor: 원본 맵(초기 상태)
      - self.MAP       : 센서 배치에 따라 strength가 누적된 맵
      - 커버 여부는 (self.MAP > self.map_tensor + eps) 기준으로 판단
    """

    FIXED_STRENGTH = 10.0

    def __init__(
        self,
        MAP: MaskLike,
        device: Optional[Union[str, torch.device]] = None,
        *,
        mask_only: bool = False,
    ):
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.mask_only = bool(mask_only)

        if isinstance(MAP, torch.Tensor):
            map_tensor = MAP.to(device=self.device, dtype=torch.float16)
        else:
            map_tensor = torch.tensor(MAP, device=self.device, dtype=torch.float16)

        if map_tensor.ndim != 2:
            raise ValueError(f"MAP must be 2D (H,W). Got shape={tuple(map_tensor.shape)}")

        self.map_tensor = map_tensor
        self.MAP = map_tensor.clone()
        self.sensor_mask: Optional[torch.Tensor] = None
        if self.mask_only:
            self.sensor_mask = torch.zeros_like(map_tensor, dtype=torch.bool, device=self.device)

        self.radius: Optional[int] = None
        self.circle: Optional[torch.Tensor] = None
        self._offsets: Optional[torch.Tensor] = None

        # (x, y, cov_radius_in_cells)
        self.sensor_log: list[tuple[int, int, int]] = []

    # -------------------------
    # Internal helpers
    # -------------------------
    @staticmethod
    def _to_bool_mask(
        x: Optional[MaskLike],
        *,
        device: torch.device,
        fallback: torch.Tensor,
    ) -> torch.Tensor:
        """
        roi_mask:
          - None이면 fallback을 ROI로 사용
          - 아니면 0/1 or bool 마스크를 torch.bool로 변환
        """
        if x is None:
            roi = fallback.bool()
        else:
            if isinstance(x, torch.Tensor):
                t = x.to(device=device)
            else:
                t = torch.as_tensor(x, device=device)
            if t.ndim != 2:
                raise ValueError(f"mask must be 2D (H,W). Got shape={tuple(t.shape)}")
            roi = t.bool()
        return roi

    @staticmethod
    def _to_numpy_bool(mask_bool: torch.Tensor) -> np.ndarray:
        return mask_bool.detach().cpu().numpy().astype(bool)

    def _create_circle(self, radius: int) -> torch.Tensor:
        d = 2 * radius + 1
        y, x = torch.meshgrid(
            torch.arange(d, device=self.device),
            torch.arange(d, device=self.device),
            indexing="ij",
        )
        c = radius
        dist = torch.sqrt((x - c) ** 2 + (y - c) ** 2)
        return (dist <= radius).float()

    def _ensure_kernel(self, coverage_cells: int) -> None:
        if (coverage_cells != self.radius) or (self.circle is None):
            self.circle = self._create_circle(coverage_cells)
            self.radius = coverage_cells
            ys, xs = torch.where(self.circle > 0)
            self._offsets = torch.stack(
                [ys - coverage_cells, xs - coverage_cells], dim=1
            ).to(torch.long)

    # -------------------------
    # Public API
    # -------------------------
    @torch.no_grad()
    def deploy(self, sensor_position: PosLike, coverage: int = 45) -> torch.Tensor:
        cov = int(coverage / 5)
        self._ensure_kernel(cov)

        H, W = self.MAP.shape
        strength = float(self.FIXED_STRENGTH)

        if isinstance(sensor_position, torch.Tensor):
            pos = sensor_position.to(device=self.device, dtype=torch.long)
        elif isinstance(sensor_position, tuple) and len(sensor_position) == 2:
            pos = torch.tensor([sensor_position], device=self.device, dtype=torch.long)
        else:
            pos = torch.tensor(sensor_position, device=self.device, dtype=torch.long)

        if pos.ndim != 2 or pos.shape[1] != 2:
            raise ValueError(f"sensor_position must be (N,2). Got shape={tuple(pos.shape)}")

        if not self.mask_only:
            for x, y in pos.tolist():
                self.sensor_log.append((int(x), int(y), int(cov)))

        if self._offsets is None:
            raise RuntimeError("Kernel offsets are not initialized.")
        x0 = pos[:, 0]
        y0 = pos[:, 1]
        dy = self._offsets[:, 0]
        dx = self._offsets[:, 1]

        yy = y0[:, None] + dy[None, :]
        xx = x0[:, None] + dx[None, :]

        valid = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)

        lin = (yy * W + xx)[valid]

        if self.mask_only and self.sensor_mask is not None:
            self.sensor_mask.view(-1)[lin] = True
            return self.MAP

        values = torch.full(
            (lin.numel(),),
            strength,
            device=self.device,
            dtype=self.MAP.dtype,
        )

        self.MAP.view(-1).scatter_add_(0, lin, values)
        return self.MAP

    @torch.no_grad()
    def remove(self, sensor_position: PosLike) -> torch.Tensor:
        if self.mask_only:
            raise RuntimeError("remove() is not supported when mask_only=True.")
        # many sensors
        if not (
            isinstance(sensor_position, tuple)
            and len(sensor_position) == 2
        ) and not isinstance(sensor_position, torch.Tensor):
            out = self.MAP
            for p in sensor_position:
                out = self.remove(tuple(p))
            return out

        # single sensor
        if isinstance(sensor_position, torch.Tensor):
            sensor_position = tuple(int(v) for v in sensor_position.tolist())

        x, y = sensor_position
        x = int(x)
        y = int(y)

        idx = None
        cov = None
        for i in range(len(self.sensor_log) - 1, -1, -1):
            sx, sy, scov = self.sensor_log[i]
            if sx == x and sy == y:
                idx = i
                cov = scov
                break

        if idx is None or cov is None:
            return self.MAP

        self.sensor_log.pop(idx)
        self._ensure_kernel(cov)

        H, W = self.MAP.shape
        strength = -float(self.FIXED_STRENGTH)

        pos = torch.tensor([(x, y)], device=self.device, dtype=torch.long)
        if self._offsets is None:
            raise RuntimeError("Kernel offsets are not initialized.")
        x0 = pos[:, 0]
        y0 = pos[:, 1]
        dy = self._offsets[:, 0]
        dx = self._offsets[:, 1]

        yy = y0[:, None] + dy[None, :]
        xx = x0[:, None] + dx[None, :]

        valid = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)

        lin = (yy * W + xx)[valid]
        values = torch.full(
            (lin.numel(),),
            strength,
            device=self.device,
            dtype=self.MAP.dtype,
        )

        self.MAP.view(-1).scatter_add_(0, lin, values)
        self.MAP.clamp_(min=0.0)
        return self.MAP

    # -------------------------
    # covered / uncovered (with overloads for Pylance)
    # -------------------------
    @overload
    def covered(
        self,
        roi_mask: Optional[MaskLike] = ...,
        *,
        eps: float = ...,
        as_tensor: Literal[False] = ...,
        points: Literal[True],
    ) -> List[XY]: ...

    @overload
    def covered(
        self,
        roi_mask: Optional[MaskLike] = ...,
        *,
        eps: float = ...,
        as_tensor: Literal[False] = ...,
        points: Literal[False] = ...,
    ) -> np.ndarray: ...

    @overload
    def covered(
        self,
        roi_mask: Optional[MaskLike] = ...,
        *,
        eps: float = ...,
        as_tensor: Literal[True],
        points: Literal[False] = ...,
    ) -> torch.Tensor: ...

    def covered(
        self,
        roi_mask: Optional[MaskLike] = None,
        *,
        eps: float = 1e-6,
        as_tensor: bool = False,
        points: bool = False,
    ):
        """
        현재 self.MAP 기준 '커버된' ROI 반환.

        반환:
        - points=True                    : List[(x,y)]
        - points=False, as_tensor=False  : numpy.ndarray(bool)
        - points=False, as_tensor=True   : torch.BoolTensor

        NOTE:
        - points=True일 때는 as_tensor와 무관하게 List[(x,y)]로 반환한다.
        """
        roi = self._to_bool_mask(
            roi_mask,
            device=self.device,
            fallback=(self.map_tensor > 0),
        )

        covered_bool = roi & (self.MAP > (self.map_tensor + eps))

        if points:
            ys, xs = torch.where(covered_bool)
            return list(zip(xs.tolist(), ys.tolist()))

        if as_tensor:
            return covered_bool
        return self._to_numpy_bool(covered_bool)

    @overload
    def uncovered(
        self,
        roi_mask: Optional[MaskLike] = ...,
        *,
        eps: float = ...,
        as_tensor: Literal[False] = ...,
        points: Literal[True],
    ) -> List[XY]: ...

    @overload
    def uncovered(
        self,
        roi_mask: Optional[MaskLike] = ...,
        *,
        eps: float = ...,
        as_tensor: Literal[False] = ...,
        points: Literal[False] = ...,
    ) -> np.ndarray: ...

    @overload
    def uncovered(
        self,
        roi_mask: Optional[MaskLike] = ...,
        *,
        eps: float = ...,
        as_tensor: Literal[True],
        points: Literal[False] = ...,
    ) -> torch.Tensor: ...

    def uncovered(
        self,
        roi_mask: Optional[MaskLike] = None,
        *,
        eps: float = 1e-6,
        as_tensor: bool = False,
        points: bool = False,
    ):
        """
        현재 self.MAP 기준 '미커버' ROI 반환.

        반환:
        - points=True                    : List[(x,y)]
        - points=False, as_tensor=False  : numpy.ndarray(bool)
        - points=False, as_tensor=True   : torch.BoolTensor

        NOTE:
        - points=True일 때는 as_tensor와 무관하게 List[(x,y)]로 반환한다.
        """
        roi = self._to_bool_mask(
            roi_mask,
            device=self.device,
            fallback=(self.map_tensor > 0),
        )

        covered_bool = roi & (self.MAP > (self.map_tensor + eps))
        uncovered_bool = roi & (~covered_bool)

        if points:
            ys, xs = torch.where(uncovered_bool)
            return list(zip(xs.tolist(), ys.tolist()))

        if as_tensor:
            return uncovered_bool
        return self._to_numpy_bool(uncovered_bool)


    def extract_only_sensor(self):
        mask = (self.MAP > self.FIXED_STRENGTH).to(dtype=self.MAP.dtype)
        return self.MAP * mask

    def extract_only_sensor_mask(self) -> torch.Tensor:
        if self.sensor_mask is not None:
            return self.sensor_mask
        return self.MAP > self.FIXED_STRENGTH