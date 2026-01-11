import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Optional, Union

RESULTS_DIR = "__RESULTS__"

class VisualTool:
    def __init__(
        self,
        save_dir: str = RESULTS_DIR,
        show: bool = False,
        save: bool = True,
        size: Tuple[int, int] = (6, 4),
        stamp_filename: bool = False,
    ):
        self.root_dir = save_dir
        self.show = show
        self.save = save
        self.figsize = size
        self.stamp_filename = stamp_filename

        self.time = datetime.now().strftime("%m-%d-%H-%M")
        self.output_dir: Optional[str] = (
            os.path.join(self.root_dir, self.time) if self.save else None
        )

    def _resolve_dir(self, path: Optional[str]) -> str:
        if path is None:
            if self.output_dir is not None:
                return self.output_dir
            return os.path.join(self.root_dir, self.time)

        if os.path.isabs(path):
            return path
        return os.path.join(self.root_dir, path)

    def _to_numpy(self, img):
        if hasattr(img, "MAP"):
            img = img.MAP

        try:
            import torch
            if isinstance(img, torch.Tensor):
                img = img.detach().cpu().numpy()
        except Exception:
            pass

        return img

    def _normalize_image(self, img: Union[np.ndarray, List]) -> np.ndarray:
        img = self._to_numpy(img)

        arr = np.asarray(img)
        if arr.dtype.kind in ("U", "S", "O"):
            try:
                arr = arr.astype(np.float32)
            except Exception as e:
                raise TypeError(
                    f"map_data must be numeric; failed to cast from {arr.dtype}: {e}"
                )
        if arr.dtype == np.bool_:
            arr = arr.astype(np.uint8)
        if arr.ndim == 2:
            return arr
        if arr.ndim == 3 and arr.shape[2] in (3, 4):
            return arr
        raise ValueError(
            f"Expected 2D grayscale or 3D RGB(A) array, got shape {arr.shape} and dtype {arr.dtype}"
        )

    def _normalize_positions(
        self,
        positions: Optional[
            List[Tuple[Union[int, float, str], Union[int, float, str]]]
        ],
    ) -> List[Tuple[float, float]]:
        if not positions:
            return []
        safe = []
        for p in positions:
            try:
                x, y = float(p[0]), float(p[1])
                safe.append((x, y))
            except Exception as e:
                raise TypeError(f"Invalid sensor position {p}: {e}")
        return safe

    def _resolve_cmap(self, cmap):
        if isinstance(cmap, list):
            try:
                return plt.cm.colors.ListedColormap(cmap)
            except Exception as e:
                raise ValueError(f"Invalid cmap list: {e}")
        return cmap

    def set_output(self, subdir: Optional[str] = None, timestamped: bool = True) -> None:
        if subdir is None:
            base = os.path.join(self.root_dir, self.time)
        else:
            base = os.path.join(self.root_dir, subdir)

        self.output_dir = os.path.join(base, self.time) if timestamped else base

    def showJetMap_circle(
        self,
        map_data: Union[np.ndarray, List],
        sensor_positions: Optional[
            List[Tuple[Union[int, float, str], Union[int, float, str]]]
        ],
        title: str = "MAP_with_sensor",
        radius: float = 45,
        cmap: Union[str, list] = "jet",
        filename: str = "map_with_sensor",
        save_path: Optional[str] = None,
    ) -> None:
        map_data = self._normalize_image(map_data)
        sensor_positions = self._normalize_positions(sensor_positions)
        cmap_custom = self._resolve_cmap(cmap)

        fig, ax = plt.subplots(figsize=self.figsize, dpi=150)
        ax.imshow(map_data, cmap=cmap_custom, interpolation="nearest", origin="upper")
        ax.set_title(title)

        for pos in sensor_positions:
            inner = plt.Circle(
                pos,
                radius=radius / 5,
                edgecolor="lime",
                facecolor="white",
                alpha=0.1,
                linewidth=0.02,
            )
            border = plt.Circle(
                pos,
                radius=radius / 5,
                edgecolor="lime",
                facecolor="none",
                linewidth=0.2,
            )
            center = plt.Circle(
                pos,
                radius=0.2,
                edgecolor="lime",
                facecolor="lime",
                linewidth=0.02,
            )
            ax.add_patch(inner)
            ax.add_patch(border)
            ax.add_patch(center)

        self.save_or_show(fig, filename, save_path)

    def showJetMap(
        self,
        map_data: Union[np.ndarray, List],
        title: str = "MAP",
        cmap: Union[str, list] = "jet",
        filename: str = "jet_map",
        save_path: Optional[str] = None,
    ) -> None:
        map_data = self._normalize_image(map_data)
        cmap_custom = self._resolve_cmap(cmap)

        fig, ax = plt.subplots(figsize=self.figsize, dpi=150)
        ax.imshow(map_data, cmap=cmap_custom, interpolation="nearest", origin="upper")
        ax.set_title(title)
        self.save_or_show(fig, filename, save_path)

    def save_or_show(
        self, fig: plt.Figure, filename: str, save_path: Optional[str] = None
    ) -> None:

        # ===== 1) 화면 출력 (타이틀 포함 상태) =====
        if self.show:
            plt.show()

        # ===== 2) 저장 직전에만 타이틀 제거 =====
        for ax in fig.axes:
            ax.set_axis_off()
            ax.set_title("")   # ★ 저장본에서는 항상 제거

        fname = (
            f"{filename}_{self.time}.png"
            if self.stamp_filename
            else f"{filename}.png"
        )

        # ===== 3) 파일 저장 =====
        if self.save:
            dirpath = self._resolve_dir(save_path)
            os.makedirs(dirpath, exist_ok=True)
            outpath = os.path.join(dirpath, fname)
            fig.savefig(outpath, bbox_inches="tight", pad_inches=0)
            print(f"Saved figure: {outpath}")

        if not self.show and not self.save:
            print("Warning: Both show=False and save=False → Nothing will happen.")

        plt.close(fig)


            
    def map_check(
        self,
        map_data,
        title: Optional[str] = None,
        return_stats: bool = False,
        *,
        target_value: int = 1,
        show_original: bool = True,
        filename: str = "map_check",
        save_path: Optional[str] = None,
    ):
        GRID_M = 5.0
        CELL_AREA_M2 = GRID_M * GRID_M          # 25 m^2
        HA_M2 = 10_000.0
        KM2_M2 = 1_000_000.0

        arr = self._normalize_image(map_data)
        if arr.ndim != 2:
            raise ValueError(f"map_check expects 2D map. Got shape={arr.shape}")

        total_cells = int(arr.size)

        total_area_m2 = total_cells * CELL_AREA_M2
        total_area_ha = total_area_m2 / HA_M2
        total_area_km2 = total_area_m2 / KM2_M2

        target_cells = int(np.sum(arr == target_value))
        target_area_m2 = target_cells * CELL_AREA_M2
        target_area_ha = target_area_m2 / HA_M2
        target_area_km2 = target_area_m2 / KM2_M2
        target_ratio = (target_cells / total_cells) if total_cells > 0 else 0.0

        unique, counts = np.unique(arr, return_counts=True)
        value_counts = {int(k): int(v) for k, v in zip(unique, counts)}

        print("========== MAP CHECK ==========")
        print(f"Map shape (H,W): {arr.shape[0]} x {arr.shape[1]}")
        print(f"Grid size: {GRID_M:.1f}m x {GRID_M:.1f}m  |  Cell area: {CELL_AREA_M2:.1f} m^2")
        print("--------------------------------")
        print(f"Total cells: {total_cells:,}")
        print(
            f"Total area : "
            f"{total_area_m2:,.2f} m^2  |  "
            f"{total_area_ha:,.4f} ha  |  "
            f"{total_area_km2:,.6f} km^2"
        )
        print("--------------------------------")
        print(f"Target value      : {target_value}")
        print(
            f"Target area       : "
            f"{target_area_m2:,.2f} m^2  |  "
            f"{target_area_ha:,.4f} ha  |  "
            f"{target_area_km2:,.6f} km^2"
        )
        print(f"Target cells      : {target_cells:,}  ({target_ratio*100:.2f}%)")
        print("--------------------------------")
        print("Value counts (entire map):")
        for k in sorted(value_counts.keys()):
            pct = (value_counts[k] / total_cells * 100.0) if total_cells > 0 else 0.0
            print(f"  - value {k}: {value_counts[k]:,} ({pct:.2f}%)")
        print("================================")

        # 시각화
        if show_original:
            vis = arr
            cmap = "jet"
        else:
            vis = (arr == target_value).astype(np.uint8)
            cmap = "gray"

        if title is None:
            title = (
                f"Target={target_value} | "
                f"{target_area_m2:,.1f} m² / "
                f"{target_area_ha:.4f} ha / "
                f"{target_area_km2:.6f} km²"
            )

        self.showJetMap(
            map_data=vis,
            title=title,
            cmap=cmap,
            filename=filename,
            save_path=save_path,
        )

        stats = {
            "shape": tuple(arr.shape),
            "grid_m": GRID_M,
            "cell_area_m2": CELL_AREA_M2,
            "total": {
                "cells": total_cells,
                "area_m2": total_area_m2,
                "area_ha": total_area_ha,
                "area_km2": total_area_km2,
            },
            "target": {
                "value": int(target_value),
                "cells": target_cells,
                "area_m2": target_area_m2,
                "area_ha": target_area_ha,
                "area_km2": target_area_km2,
                "ratio": target_ratio,
            },
            "value_counts": value_counts,
        }

        return stats if return_stats else None

