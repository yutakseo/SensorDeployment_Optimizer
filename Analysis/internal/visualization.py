import os
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch, Rectangle
import numpy as np
from typing import Dict, List, Optional, Sequence, Tuple, Union

RESULTS_DIR = "__RESULTS__"
GRID_SIZE_M = 5.0
MAP_OVERVIEW_COLORS = {
    0: "#f7f7f3",
    1: "#4a4a4a",
    2: "#ffffff",
    3: "#d8d8d2",
    4: "#b8b8b0",
}
MAP_OVERVIEW_HATCHES = {
    1: None,
    2: None,
    3: "///",
    4: None,
}
MAP_OVERVIEW_LABELS = {
    1: "Existing structure",
    2: "Installable area",
    3: "Restricted area",
}
INSTALLABLE_OVERVIEW_LABELS = {2: "Installable area"}
RESTRICTED_OVERVIEW_LABELS = {3: "Restricted area"}
JOBSITE_OVERVIEW_LABELS = {4: "Jobsite area"}
SOFT_GRAY_CMAP = LinearSegmentedColormap.from_list(
    "sensor_soft_gray",
    ["#f7f7f3", "#c9c9c2", "#8f8f88"],
)


class VisualTool:
    def __init__(
        self,
        save_dir: str = RESULTS_DIR,
        show: bool = False,
        save: bool = True,
        size: Tuple[int, int] = (6, 4),
        dpi: int = 150,                 # DPI 조절
        stamp_filename: bool = False,
        tight: bool = True,             # bbox/pad 제어
        pad_inches: float = 0.0,
        facecolor: Optional[str] = None,  # 저장 배경 (None=matplotlib default)
        save_title: bool = False,
    ):
        self.root_dir = save_dir
        self.show = show
        self.save = save
        self.figsize = size
        self.dpi = int(dpi)
        self.stamp_filename = stamp_filename

        self.tight = bool(tight)
        self.pad_inches = float(pad_inches)
        self.facecolor = facecolor
        self.save_title = bool(save_title)

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
        safe: List[Tuple[float, float]] = []
        for p in positions:
            try:
                x, y = float(p[0]), float(p[1])
                safe.append((x, y))
            except Exception as e:
                raise TypeError(f"Invalid sensor position {p}: {e}")
        return safe

    def _resolve_cmap(self, cmap):
        """
        cmap:
          - str: matplotlib colormap name
          - list: list of colors -> ListedColormap
        """
        if isinstance(cmap, list):
            try:
                return plt.cm.colors.ListedColormap(cmap)
            except Exception as e:
                raise ValueError(f"Invalid cmap list: {e}")
        return cmap

    def setOutput(self, subdir: Optional[str] = None, timestamped: bool = True) -> None:
        if subdir is None:
            base = os.path.join(self.root_dir, self.time)
        else:
            base = os.path.join(self.root_dir, subdir)

        self.output_dir = os.path.join(base, self.time) if timestamped else base

    def _imshow(
        self,
        ax: plt.Axes,
        map_data: np.ndarray,
        cmap_custom,
        *,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        zone_style: Optional[str] = None,
    ) -> None:
        ax.set_axis_off()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(
            left=False,
            bottom=False,
            labelleft=False,
            labelbottom=False,
        )
        if self._draw_zone_map(ax, map_data, zone_style=zone_style):
            return

        # vmin/vmax를 고정해 정규화 흔들림(체감상 "jet처럼 보임") 방지
        if vmin is None:
            vmin = float(np.min(map_data))
        if vmax is None:
            vmax = float(np.max(map_data))

        ax.imshow(
            map_data,
            cmap=cmap_custom,
            interpolation="nearest",
            origin="upper",
            vmin=vmin,
            vmax=vmax,
        )

    def _draw_zone_map(
        self,
        ax: plt.Axes,
        map_data: np.ndarray,
        *,
        zone_style: Optional[str] = None,
    ) -> bool:
        if map_data.ndim != 2:
            return False

        style = str(zone_style or "auto").lower()
        base_area = np.empty((0, 2), dtype=int)
        if style in {"installable", "installable_layer"}:
            installable = np.argwhere(map_data > 0)
            restricted = np.empty((0, 2), dtype=int)
        elif style in {"restricted", "restricted_layer", "road", "road_layer"}:
            installable = np.empty((0, 2), dtype=int)
            restricted = np.argwhere(map_data > 0)
        elif style in {"jobsite", "jobsite_layer"}:
            installable = np.argwhere(map_data > 0)
            restricted = np.empty((0, 2), dtype=int)
        else:
            base_area = np.argwhere(map_data == 1)
            installable = np.argwhere(map_data == 2)
            restricted = np.argwhere(map_data == 3)

        if style != "auto":
            base_area = np.empty((0, 2), dtype=int)

        if base_area.size == 0 and installable.size == 0 and restricted.size == 0:
            return False

        height, width = map_data.shape
        ax.set_facecolor("white")
        ax.set_xlim(-0.5, width - 0.5)
        ax.set_ylim(height - 0.5, -0.5)
        ax.set_aspect("equal")

        if base_area.size:
            patches = [
                Rectangle((float(x) - 0.5, float(y) - 0.5), 1.0, 1.0)
                for y, x in base_area
            ]
            ax.add_collection(
                PatchCollection(
                    patches,
                    facecolor="0.35",
                    edgecolor="0.35",
                    linewidth=0.0,
                    match_original=False,
                )
            )

        if installable.size:
            patches = [
                Rectangle((float(x) - 0.5, float(y) - 0.5), 1.0, 1.0)
                for y, x in installable
            ]
            ax.add_collection(
                PatchCollection(
                    patches,
                    facecolor="white",
                    edgecolor="0.35",
                    linewidth=0.6,
                    match_original=False,
                )
            )

        if restricted.size:
            patches = [
                Rectangle((float(x) - 0.5, float(y) - 0.5), 1.0, 1.0)
                for y, x in restricted
            ]
            ax.add_collection(
                PatchCollection(
                    patches,
                    facecolor="white",
                    edgecolor="0.25",
                    linewidth=0.8,
                    hatch="///",
                    match_original=False,
                )
            )

        ax.add_patch(
            Rectangle(
                (-0.5, -0.5),
                float(width),
                float(height),
                facecolor="none",
                edgecolor="black",
                linewidth=1.2,
            )
        )

        return True

    def showMapCircle(
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
        *,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        zone_style: Optional[str] = None,
        grid_m: float = GRID_SIZE_M,
    ) -> None:
        _ = (cmap, vmin, vmax)
        map_data = self._normalize_image(map_data)
        if map_data.ndim != 2:
            raise ValueError(
                f"showMapCircle expects 2D map data. Got shape {map_data.shape}"
            )

        sensor_positions = self._normalize_positions(sensor_positions)
        map_data, labels = self._prepareOverviewMap(
            map_data,
            zone_style=zone_style,
        )
        map_data, pad_left, pad_top = self._padSquareWithOffset(
            map_data,
            fill_value=0,
        )
        sensor_positions = self._shiftPositions(
            sensor_positions,
            x_offset=pad_left,
            y_offset=pad_top,
        )

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        self._drawMapOverview(ax, map_data, title=title, grid_m=grid_m, labels=labels)
        self._addSensorCircles(
            ax,
            sensor_positions=sensor_positions,
            radius_cells=float(radius) / float(grid_m),
        )
        self.saveOrShow(
            fig,
            filename,
            save_path,
            preserve_axes=True,
            square_output=True,
        )
        
        
    def showMapDot(
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
        *,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        zone_style: Optional[str] = None,
    ) -> None:
        map_data = self._normalize_image(map_data)
        sensor_positions = self._normalize_positions(sensor_positions)
        cmap_custom = self._resolve_cmap(cmap)

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        self._imshow(ax, map_data, cmap_custom, vmin=vmin, vmax=vmax, zone_style=zone_style)
        ax.set_title(title)

        for pos in sensor_positions:
            center = plt.Circle(
                pos,
                radius=0.5,
                edgecolor="red",
                facecolor="red",
                linewidth=1.0,
            )

            ax.add_patch(center)

        self.saveOrShow(fig, filename, save_path)



    def showMap(
        self,
        map_data: Union[np.ndarray, List],
        title: str = "MAP",
        cmap: Union[str, list] = "jet",
        filename: str = "map",
        save_path: Optional[str] = None,
        *,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        zone_style: Optional[str] = None,
    ) -> None:
        map_data = self._normalize_image(map_data)
        cmap_custom = self._resolve_cmap(cmap)

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        self._imshow(ax, map_data, cmap_custom, vmin=vmin, vmax=vmax, zone_style=zone_style)
        ax.set_title(title)
        self.saveOrShow(fig, filename, save_path)

    def showMapOverview(
        self,
        map_data: Union[np.ndarray, List],
        title: str = "Construction Site Map",
        filename: str = "construction_map_overview",
        save_path: Optional[str] = None,
        *,
        grid_m: float = GRID_SIZE_M,
        zone_style: Optional[str] = None,
        cmap: Union[str, list] = "Greys",
        colorbar_label: str = "Response",
        show_colorbar: bool = True,
        base_map: Optional[Union[np.ndarray, List]] = None,
        overlay_alpha: float = 0.72,
        overlay_percentile: float = 99.0,
        overlay_threshold_percentile: Optional[float] = None,
        overlay_spread: int = 0,
        overlay_gamma: float = 1.0,
        overlay_cmap: Union[str, list] = "YlOrRd",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> None:
        map_data = self._normalize_image(map_data)
        if map_data.ndim != 2:
            raise ValueError(
                f"showMapOverview expects 2D map data. Got shape {map_data.shape}"
            )

        if self._isHeatmapOverlay(zone_style):
            if base_map is None:
                raise ValueError("base_map is required for heatmap overlay overview.")
            self.showHeatmapOverlayOverview(
                heatmap_data=map_data,
                base_map=base_map,
                title=title,
                filename=filename,
                save_path=save_path,
                grid_m=grid_m,
                alpha=overlay_alpha,
                vmax_percentile=overlay_percentile,
                threshold_percentile=overlay_threshold_percentile,
                spread=overlay_spread,
                gamma=overlay_gamma,
                cmap=overlay_cmap,
                vmin=vmin,
                vmax=vmax,
            )
            return

        if self._isScalarOverview(zone_style):
            self.showScalarOverview(
                map_data=map_data,
                title=title,
                filename=filename,
                save_path=save_path,
                cmap=cmap,
                grid_m=grid_m,
                colorbar_label=colorbar_label,
                show_colorbar=show_colorbar,
                vmin=vmin,
                vmax=vmax,
            )
            return

        map_data, labels = self._prepareOverviewMap(
            map_data,
            zone_style=zone_style,
        )
        map_data = self._padSquare(map_data, fill_value=0)
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        self._drawMapOverview(ax, map_data, title=title, grid_m=grid_m, labels=labels)
        self.saveOrShow(fig, filename, save_path, preserve_axes=True)

    def _isScalarOverview(self, zone_style: Optional[str]) -> bool:
        style = str(zone_style or "").lower()
        return style in {"scalar", "continuous", "heatmap", "blur", "gaussian"}

    def _isHeatmapOverlay(self, zone_style: Optional[str]) -> bool:
        style = str(zone_style or "").lower()
        return style in {
            "heatmap_overlay",
            "corner_heatmap",
            "harris",
            "harris_corner",
        }

    def _prepareOverviewMap(
        self,
        map_data: np.ndarray,
        *,
        zone_style: Optional[str],
    ) -> Tuple[np.ndarray, Dict[int, str]]:
        style = str(zone_style or "auto").lower()
        if style in {"auto", "full", "construction"}:
            return map_data, MAP_OVERVIEW_LABELS

        layer = (map_data > 0).astype(np.uint8)
        if style in {"installable", "installable_layer"}:
            return layer * 2, INSTALLABLE_OVERVIEW_LABELS
        if style in {"restricted", "restricted_layer", "road", "road_layer"}:
            return layer * 3, RESTRICTED_OVERVIEW_LABELS
        if style in {"jobsite", "jobsite_layer"}:
            return layer * 4, JOBSITE_OVERVIEW_LABELS

        raise ValueError(
            "zone_style must be one of auto, installable, restricted, jobsite, "
            "scalar, or heatmap_overlay."
        )

    def _padSquare(
        self,
        map_data: np.ndarray,
        *,
        fill_value: Union[int, float],
    ) -> np.ndarray:
        padded, _, _ = self._padSquareWithOffset(
            map_data,
            fill_value=fill_value,
        )
        return padded

    def _padSquareWithOffset(
        self,
        map_data: np.ndarray,
        *,
        fill_value: Union[int, float],
    ) -> Tuple[np.ndarray, int, int]:
        height, width = map_data.shape
        if height == width:
            return map_data, 0, 0

        size = max(height, width)
        pad_height = size - height
        pad_width = size - width
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        padded = np.pad(
            map_data,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=fill_value,
        )
        return padded, pad_left, pad_top

    def _shiftPositions(
        self,
        positions: List[Tuple[float, float]],
        *,
        x_offset: int,
        y_offset: int,
    ) -> List[Tuple[float, float]]:
        if x_offset == 0 and y_offset == 0:
            return positions
        return [(x + x_offset, y + y_offset) for x, y in positions]

    def _addSensorCircles(
        self,
        ax: plt.Axes,
        *,
        sensor_positions: List[Tuple[float, float]],
        radius_cells: float,
    ) -> None:
        for pos in sensor_positions:
            coverage = plt.Circle(
                pos,
                radius=radius_cells,
                edgecolor="#2563eb",
                facecolor="#60a5fa",
                alpha=0.18,
                linewidth=0.8,
            )
            border = plt.Circle(
                pos,
                radius=radius_cells,
                edgecolor="#1d4ed8",
                facecolor="none",
                alpha=0.9,
                linewidth=0.8,
            )
            center = plt.Circle(
                pos,
                radius=0.55,
                edgecolor="#ff0000",
                facecolor="#ff0000",
                linewidth=0.7,
            )
            ax.add_patch(coverage)
            ax.add_patch(border)
            ax.add_patch(center)

    def _drawMapOverview(
        self,
        ax: plt.Axes,
        map_data: np.ndarray,
        *,
        title: str,
        grid_m: float,
        labels: Dict[int, str],
    ) -> None:
        height, width = map_data.shape
        ax.set_facecolor(MAP_OVERVIEW_COLORS[0])
        ax.set_xlim(-0.5, width - 0.5)
        ax.set_ylim(height - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.set_title(
            title,
            pad=6,
            fontsize=10,
            fontfamily="serif",
            fontweight="normal",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(
            left=False,
            bottom=False,
            labelleft=False,
            labelbottom=False,
        )
        ax.grid(color="#deded8", linewidth=0.3, alpha=0.45)
        for spine in ax.spines.values():
            spine.set_visible(False)

        self._addOverviewCells(ax, map_data)
        self._addOverviewBoundary(ax, width=width, height=height)
        self._addOverviewScale(ax, width=width, height=height, grid_m=grid_m)
        self._addOverviewLegend(ax, labels=labels)

    def _addOverviewCells(self, ax: plt.Axes, map_data: np.ndarray) -> None:
        for value, color in MAP_OVERVIEW_COLORS.items():
            if value == 0:
                continue
            cells = np.argwhere(map_data == value)
            if cells.size == 0:
                continue

            patches = [
                Rectangle((float(x) - 0.5, float(y) - 0.5), 1.0, 1.0)
                for y, x in cells
            ]
            hatch = MAP_OVERVIEW_HATCHES[value]
            edgecolor = "#6f6f69" if value == 2 else "#4f4f49"
            linewidth = 0.45 if value == 2 else 0.15
            ax.add_collection(
                PatchCollection(
                    patches,
                    facecolor=color,
                    edgecolor=edgecolor,
                    linewidth=linewidth,
                    hatch=hatch,
                    match_original=False,
                )
            )

    def _addOverviewBoundary(
        self,
        ax: plt.Axes,
        *,
        width: int,
        height: int,
        zorder: float = 3,
    ) -> None:
        ax.add_patch(
            Rectangle(
                (-0.5, -0.5),
                float(width),
                float(height),
                facecolor="none",
                edgecolor="#1f1f1f",
                linewidth=1.0,
                zorder=zorder,
            )
        )

    def _addOverviewScale(
        self,
        ax: plt.Axes,
        *,
        width: int,
        height: int,
        grid_m: float,
        zorder: float = 4,
    ) -> None:
        scale_cells = max(1, int(round(50.0 / grid_m)))
        x_start = max(1.0, width - scale_cells - 4.0)
        x_end = x_start + scale_cells
        y = height - 4.0
        ax.plot(
            [x_start, x_end],
            [y, y],
            color="#1f1f1f",
            linewidth=1.6,
            zorder=zorder,
        )
        ax.plot(
            [x_start, x_start],
            [y - 0.8, y + 0.8],
            color="#1f1f1f",
            linewidth=1.0,
            zorder=zorder,
        )
        ax.plot(
            [x_end, x_end],
            [y - 0.8, y + 0.8],
            color="#1f1f1f",
            linewidth=1.0,
            zorder=zorder,
        )
        ax.text(
            (x_start + x_end) / 2.0,
            y - 1.8,
            "50 m",
            ha="center",
            va="bottom",
            fontsize=8,
            fontfamily="serif",
            color="#1f1f1f",
            zorder=zorder,
        )

    def _addOverviewLegend(self, ax: plt.Axes, *, labels: Dict[int, str]) -> None:
        handles = [
            Patch(
                facecolor=MAP_OVERVIEW_COLORS[value],
                edgecolor="#4f4f49",
                hatch=MAP_OVERVIEW_HATCHES[value],
                label=label,
            )
            for value, label in labels.items()
        ]
        legend = ax.legend(
            handles=handles,
            loc="upper right",
            borderaxespad=0.7,
            frameon=True,
            framealpha=0.92,
            edgecolor="#d0d0ca",
            facecolor="white",
            prop={"family": "serif", "size": 8},
        )
        legend.set_zorder(10)

    def showScalarOverview(
        self,
        map_data: Union[np.ndarray, List],
        title: str = "Scalar Map Overview",
        filename: str = "scalar_map_overview",
        save_path: Optional[str] = None,
        *,
        cmap: Union[str, list] = "Greys",
        grid_m: float = GRID_SIZE_M,
        colorbar_label: str = "Response",
        show_colorbar: bool = True,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> None:
        map_data = self._normalize_image(map_data)
        if map_data.ndim != 2:
            raise ValueError(
                f"showScalarOverview expects 2D map data. Got shape {map_data.shape}"
            )

        if vmin is None:
            vmin = float(np.min(map_data))
        if vmax is None:
            vmax = float(np.max(map_data))

        map_data = self._padSquare(map_data, fill_value=vmin)
        cmap_custom = self._resolveScalarCmap(cmap)

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        image = self._drawScalarOverview(
            ax,
            map_data,
            title=title,
            grid_m=grid_m,
            cmap=cmap_custom,
            vmin=vmin,
            vmax=vmax,
        )
        if show_colorbar:
            self._addScalarColorbar(ax, image, label=colorbar_label)
        self.saveOrShow(fig, filename, save_path, preserve_axes=True)

    def _resolveScalarCmap(self, cmap):
        if isinstance(cmap, str) and cmap.lower() in {"soft_gray", "soft_grey"}:
            return SOFT_GRAY_CMAP
        return self._resolve_cmap(cmap)

    def _drawScalarOverview(
        self,
        ax: plt.Axes,
        map_data: np.ndarray,
        *,
        title: str,
        grid_m: float,
        cmap,
        vmin: float,
        vmax: float,
    ):
        height, width = map_data.shape
        ax.set_facecolor(MAP_OVERVIEW_COLORS[0])
        image = ax.imshow(
            map_data,
            cmap=cmap,
            interpolation="nearest",
            origin="upper",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xlim(-0.5, width - 0.5)
        ax.set_ylim(height - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.set_title(
            title,
            pad=6,
            fontsize=10,
            fontfamily="serif",
            fontweight="normal",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(
            left=False,
            bottom=False,
            labelleft=False,
            labelbottom=False,
        )
        for spine in ax.spines.values():
            spine.set_visible(False)

        self._addOverviewBoundary(ax, width=width, height=height)
        self._addOverviewScale(ax, width=width, height=height, grid_m=grid_m)
        return image

    def _addScalarColorbar(self, ax: plt.Axes, image, *, label: str) -> None:
        cax = ax.inset_axes([0.74, 0.08, 0.025, 0.24])
        colorbar = ax.figure.colorbar(image, cax=cax)
        colorbar.set_label(label, fontsize=8, fontfamily="serif")
        colorbar.ax.tick_params(length=2, width=0.6, labelsize=7)
        for tick in colorbar.ax.get_yticklabels():
            tick.set_fontfamily("serif")

    def showHeatmapOverlayOverview(
        self,
        heatmap_data: Union[np.ndarray, List],
        base_map: Union[np.ndarray, List],
        title: str = "Harris Corner Heatmap Overview",
        filename: str = "harris_corner_heatmap_overview",
        save_path: Optional[str] = None,
        *,
        grid_m: float = GRID_SIZE_M,
        alpha: float = 0.72,
        vmax_percentile: float = 99.0,
        threshold_percentile: Optional[float] = None,
        spread: int = 0,
        gamma: float = 1.0,
        cmap: Union[str, list] = "YlOrRd",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> None:
        heatmap_data = self._normalize_image(heatmap_data)
        base_map = self._normalize_image(base_map)
        if heatmap_data.ndim != 2 or base_map.ndim != 2:
            raise ValueError("heatmap_data and base_map must be 2D arrays.")
        if heatmap_data.shape != base_map.shape:
            raise ValueError(
                "heatmap_data and base_map must have the same shape. "
                f"Got {heatmap_data.shape} and {base_map.shape}."
            )

        if vmin is None:
            vmin = float(np.min(heatmap_data))
        if threshold_percentile is not None:
            vmin = self._heatmapVmax(
                heatmap_data,
                percentile=float(threshold_percentile),
            )
        if vmax is None:
            vmax = self._heatmapVmax(heatmap_data, percentile=vmax_percentile)
        if vmax <= vmin:
            vmax = float(np.max(heatmap_data))

        heatmap_data = self._spreadHeatmap(heatmap_data, radius=spread)
        heatmap_data = self._gammaHeatmap(
            heatmap_data,
            vmin=vmin,
            vmax=vmax,
            gamma=gamma,
        )
        vmin = float(np.min(heatmap_data))
        vmax = self._heatmapVmax(heatmap_data, percentile=vmax_percentile)

        base_map = self._padSquare(base_map, fill_value=0)
        heatmap_data = self._padSquare(heatmap_data, fill_value=vmin)

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        self._drawMapOverview(
            ax,
            base_map,
            title=title,
            grid_m=grid_m,
            labels=MAP_OVERVIEW_LABELS,
        )
        self._addHeatmapOverlay(
            ax,
            heatmap_data,
            alpha=alpha,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )
        height, width = heatmap_data.shape
        self._addOverviewBoundary(ax, width=width, height=height, zorder=8)
        self._addOverviewScale(ax, width=width, height=height, grid_m=grid_m, zorder=9)
        self.saveOrShow(fig, filename, save_path, preserve_axes=True)

    def _heatmapVmax(self, heatmap_data: np.ndarray, *, percentile: float) -> float:
        positive_values = heatmap_data[heatmap_data > 0]
        if positive_values.size == 0:
            return float(np.max(heatmap_data))

        clipped_percentile = min(100.0, max(0.0, float(percentile)))
        vmax = float(np.percentile(positive_values, clipped_percentile))
        if vmax <= 0.0:
            return float(np.max(positive_values))
        return vmax

    def _spreadHeatmap(self, heatmap_data: np.ndarray, *, radius: int) -> np.ndarray:
        radius = max(0, int(radius))
        if radius == 0:
            return heatmap_data

        padded = np.pad(
            heatmap_data,
            ((radius, radius), (radius, radius)),
            mode="constant",
            constant_values=float(np.min(heatmap_data)),
        )
        spread = np.zeros_like(heatmap_data, dtype=float)
        height, width = heatmap_data.shape
        for y_offset in range(0, radius * 2 + 1):
            for x_offset in range(0, radius * 2 + 1):
                window = padded[y_offset : y_offset + height, x_offset : x_offset + width]
                spread = np.maximum(spread, window)
        return spread

    def _gammaHeatmap(
        self,
        heatmap_data: np.ndarray,
        *,
        vmin: float,
        vmax: float,
        gamma: float,
    ) -> np.ndarray:
        gamma = max(0.1, float(gamma))
        if vmax <= vmin:
            return heatmap_data

        normalized = np.clip((heatmap_data - vmin) / (vmax - vmin), 0.0, 1.0)
        adjusted = np.power(normalized, gamma)
        return adjusted * (vmax - vmin) + vmin

    def _addHeatmapOverlay(
        self,
        ax: plt.Axes,
        heatmap_data: np.ndarray,
        *,
        alpha: float,
        vmin: float,
        vmax: float,
        cmap: Union[str, list],
    ) -> None:
        masked_heatmap = np.ma.masked_where(heatmap_data <= vmin, heatmap_data)
        cmap_custom = self._resolve_cmap(cmap)
        if hasattr(cmap_custom, "copy"):
            cmap_custom = cmap_custom.copy()
            cmap_custom.set_bad((1.0, 1.0, 1.0, 0.0))
        ax.imshow(
            masked_heatmap,
            cmap=cmap_custom,
            interpolation="nearest",
            origin="upper",
            alpha=float(alpha),
            vmin=vmin,
            vmax=vmax,
            zorder=6,
        )

    def saveOrShow(
        self,
        fig: plt.Figure,
        filename: str,
        save_path: Optional[str] = None,
        *,
        preserve_axes: bool = False,
        square_output: bool = False,
    ) -> None:
        """
        - show=True: 타이틀 포함 상태로 화면 표시
        - save=True: 저장 직전에 타이틀 제거 후 저장
        - preserve_axes=False: 저장 직전에 축까지 제거 후 저장
        """
        # 1) 화면 출력 (타이틀 보임)
        if self.show:
            plt.show()

        # 2) 저장 직전에만 타이틀/축 제거
        if self.save and not self.save_title:
            for ax in fig.axes:
                ax.set_title("")

        if not preserve_axes:
            for ax in fig.axes:
                ax.set_axis_off()
                ax.set_xticks([])
                ax.set_yticks([])
                ax.tick_params(
                    left=False,
                    bottom=False,
                    labelleft=False,
                    labelbottom=False,
                )

        fname = (
            f"{filename}_{self.time}.png"
            if self.stamp_filename
            else f"{filename}.png"
        )

        # 3) 저장
        if self.save:
            dirpath = self._resolve_dir(save_path)
            os.makedirs(dirpath, exist_ok=True)
            outpath = os.path.join(dirpath, fname)

            save_kwargs = {}
            if self.tight and not square_output:
                save_kwargs["bbox_inches"] = "tight"
                save_kwargs["pad_inches"] = self.pad_inches
            if self.facecolor is not None:
                save_kwargs["facecolor"] = self.facecolor

            fig.savefig(outpath, **save_kwargs)
            print(f"Saved figure: {outpath}")

        if not self.show and not self.save:
            print("Warning: Both show=False and save=False → Nothing will happen.")

        plt.close(fig)

    def mapCheck(
        self,
        map_data,
        title: Optional[str] = None,
        return_stats: bool = False,
        *,
        target_value: int = 1,
        target_values: Optional[Sequence[int]] = None,
        cmap: Union[str, list] = "jet",     # ✅ 호출자가 지정한 cmap 그대로 사용
        filename: str = "map_check",
        save_path: Optional[str] = None,
        show_original: bool = True,         # ✅ 원래 기능을 명시적으로 유지
        zone_style: Optional[str] = None,
    ):
        GRID_M = 5.0
        CELL_AREA_M2 = GRID_M * GRID_M
        HA_M2 = 10_000.0
        KM2_M2 = 1_000_000.0

        arr = self._normalize_image(map_data)
        if arr.ndim != 2:
            raise ValueError(f"mapCheck expects 2D map. Got shape={arr.shape}")

        total_cells = int(arr.size)
        total_area_m2 = total_cells * CELL_AREA_M2
        total_area_ha = total_area_m2 / HA_M2
        total_area_km2 = total_area_m2 / KM2_M2

        targets = [int(v) for v in (target_values if target_values is not None else [target_value])]
        target_mask = np.isin(arr, targets)
        target_cells = int(np.sum(target_mask))
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
        target_label = targets[0] if len(targets) == 1 else targets
        print(f"Target value      : {target_label}")
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

        # ✅ 시각화 데이터 선택 (cmap은 절대 덮어쓰지 않음)
        if show_original:
            vis = arr
            vmin, vmax = float(np.min(arr)), float(np.max(arr))
        else:
            vis = target_mask.astype(np.uint8)
            vmin, vmax = 0.0, 1.0

        if title is None:
            title = (
                f"Target={target_label} | "
                f"{target_area_m2:,.1f} m² / "
                f"{target_area_ha:.4f} ha / "
                f"{target_area_km2:.6f} km²"
            )

        self.showMap(
            map_data=vis,
            title=title,
            cmap=cmap,          # ✅ 호출자가 넘긴 cmap 그대로 전달
            filename=filename,
            save_path=save_path,
            vmin=vmin,
            vmax=vmax,
            zone_style=zone_style,
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
                "value": target_label,
                "cells": target_cells,
                "area_m2": target_area_m2,
                "area_ha": target_area_ha,
                "area_km2": target_area_km2,
                "ratio": target_ratio,
            },
            "value_counts": value_counts,
        }

        return stats if return_stats else None


VisualTool.showMap_circle = VisualTool.showMapCircle
VisualTool.showMap_dot = VisualTool.showMapDot
VisualTool.showMap_overview = VisualTool.showMapOverview
VisualTool.showScalar_overview = VisualTool.showScalarOverview
VisualTool.map_check = VisualTool.mapCheck
