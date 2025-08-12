import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = "__RESULTS__"

class VisualTool:
    def __init__(self, save_dir=RESULTS_DIR, show=False, figsize=(24, 16), dpi=600):
        self.save_dir = save_dir
        self.show = show
        self.figsize = figsize
        self.dpi = dpi
        os.makedirs(save_dir, exist_ok=True)
        self.time = datetime.now().strftime("%m-%d-%H-%M")

    # 내부 유틸
    def _resolve_path(self, filename, save_path):
        if save_path is None:
            return os.path.join(self.save_dir, f"{filename}_{self.time}.png")
        os.makedirs(save_path, exist_ok=True)
        return os.path.join(save_path, f"{filename}.png")

    def _save(self, fig, outpath):
        fig.subplots_adjust(0, 0, 1, 1)  # 여백 제거
        fig.savefig(outpath, bbox_inches='tight', pad_inches=0)
        print(f"그래프 저장 완료 : {outpath}")
        if self.show:
            plt.show()
        plt.close(fig)

    def _normalize_image(self, arr):
        a = np.asarray(arr)
        if a.dtype.kind in ("U", "S", "O"):
            a = a.astype(np.float32)
        if a.dtype == np.bool_:
            a = a.astype(np.uint8)
        return a

    # 0) 기본 맵 저장 (원본 맵)
    def showJetMap(self, map_data, cmap='jet', filename="jet_map", save_path=None):
        map_data = self._normalize_image(map_data)
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        ax.imshow(map_data, cmap=cmap, interpolation='nearest', origin='upper')
        ax.axis('off')  # 축/타이틀 X
        out = self._resolve_path(filename, save_path)
        self._save(fig, out)

    def showJetMap_circle(self, map_data, radius, sensor_positions,
                        cmap='jet', filename="jet_map_circle", save_path=None):
        # 입력 정규화(문자열/불리언 dtype 방지)
        map_data = self._normalize_image(map_data)

        fig, ax = plt.subplots(figsize=(24, 16), dpi=300)
        cmap_custom = plt.cm.colors.ListedColormap(cmap) if isinstance(cmap, list) else cmap
        ax.imshow(map_data, cmap=cmap_custom, interpolation='nearest', origin='upper')

        # 타이틀, 축 제거 (그림만)
        ax.set_title("")  # 혹시 외부에서 넘긴 title 무시하고 그림만 저장하려면 빈 문자열 유지
        ax.axis('off')

        if sensor_positions:
            for pos in sensor_positions:
                inner = plt.Circle(pos, radius=radius, edgecolor='green', facecolor='white', alpha=0.1, linewidth=0.02)
                border = plt.Circle(pos, radius=radius, edgecolor='green', facecolor='none', linewidth=0.2)
                center = plt.Circle(
    pos,
    radius=0.25,
    facecolor='lime',        # 내부를 진하게
    edgecolor='white',        # 배경과 대비되도록
    linewidth=1.0,            # 두께 확실히
    zorder=10                 # 다른 원 위로
)

                ax.add_patch(inner)
                ax.add_patch(border)
                ax.add_patch(center)

        # 여백 제거
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # ✅ 저장 경로 생성 후 _save에 전달 (버그 픽스)
        out = self._resolve_path(filename, save_path)
        self._save(fig, out)


    # 2) 원본 맵 위에 Harris 응답 오버레이
    def overlay_scalar_on_base(self, base, overlay,
                               base_cmap='gray', overlay_cmap='jet', alpha=0.5,
                               filename="harris_on_base", save_path=None):
        base = self._normalize_image(base)
        overlay = self._normalize_image(overlay)

        # 응답맵 정규화(보기 좋게)
        omin, omax = float(overlay.min()), float(overlay.max())
        if omax - omin < 1e-12:
            overlay_norm = np.zeros_like(overlay, dtype=np.float32)
        else:
            overlay_norm = (overlay - omin) / (omax - omin)

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        ax.imshow(base, cmap=base_cmap, interpolation='nearest', origin='upper')
        ax.imshow(overlay_norm, cmap=overlay_cmap, interpolation='nearest', origin='upper', alpha=alpha)
        ax.axis('off')
        out = self._resolve_path(filename, save_path)
        self._save(fig, out)

    # 3) 원본 맵 위에 NMS 바이너리 마스크 오버레이(흑백)
    def overlay_mask_on_base(self, base, mask, base_cmap='gray', alpha=0.5,
                             filename="nms_on_base", save_path=None):
        base = self._normalize_image(base)
        mask = (np.asarray(mask) > 0).astype(np.float32)

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        ax.imshow(base, cmap=base_cmap, interpolation='nearest', origin='upper')
        ax.imshow(mask, cmap='gray', interpolation='nearest', origin='upper', alpha=alpha)
        ax.axis('off')
        out = self._resolve_path(filename, save_path)
        self._save(fig, out)
