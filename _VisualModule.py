import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os
from datetime import datetime

class VisualTool:
    def __init__(self, save_dir="__RESULTS__", show=False):
        """
        시각화 도구
        :param save_dir: 기본 이미지 저장 경로 (별도 폴더 지정 가능)
        :param show: True이면 plt.show() 실행, False이면 이미지 저장만 수행
        """
        self.save_dir = save_dir
        self.show = show  # GUI 출력 여부

        # ✅ 저장 폴더가 없으면 생성
        os.makedirs(save_dir, exist_ok=True)

        # ✅ 중복 방지를 위해 날짜/시간 추가
        now = datetime.now()
        self.time = now.strftime("%m-%d-%H-%M")

    def showBinaryMap(self, title: str, data, save_path=None):
        self._plot_map(title, data, cmap=['gray', 'white'], filename="binary_map", save_path=save_path)

    def showJetMap(self, title: str, data, save_path=None):
        self._plot_map(title, data, cmap='jet', filename="jet_map", save_path=save_path)

    def showJetMap_circle(self, title, map_data, radius, sensor_positions, save_path=None):
        self._plot_map_with_circles(title, map_data, radius, sensor_positions, cmap='jet', filename="jet_map_circle", save_path=save_path)

    def showBinaryMap_circle(self, title, map_data, radius, sensor_positions, save_path=None):
        self._plot_map_with_circles(title, map_data, radius, sensor_positions, cmap=['black', 'white'], filename="binary_map_circle", save_path=save_path)

    def _plot_map(self, title, data, cmap, filename, save_path):
        """
        기본적인 맵 시각화 및 저장 기능
        """
        fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
        cmap_custom = plt.cm.colors.ListedColormap(cmap) if isinstance(cmap, list) else cmap
        ax.imshow(data, cmap=cmap_custom, interpolation='nearest', origin='upper')
        ax.set_title(title)

        self._save_or_show(fig, filename, save_path)

    def _plot_map_with_circles(self, title, map_data, radius, sensor_positions, cmap, filename, save_path):
        """
        센서 범위 (원) 표시 맵 시각화 및 저장 기능
        """
        fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
        cmap_custom = plt.cm.colors.ListedColormap(cmap) if isinstance(cmap, list) else cmap
        ax.imshow(map_data, cmap=cmap_custom, interpolation='nearest', origin='upper')
        ax.set_title(title)

        if sensor_positions:
            for pos in sensor_positions:
                inner = Circle(pos, radius=radius, edgecolor='green', facecolor='white', alpha=0.1, linewidth=0.02)
                border = Circle(pos, radius=radius, edgecolor='green', facecolor='none', linewidth=0.2)
                center = Circle(pos, radius=0.2, edgecolor='red', facecolor='red', linewidth=0.02)
                ax.add_patch(inner)
                ax.add_patch(border)
                ax.add_patch(center)

        self._save_or_show(fig, filename, save_path)

    def _save_or_show(self, fig, filename, save_path):
        """
        이미지 저장 또는 GUI 출력
        """
        # ✅ save_path가 파일명이 아닌 **폴더 경로**여야 함.
        if save_path is None:
            save_path = os.path.join(self.save_dir, f"{filename}_{self.time}.png")
        else:
            os.makedirs(save_path, exist_ok=True)  # ✅ 폴더 없으면 생성
            save_path = os.path.join(save_path, f"{filename}.png")  # ✅ 올바른 파일 경로 지정

        fig.savefig(save_path, bbox_inches='tight')
        print(f"📌 그래프 저장 완료: {save_path}")

        if self.show:
            plt.show()

        plt.close(fig)  # ✅ 메모리 해제
