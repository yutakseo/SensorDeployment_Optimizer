import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os

class VisualTool:
    def __init__(self, save_dir="__RESULTS__", show=False):
        """
        시각화 도구
        :param save_dir: 이미지 저장 경로
        :param show: True이면 plt.show() 실행, False이면 이미지 저장만 수행
        """
        self.save_dir = save_dir
        self.show = show  # GUI 출력 여부
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def showNumpyMap(self, title, data):
        print(title)
        matrix = np.array(data)
        print(matrix)

    def showBinaryMap(self, title: str, data):
        self._plot_map(title, data, cmap=['gray', 'white'], filename="binary_map.png")

    def showJetMap(self, title: str, data):
        self._plot_map(title, data, cmap='jet', filename="jet_map.png")

    def showJetMap_circle(self, title, map_data, radius, sensor_positions: list):
        self._plot_map_with_circles(title, map_data, radius, sensor_positions, cmap='jet', filename="jet_map_circle.png")

    def showBinaryMap_circle(self, title, map_data, radius, sensor_positions: list):
        self._plot_map_with_circles(title, map_data, radius, sensor_positions, cmap=['black', 'white'], filename="binary_map_circle.png")

    def _plot_map(self, title, data, cmap, filename):
        """
        기본적인 맵 시각화 및 저장 기능
        """
        fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
        cmap_custom = plt.cm.colors.ListedColormap(cmap) if isinstance(cmap, list) else cmap
        ax.imshow(data, cmap=cmap_custom, interpolation='nearest', origin='upper')
        ax.set_title(title)

        self._save_or_show(fig, filename)

    def _plot_map_with_circles(self, title, map_data, radius, sensor_positions, cmap, filename):
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

        self._save_or_show(fig, filename)

    def _save_or_show(self, fig, filename):
        """
        이미지 저장 또는 GUI 출력
        """
        save_path = os.path.join(self.save_dir, filename)
        fig.savefig(save_path, bbox_inches='tight')
        print(f"📌 그래프 저장 완료: {save_path}")

        if self.show:
            plt.show()

        plt.close(fig)  # 메모리 해제

    def returnCordinate(self, data):
        grid = [(j+1, i+1) for i in range(len(data)) for j in range(len(data[0])) if data[i][j] == 1]
        return grid
