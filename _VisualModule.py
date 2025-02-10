import numpy as np
import random
import copy
import os
import csv
from datetime import datetime
import matplotlib.pyplot as plt

RESULTS_DIR = "__RESULTS__"
class VisualTool:
    def __init__(self, save_dir="__RESULTS__", show=False):
        """
        시각화 도구.
        
        Parameters:
        - save_dir: 이미지 저장 경로.
        - show: True이면 plt.show()를 실행, False이면 이미지 저장만 수행.
        """
        self.save_dir = save_dir
        self.show = show
        os.makedirs(save_dir, exist_ok=True)
        now = datetime.now()
        self.time = now.strftime("%m-%d-%H-%M")

    def showJetMap_circle(self, title: str, map_data, radius, sensor_positions, save_path=None):
        self._plot_map_with_circles(title, map_data, radius, sensor_positions, cmap='jet', filename="jet_map_circle", save_path=save_path)

    def _plot_map_with_circles(self, title, map_data, radius, sensor_positions, cmap, filename, save_path):
        fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
        cmap_custom = plt.cm.colors.ListedColormap(cmap) if isinstance(cmap, list) else cmap
        ax.imshow(map_data, cmap=cmap_custom, interpolation='nearest', origin='upper')
        ax.set_title(title)

        if sensor_positions:
            for pos in sensor_positions:
                inner = plt.Circle(pos, radius=radius, edgecolor='green', facecolor='white', alpha=0.1, linewidth=0.02)
                border = plt.Circle(pos, radius=radius, edgecolor='green', facecolor='none', linewidth=0.2)
                center = plt.Circle(pos, radius=0.2, edgecolor='red', facecolor='red', linewidth=0.02)
                ax.add_patch(inner)
                ax.add_patch(border)
                ax.add_patch(center)

        self._save_or_show(fig, filename, save_path)

    def _save_or_show(self, fig, filename, save_path):
        if save_path is None:
            save_path = os.path.join(self.save_dir, f"{filename}_{self.time}.png")
        else:
            os.makedirs(save_path, exist_ok=True)
            save_path = os.path.join(save_path, f"{filename}.png")
        fig.savefig(save_path, bbox_inches='tight')
        print(f"📌 그래프 저장 완료: {save_path}")
        if self.show:
            plt.show()
        plt.close(fig)