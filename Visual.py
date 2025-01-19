import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation


class VisualTool:
    def showNumpyMap(self, title, data):
        print(title)
        matrix = np.array(data)
        print(matrix)
    
    def showBinaryMap(self, title: str, data):
        # 그래프 크기 설정
        plt.figure(figsize=(12, 8))
        matrix = np.array(data)
        rows, cols = matrix.shape
        cmap_custom = plt.cm.colors.ListedColormap(['gray', 'white'])
        plt.imshow(matrix, cmap=cmap_custom, interpolation='nearest', extent=[0, cols, rows, 0], origin='upper')
        plt.title(title)
        plt.show()
        return matrix

    def showJetMap(self, title: str, data):
        # 그래프 크기 설정
        plt.figure(figsize=(12, 8))
        plt.imshow(data, cmap='jet')
        plt.title(title)
        plt.show()

    def showJetMap_circle(self, title, map_data, radius, sensor_positions: list):
        # 그래프 크기 설정
        plt.figure(figsize=(12, 8))
        plt.imshow(map_data, cmap='jet')
        plt.title(title)
        if sensor_positions:
            for pos in sensor_positions:
                inner = Circle(pos, radius=radius, edgecolor='green', facecolor='white', alpha=0.1, linewidth=0.02)
                border = Circle(pos, radius=radius, edgecolor='green', facecolor='none', linewidth=0.2)
                center = Circle(pos, radius=0.2, edgecolor='red', facecolor='red', linewidth=0.02)
                plt.gca().add_patch(inner)
                plt.gca().add_patch(border)
                plt.gca().add_patch(center)
        plt.show()
            
    def showBinaryMap_circle(self, title, map_data, radius, sensor_positions: list):
        # 그래프 크기 설정
        plt.figure(figsize=(12, 8))
        cmap_custom = plt.cm.colors.ListedColormap(['black', 'white'])
        plt.imshow(map_data, cmap=cmap_custom)
        plt.title(title)
        if sensor_positions:
            for pos in sensor_positions:
                inner = Circle(pos, radius=radius, edgecolor='green', facecolor='white', alpha=0.03, linewidth=0.02)
                border = Circle(pos, radius=radius, edgecolor='green', facecolor='none', linewidth=0.2)
                center = Circle(pos, radius=0.35, edgecolor='red', facecolor='red', linewidth=0.05)
                plt.gca().add_patch(inner)
                plt.gca().add_patch(border)
                plt.gca().add_patch(center)
        plt.show()
            
    def returnCordinate(self, data):
        grid = []
        for i in range(len(data)):
            for j in range(len(data[0])):
                if data[i][j] == 1:
                    grid.append("("+str(j+1)+","+str(i+1)+")")
        return grid

    def showEvolution(self, title: str, evolution_data: list):
        """
        시각화를 통해 각 세대의 데이터를 애니메이션으로 표시합니다.
        evolution_data: 세대별 데이터 리스트
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        def update(frame):
            ax.clear()
            ax.set_title(f"{title} - Generation: {frame + 1}")
            ax.imshow(evolution_data[frame], cmap='jet', interpolation='nearest')

        ani = FuncAnimation(fig, update, frames=len(evolution_data), repeat=False)
        plt.show()

    def visualize_pso_progress(self, title: str, map_2d, particle_positions):
        """
        각 반복(iteration)에서 입자 위치를 시각화합니다.
        map_2d: 기본 지도 데이터 (numpy array)
        particle_positions: 현재 입자 위치 리스트
        """
        plt.figure(figsize=(12, 8))
        plt.imshow(map_2d, cmap='binary', interpolation='nearest')
        plt.title(title)
        for pos in particle_positions:
            circle = Circle((pos[1], pos[0]), radius=0.4, edgecolor='red', facecolor='red', alpha=0.8)
            plt.gca().add_patch(circle)
        plt.show()
