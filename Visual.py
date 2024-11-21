import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


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
                inner = Circle(pos, radius=radius, edgecolor='green', facecolor='white', alpha=0.02, linewidth=0.02)
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
