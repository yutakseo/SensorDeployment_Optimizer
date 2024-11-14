import numpy as np
from scipy.ndimage import distance_transform_edt
from numba import cuda

class Sensor:
    def __init__(self, MAP):
        self.map_data = np.array(MAP)
        self.width = self.map_data.shape[1]
        self.height = self.map_data.shape[0]
        self.circle = None
        self.radius = None

    def create_circle(self, radius):
        if self.circle is not None and self.radius == radius:
            return self.circle

        L = 2 * radius + 1
        grid = np.zeros((L, L))
        center = (radius, radius)
        grid[center] = 1
        distance = distance_transform_edt(1 - grid)
        circle_shape = distance <= radius
        
        self.circle = circle_shape.astype(np.int8)
        self.radius = radius
        return self.circle

    def deploy(self, sensor_position: tuple, coverage: int):
        circle = self.create_circle(coverage)
        center_y, center_x = sensor_position

        # GPU로 데이터 전송
        map_data_device = cuda.to_device(self.map_data)
        circle_device = cuda.to_device(circle)
        
        # 그리드 및 블록 설정
        threadsperblock = (32, 32)  # 블록당 스레드 수를 32x32로 설정
        blockspergrid_x = int(np.ceil(self.width / threadsperblock[0]))
        blockspergrid_y = int(np.ceil(self.height / threadsperblock[1]))
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        # CUDA 커널 호출
        deploy_circle_gpu[blockspergrid, threadsperblock](map_data_device, circle_device, center_x, center_y, coverage, self.width, self.height)
        
        # 결과를 다시 CPU로 가져옴
        self.map_data = map_data_device.copy_to_host()
        return self.map_data

    def result(self):
        return self.map_data

@cuda.jit
def deploy_circle_gpu(map_data, circle, center_x, center_y, coverage, width, height):
    i, j = cuda.grid(2)  # 2D 그리드
    if i < -coverage or i > coverage or j < -coverage or j > coverage:
        return
    
    map_x = center_x + j
    map_y = center_y + i
    
    if 0 <= map_x < width and 0 <= map_y < height:
        if circle[i + coverage, j + coverage]:
            cuda.atomic.add(map_data, (map_y, map_x), 10)

