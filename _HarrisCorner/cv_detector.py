import cv2
import numpy as np
from Visual import *

class ComputerVision():
    def __init__(self, map):
        self.map = map
        self.map_data = np.array(self.map, dtype=np.uint8)
        self.vis = VisualTool()
        
    def harris_corner(self, block_size, ksize, k):
        #입력된 맵 데이터 가우시안 블러 처리
        self.map_data = cv2.GaussianBlur(self.map_data, (9,9), 0,0)
        #헤리스 코너 탐색
        self.result = cv2.cornerHarris(self.map_data, block_size, ksize, k)
        
        #추출된 결과 상 최대값 검출
        max_val = np.max(self.result)
        #0과 1로 구성된 맵 데이터 생성
        binary_image = np.zeros_like(self.result, dtype=np.uint8)
        #이진 맵 상에서 검출된 최대값 대비 50% 이상 값을 갖는 지점에 대하여 코너점이라 판별
        binary_image[self.result >= 0.5*max_val] = 1
        dst = np.where(binary_image == 1)
        
        return list(zip(dst[0],dst[1]))

