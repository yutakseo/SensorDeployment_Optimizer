import cv2
import numpy as np

class HarrisCorner():
    def __init__(self, MAP):
        self.map_data = np.array(MAP, dtype=np.uint8)
    
    def gaussianBlur(self, map, ksize=(9,9), sigX=0, sigY=0):
        blurred_map = cv2.GaussianBlur(src=np.array(map,dtype=np.uint8), ksize=ksize, sigmaX=sigX, sigmaY=sigY)
        return blurred_map
    
    def Blur(self, map, ksize=(9,9)):
        blurred_map = cv2.blur(src=np.array(map), ksize=ksize)
        return blurred_map
    
    def harrisCorner(self, map, block_size=3, ksize=3, k=0.01):
        # Harris 코너 탐지 실행
        filtered_map = cv2.cornerHarris(src=np.array(map,dtype=np.uint8), blockSize=block_size, ksize=ksize, k=k)
        # 임계값 설정
        threshold = 0.1 * filtered_map.max()
        filtered_map[filtered_map < threshold] = 0
        
        # 비최대 억제 적용
        dilated_map = cv2.dilate(filtered_map, None)  # 최대 응답값 지점 강조
        non_max_suppressed = np.zeros_like(filtered_map)
        non_max_suppressed[(filtered_map == dilated_map) & (filtered_map > threshold)] = filtered_map[(filtered_map == dilated_map) & (filtered_map > threshold)]
        
        # 이진화된 결과 생성
        binarized_result = np.zeros_like(non_max_suppressed, dtype=np.uint8)
        binarized_result[non_max_suppressed > 0] = 1
        
        return binarized_result
    
    def extract(self, map):
        points = np.where(map == 1)
        return list(zip(points[0], points[1]))