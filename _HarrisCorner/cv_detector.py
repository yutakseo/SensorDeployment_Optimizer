import cv2
import numpy as np

class HarrisCorner():
    def __init__(self, MAP):
        self.map_data = np.array(MAP, dtype=np.uint8)

    """가우시안 블러 적용"""
    def gaussianBlur(self, map, ksize=(9,9), sigX=0, sigY=0):
        blurred_map = cv2.GaussianBlur(src=np.array(map,dtype=np.uint8),
                                       ksize=ksize,
                                       sigmaX=sigX,
                                       sigmaY=sigY)
        return blurred_map
    
    """비최대 억제 포함한 Harris 코너 검출"""
    def harrisCorner(self, map, block_size=3, ksize=3, k=0.05, dilate_size=5):
        # 1) Harris 코너 검출
        harris_response = cv2.cornerHarris(src=np.array(map, dtype=np.uint8),
                                           blockSize=block_size,
                                           ksize=ksize,
                                           k=k)
        
        # 2) 임계값 설정 (동적으로 조절)
        threshold = 0.1 * harris_response.max()
        harris_response[harris_response < threshold] = 0

        # 3) 비최대 억제 (Dilate 연산을 더 강하게 적용)
        dilated = cv2.dilate(harris_response, np.ones((dilate_size, dilate_size), np.uint8))

        # 4) 최댓값 비교하여 코너 선택
        non_max_suppressed = np.zeros_like(dilated)
        non_max_suppressed[(harris_response == dilated) & (harris_response > threshold)] = 1

        return non_max_suppressed
    
    """가까운 코너 제거 (Euclidean Distance 기반)"""
    def filter_close_corners(self, points, min_distance=5):
        filtered_points = []
        for p in points:
            if all(np.linalg.norm(np.array(p) - np.array(fp)) >= min_distance for fp in filtered_points):
                filtered_points.append(p)
        return filtered_points

    """최종 코너 좌표 반환"""
    def extract(self, map):
        # y, x --> (x, y) 변환
        points = np.where(map == 1)
        raw_corners = list(zip(points[1], points[0]))
        return self.filter_close_corners(raw_corners)
