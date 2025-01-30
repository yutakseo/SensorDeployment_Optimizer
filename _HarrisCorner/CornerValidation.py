from _HarrisCorner.cv_detector import HarrisCorner
from __MAPS__.validation_maps.Naju_MAP import *
from Visual import *
from mpl_toolkits.mplot3d import Axes3D
from Visual import *
from _HarrisCorner.cv_detector import *


data = MAP
tool = VisualTool()
instance = HarrisCorner(MAP)

#그라운드 맵
#tool.showJetMap("GroundTruth",data)

#tool.showJetMap("Original", np.array(MAP))

#평균값 블러 적용
#tool.showJetMap(instance.Blur(MAP))

#가우시안 블러 적용
#tool.showJetMap("Gaussian Map",instance.gaussianBlur(MAP))

#헤리스 코너만 탐색
#tool.showJetMap("Harris Corner Result",instance.harrisCorner(MAP))
#tool.showBinaryMap("Harris Corner Result",instance.onlyHarris(MAP))

#가우시안 적용 후, 헤리스 코너 탐색
#tool.showBinaryMap("Harris after gaussian",instance.harrisCorner(instance.gaussianBlur(MAP)))
#tool.showBinaryMap("Harris after gaussian",instance.harrisCorner_non_dilated(instance.gaussianBlur(MAP)))

#print(HarrisCorner(MAP).extract())



positions = [(37,5), (77,20), (123,32), (178,57), (210,72), (160,63), (110,52), (95,62)]

vis = VisualTool()
vis.showJetMap_circle("RESULT", data, 45, sensor_positions=positions)






"""
# 두 점 집합의 위치 차이 계산 (유클리드 거리)
def compute_euclidean_distance(A, B):
    return np.linalg.norm(A - B, axis=1)

filter_sizes = []
gaussian_sizes = []
mean_distances = []
# 최소 평균 유클리드 거리 및 필터 크기, 가우시안 마스크 크기 추적
min_distance = float('inf')
best_filter_size = None
best_gaussian_size = None

for i in range(3, 50, 2):  # Harris Corner 필터 크기
    for j in range(1, 50, 2):  # Gaussian Mask 크기
        print(f"Harris Corner Filter Size : ({i},{i})")
        print(f"Gaussian Mask Size : ({j},{j})")

        # Harris Corner 적용 후 결과 계산
        blurred_map = instance.gaussianBlur(MAP, ksize=(j, j))
        B = instance.harrisCorner(blurred_map)

        # 평균 유클리드 거리 계산
        distances = compute_euclidean_distance(true_map, B)
        mean_distance = np.mean(distances)

        print(f"각 점 간의 평균 유클리드 거리: {mean_distance}\n")
         # 데이터 저장
        filter_sizes.append(i)
        gaussian_sizes.append(j)
        mean_distances.append(mean_distance)
        
        # 최소 평균 거리 추적
        if mean_distance < min_distance:
            min_distance = mean_distance
            best_filter_size = (i, i)
            best_gaussian_size = (j, j)

# 결과 출력
print(f"\n\n최소 평균 유클리드 거리: {min_distance}")
print(f"최적 Harris Corner Filter Size: {best_filter_size}")
print(f"최적 Gaussian Mask Size: {best_gaussian_size}")
        

    






"""