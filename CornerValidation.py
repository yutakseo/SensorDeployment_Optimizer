from _HarrisCorner.cv_detector import HarrisCorner
from __MAPS__.validation_maps.bot_uav import *
from Visual import *
from mpl_toolkits.mplot3d import Axes3D

data = MAP
tool = VisualTool()
instance = HarrisCorner(MAP)
tool.showJetMap("Original", np.array(MAP))
#가우시안 블러 적용
#tool.showJetMap(instance.Blur(MAP))
#tool.showJetMap("Gaussian Map",instance.gaussianBlur(MAP))
#헤리스 코너만 탐색
#tool.showJetMap(instance.harrisCorner(MAP))
#가우시안 적용 후, 헤리스 코너 탐색
#tool.showBinaryMap("Harris after gaussian",instance.harrisCorner(instance.gaussianBlur(MAP)))
#tool.showJetMap("Harris after gaussian",instance.harrisCorner_non_dilated(instance.gaussianBlur(MAP)))
#print(HarrisCorner(MAP).extract())


"""3D 모델링"""
matrix = np.array(instance.harrisCorner_non_dilated(instance.gaussianBlur(MAP)))
# 3D 플롯 생성
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# x, y, z 값 생성 (행렬의 각 인덱스를 좌표로 사용)
x, y= np.indices(matrix.shape)
z = matrix
# 행렬 값을 z 값으로 사용하여 플로팅
ax.scatter(x, y, z, c=matrix.flatten(), cmap='viridis')
# 레이블 설정
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.show()
