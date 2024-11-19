from _HarrisCorner.cv_detector import HarrisCorner
from __MAPS__.validation_maps.bot_uav import *
from Visual import *


data = MAP
tool = VisualTool()
instance = HarrisCorner(MAP)
tool.showJetMap("Original", np.array(MAP))
#가우시안 블러 적용
#tool.showJetMap(instance.Blur(MAP))
#tool.showJetMap(instance.gaussianBlur(MAP))
#헤리스 코너만 탐색
#tool.showJetMap(instance.harrisCorner(MAP))
#가우시안 적용 후, 헤리스 코너 탐색
tool.showBinaryMap("Harris after gaussian",instance.harrisCorner(instance.gaussianBlur(MAP)))
#print(HarrisCorner(MAP).extract())