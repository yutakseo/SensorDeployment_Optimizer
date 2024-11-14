from _HarrisCorner.cv_detector import HarrisCorner
from __MAPS__.validation_maps.top_uav import *
from Visual import *


data = MAP
tool = VisualTool()
instance = HarrisCorner(MAP)
tool.showJetMap(np.array(MAP))
#가우시안 블러 적용
tool.showJetMap(instance.gaussianBlur(MAP))
#헤리스 코너만 탐색
tool.showJetMap(instance.harrisCorner(MAP))
#가우시안 적용 후, 헤리스 코너 탐색
tool.showJetMap(instance.harrisCorner(instance.gaussianBlur(MAP)))
#print(HarrisCorner(MAP).extract())