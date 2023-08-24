import time
from matplotlib import pyplot as plt
from TEST_DATASET import *
from TEST_DATASET_ANSWER import *
from createArray import *
from cornerDetector import harris_corner
from modelEvaluation import evaluation



src = test_data4
dst = harris_corner(src, 2, 3, 0.01)



#코너탐색의 결과를 시각화
visual_array("src", src)
visual_array("harris result", dst)

plt.imshow(dst, cmap='jet')  # jet colormap을 사용하여 색상을 조정
#plt.colorbar()
plt.show()

visual_array("bin",binary_corner(dst))


evaluation(test_data1_answer, binary_corner(dst))
