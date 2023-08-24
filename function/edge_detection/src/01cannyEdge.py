import cv2
import numpy as np
from test_dataset import *


src = np.array(test_data2, dtype=np.uint8)


canny = cv2.Canny(src, 0, 1)
canny = canny / 255

print(canny)