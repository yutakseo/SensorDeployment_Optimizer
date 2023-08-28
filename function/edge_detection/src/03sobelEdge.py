import cv2
import numpy as np
from test_dataset import *

src = np.array(test_data2, dtype=np.uint8)

sobel = cv2.Sobel(src, cv2.CV_8U, 1, 0, 3)
print(sobel)