import cv2
import numpy as np
from test_dataset import *

src = np.array(test_data2, dtype=np.uint8)

laplacian = cv2.Laplacian(src, cv2.CV_8U, ksize=3)

print(laplacian)