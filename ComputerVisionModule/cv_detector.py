import cv2
import numpy as np
from Visual import *

class ComputerVision():
    def __init__(self, map):
        self.map = map
        self.map_data = np.array(self.map, dtype=np.uint8)
        self.vis = VisualTool()
        
    def harris_corner(self, block_size, ksize, k):
        self.map_data = cv2.GaussianBlur(self.map_data, (7,7), 0,0)
        self.vis.showJetMap("RESULT", self.map_data)
        self.result = cv2.cornerHarris(self.map_data, block_size, ksize, k)
        max_val = np.max(self.result)
        binary_image = np.zeros_like(self.result, dtype=np.uint8)
        binary_image[self.result >= 0.5*max_val] = 1 
        dst = np.where(binary_image == 1)
        
        return list(zip(dst[0],dst[1]))

