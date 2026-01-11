# OuterDeployment/HarrisCorner.py
import cv2
import numpy as np


class HarrisCorner:
    def __init__(self, MAP):
        self.map_data = np.asarray(MAP, dtype=np.uint8)
        self.blur_map = None
        self.harris_map = None
        self.nms_map = None
        self.corners = []
        self.count = 0

    def gaussianBlur(self, grid: np.ndarray, ksize=(9, 9), sigX=0, sigY=0) -> np.ndarray:
        return cv2.GaussianBlur(np.asarray(grid, dtype=np.uint8), ksize, sigX, sigY)

    def harrisCorner(self, grid: np.ndarray, block_size=3, ksize=3, k=0.05, threshold=0.01) -> np.ndarray:
        resp = cv2.cornerHarris(np.asarray(grid, dtype=np.float32), block_size, ksize, k)
        thr = float(threshold) * float(resp.max() if resp.size else 0.0)
        resp[resp < thr] = 0
        return resp

    def non_max_suppression(self, resp: np.ndarray, thr: float = 0.01, dilate_size: int = 5) -> np.ndarray:
        if resp.size == 0:
            return np.zeros_like(resp, dtype=np.uint8)
        dil = cv2.dilate(resp, np.ones((dilate_size, dilate_size), np.uint8))
        det = np.zeros_like(resp, dtype=np.uint8)
        det[(resp == dil) & (resp > thr)] = 1
        return det

    def extract(self, det: np.ndarray) -> list[tuple[int, int]]:
        y, x = np.where(det == 1)
        return [(int(xi), int(yi)) for xi, yi in zip(x, y)]

    def filter_close(self, points: list[tuple[int, int]], min_dist: int = 5) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        md2 = int(min_dist) * int(min_dist)
        for x, y in points:
            ok = True
            for ox, oy in out:
                dx = x - ox
                dy = y - oy
                if dx * dx + dy * dy < md2:
                    ok = False
                    break
            if ok:
                out.append((int(x), int(y)))
        return out

    def filter_installable(self, points: list[tuple[int, int]], installable_layer: np.ndarray) -> list[tuple[int, int]]:
        mask = (np.asarray(installable_layer) > 0)
        h, w = mask.shape
        return [(x, y) for (x, y) in points if 0 <= x < w and 0 <= y < h and mask[y, x]]
    
    def LMX(self, harris_map, installable_map, nms_ratio=0.1, dilate_size=5, min_dist=5):
        mx = float(harris_map.max() if harris_map.size else 0.0)
        thr = float(nms_ratio) * mx
        self.nms_map = self.non_max_suppression(resp=harris_map, thr=thr, dilate_size=int(dilate_size))

        # 4) corners -> close filter -> installable filter
        pts = self.extract(self.nms_map)
        pts = self.filter_close(pts, min_dist=int(min_dist))
        if installable_map is not None:
            return self.filter_installable(pts, installable_map)

    def run(
        self,
        grid: np.ndarray,
        *,
        installable_layer: np.ndarray | None = None,
        blockSize: int = 3,
        ksize: int = 3,
        k: float = 0.05,
        threshold: float = 0.10,
        dilate_size: int = 5,
        nms_ratio: float = 0.10,
        min_dist: int = 5,
    ) -> list[tuple[int, int]]:
        # 1) blur
        self.blur_map = self.gaussianBlur(grid)

        # 2) harris
        self.harris_map = self.harrisCorner(
            self.blur_map,
            block_size=int(blockSize),
            ksize=int(ksize),
            k=float(k),
            threshold=float(threshold),
        )

        # 3) NMS
        mx = float(self.harris_map.max() if self.harris_map.size else 0.0)
        thr = float(nms_ratio) * mx
        self.nms_map = self.non_max_suppression(self.harris_map, thr=thr, dilate_size=int(dilate_size))

        # 4) corners -> close filter -> installable filter
        pts = self.extract(self.nms_map)
        pts = self.filter_close(pts, min_dist=int(min_dist))
        if installable_layer is not None:
            pts = self.filter_installable(pts, installable_layer)

        self.corners = pts
        self.count = len(pts)
        return pts
