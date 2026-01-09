#OuterDeployment/HarrisCorner.py
import cv2
import numpy as np


class HarrisCorner():
    def __init__(self, MAP):
        self.map_data: np.ndarray = np.array(MAP, dtype=np.uint8)
        self.gaussian_map: np.ndarray | None = None
        self.harris_response: np.ndarray | None = None
        self.nms_processed: np.ndarray | None = None
        self.corner_positions: list | None = []
        self.total_sensors: int = 0

    # Apply Gaussian blur to the map
    def gaussianBlur(
        self,
        map: np.ndarray,
        ksize: tuple = (9, 9),
        sigX: float = 0,
        sigY: float = 0
    ) -> np.ndarray:
        blurred_map = cv2.GaussianBlur(
            src=np.array(map, dtype=np.uint8),
            ksize=ksize,
            sigmaX=sigX,
            sigmaY=sigY
        )
        return blurred_map

    # Non-Maximum Suppression
    def non_max_suppression(
        self,
        response: np.ndarray,
        nmx_threshold: float = 0.01,
        dilate_size: int = 5
    ) -> np.ndarray:
        dilated = cv2.dilate(response, np.ones((dilate_size, dilate_size), np.uint8))
        det = np.zeros_like(dilated)
        det[(response == dilated) & (response > nmx_threshold)] = 1
        return det

    # Harris corner detection
    def harrisCorner(
        self,
        map: np.ndarray,
        block_size: int = 3,
        ksize: int = 3,
        k: float = 0.05,
        threshold: float = 0.01
    ) -> np.ndarray:
        harris_response = cv2.cornerHarris(
            src=np.array(map, dtype=np.float32),
            blockSize=block_size,
            ksize=ksize,
            k=k
        )
        harris_threshold = threshold * harris_response.max()
        harris_response[harris_response < harris_threshold] = 0
        return harris_response

    # return final corner coordinates
    def extract(self, map: np.ndarray) -> list[tuple[int, int]]:
        points = np.where(map == 1)
        raw_corners = [(int(x), int(y)) for x, y in zip(points[1], points[0])]
        return raw_corners

    # filter out close corners based on Euclidean distance
    def filter_close_corners(
        self,
        points: list[tuple[int, int]],
        min_distance: int = 5
    ) -> list[tuple[int, int]]:
        filtered_points: list[tuple[int, int]] = []
        for p in points:
            if all(np.linalg.norm(np.array(p) - np.array(fp)) >= min_distance for fp in filtered_points):
                filtered_points.append((int(p[0]), int(p[1])))
        return filtered_points

    # run (installable 후보정 통합: close 정리 후 -> installable 필터)
    def run(
        self,
        map: np.ndarray,
        installable_layer: np.ndarray | None = None,  # ✅ 추가
        blockSize: int = 3,
        ksize: int = 3,
        k: float = 0.05,
        dilate_size: int = 5
    ) -> list:
        # 1) Apply Gaussian blur
        self.gaussian_map = self.gaussianBlur(map=map)

        # 2) Harris corner detection
        self.harris_response = self.harrisCorner(
            map=self.gaussian_map,
            block_size=blockSize,
            ksize=ksize,
            k=k,
            threshold=0.1
        )
        nms_threshold = (0.1 * self.harris_response.max())

        # 3) Non-Maximum Suppression
        self.nms_processed = self.non_max_suppression(
            response=self.harris_response,
            nmx_threshold=nms_threshold,
            dilate_size=dilate_size
        )

        # 4-1) Extract corners
        self.corner_positions = self.extract(map=self.nms_processed)

        # 4-2) Filter close corners
        self.corner_positions = self.filter_close_corners(self.corner_positions, min_distance=5)
        if installable_layer is not None:
            installable = np.asarray(installable_layer).astype(bool)  # True = installable
            H, W = installable.shape

            self.corner_positions = [
                (x, y) for (x, y) in self.corner_positions
                if 0 <= x < W and 0 <= y < H and installable[y, x]
            ]

        # 4-4) Count total number of corners
        self.total_sensors = len(self.corner_positions)
        return self.corner_positions
