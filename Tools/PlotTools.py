import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Optional, Union

RESULTS_DIR = "__RESULTS__"

class VisualTool:
    def __init__(self, 
                 save_dir: str = RESULTS_DIR, 
                 show: bool = False,
                 save: bool = True,
                 size: Tuple[int, int] = (6, 4), 
                 stamp_filename: bool = False):
        """
        - save_dir: 결과 루트 디렉터리(기본: __RESULTS__)
        - show: True면 plt.show()
        - save: True면 파일 저장 (폴더 생성도 이때만)
        - stamp_filename: 파일명 뒤에 타임스탬프 붙일지 여부
        """
        self.root_dir = save_dir
        self.show = show
        self.save = save
        self.figsize = size
        self.stamp_filename = stamp_filename

        self.time = datetime.now().strftime("%m-%d-%H-%M")

        # save=True일 때만 기본 출력 디렉터리 경로를 설정
        self.output_dir: Optional[str] = (
            os.path.join(self.root_dir, self.time) if self.save else None
        )
        # 디렉터리는 실제 저장 시점(save_or_show)에서 생성

    # ----------------- 내부 유틸 -----------------
    def _resolve_dir(self, path: Optional[str]) -> str:
        """
        save_path 해석:
        - None: self.output_dir (없으면 __RESULTS__/<timestamp>)
        - 절대경로: 그대로 사용
        - 상대경로: __RESULTS__/path 로 사용
        실제 디렉터리 생성은 save_or_show에서 수행
        """
        if path is None:
            if self.output_dir is not None:
                return self.output_dir
            # save=False로 만들어졌지만, 저장을 시도하면 기본 위치를 만듦
            return os.path.join(self.root_dir, self.time)

        if os.path.isabs(path):
            return path
        return os.path.join(self.root_dir, path)

    def _normalize_image(self, img: Union[np.ndarray, List]) -> np.ndarray:
        """imshow에 안전한 형태(2D or 3채널/4채널)로 캐스팅."""
        arr = np.asarray(img)
        if arr.dtype.kind in ("U", "S", "O"):
            try:
                arr = arr.astype(np.float32)
            except Exception as e:
                raise TypeError(f"map_data must be numeric; failed to cast from {arr.dtype}: {e}")
        if arr.dtype == np.bool_:
            arr = arr.astype(np.uint8)
        if arr.ndim == 2:
            return arr
        if arr.ndim == 3 and arr.shape[2] in (3, 4):
            return arr
        raise ValueError(f"Expected 2D grayscale or 3D RGB(A) array, got shape {arr.shape} and dtype {arr.dtype}")

    def _normalize_positions(self, positions: Optional[List[Tuple[Union[int,float,str], Union[int,float,str]]]]
                             ) -> List[Tuple[float, float]]:
        if not positions:
            return []
        safe = []
        for p in positions:
            try:
                x, y = float(p[0]), float(p[1])
                safe.append((x, y))
            except Exception as e:
                raise TypeError(f"Invalid sensor position {p}: {e}")
        return safe

    def _resolve_cmap(self, cmap):
        if isinstance(cmap, list):
            try:
                return plt.cm.colors.ListedColormap(cmap)
            except Exception as e:
                raise ValueError(f"Invalid cmap list: {e}")
        return cmap  # string or Colormap 객체

    # ----------------- 공개 API -----------------
    def set_output(self, subdir: Optional[str] = None, timestamped: bool = True) -> None:
        """
        기본 저장 위치를 변경(논리적 경로만 갱신).
        - subdir가 None이면 __RESULTS__/<timestamp>
        - subdir가 주어지면 __RESULTS__/<subdir> (상대경로)
        - timestamped=True이면 뒤에 /<timestamp> 추가
        실제 디렉터리 생성은 save_or_show에서 수행.
        """
        if subdir is None:
            base = os.path.join(self.root_dir, self.time)
        else:
            base = os.path.join(self.root_dir, subdir)

        self.output_dir = os.path.join(base, self.time) if timestamped else base

    def showJetMap_circle(self,
                          map_data: Union[np.ndarray, List],
                          sensor_positions: Optional[List[Tuple[Union[int,float,str], Union[int,float,str]]]],
                          title: str = "MAP_with_sensor",
                          radius: float = 45,
                          cmap: Union[str, list] = 'jet',
                          filename: str = "map_with_sensor",
                          save_path: Optional[str] = None) -> None:
        map_data = self._normalize_image(map_data)
        sensor_positions = self._normalize_positions(sensor_positions)
        cmap_custom = self._resolve_cmap(cmap)

        fig, ax = plt.subplots(figsize=self.figsize, dpi=150)
        ax.imshow(map_data, cmap=cmap_custom, interpolation='nearest', origin='upper')
        ax.set_title(title)

        for pos in sensor_positions:
            inner = plt.Circle(pos, radius=radius/5, edgecolor='lime', facecolor='white', alpha=0.1, linewidth=0.02)
            border = plt.Circle(pos, radius=radius/5, edgecolor='lime', facecolor='none', linewidth=0.2)
            center = plt.Circle(pos, radius=0.2, edgecolor='lime', facecolor='lime', linewidth=0.02)
            ax.add_patch(inner); ax.add_patch(border); ax.add_patch(center)

        self.save_or_show(fig, filename, save_path)

    def showJetMap(self,
                   map_data: Union[np.ndarray, List],
                   title: str = "MAP",
                   cmap: Union[str, list] = 'jet',
                   filename: str = "jet_map",
                   save_path: Optional[str] = None) -> None:
        map_data = self._normalize_image(map_data)
        cmap_custom = self._resolve_cmap(cmap)

        fig, ax = plt.subplots(figsize=self.figsize, dpi=150)
        ax.imshow(map_data, cmap=cmap_custom, interpolation='nearest', origin='upper')
        ax.set_title(title)
        self.save_or_show(fig, filename, save_path)

    def save_or_show(self, fig:plt.Figure, filename:str, save_path:Optional[str] = None) -> None:
        # 축/레이블/틱 제거 → 그림만 저장
        for ax in fig.axes:
            ax.set_axis_off()

        # 파일명 타임스탬프 옵션
        fname = f"{filename}_{self.time}.png" if self.stamp_filename else f"{filename}.png"

        # 화면 표시
        if self.show:
            plt.show()

        # 저장
        if self.save:
            dirpath = self._resolve_dir(save_path)
            os.makedirs(dirpath, exist_ok=True)  # save=True일 때만 실제 생성
            outpath = os.path.join(dirpath, fname)
            fig.savefig(outpath, bbox_inches='tight', pad_inches=0)
            print(f"Saved figure: {outpath}")

        if not self.show and not self.save:
            print("Warning: Both show=False and save=False → Nothing will happen.")

        plt.close(fig)
