from typing import List, Optional, Tuple
import os
import numpy as np
import matplotlib.pyplot as plt
import runpy

# 타입 힌트용
GridType = List[List[int]]

# ---------------------------------------------------------
# 1) 파이썬 파일에서 2D 리스트(맵 데이터) 읽어오기
# ---------------------------------------------------------
def load_grid_from_py(py_path: str, var_name: str = "GRID") -> GridType:
    """
    py_path 에 있는 파이썬 파일을 실행해서,
    var_name 이름의 2D 리스트 변수를 읽어온다.
    """
    ns = runpy.run_path(py_path)
    if var_name not in ns:
        raise KeyError(f"{py_path} 안에 '{var_name}' 변수가 없습니다.")
    grid = ns[var_name]
    return grid


# ---------------------------------------------------------
# 1-1) GRID 를 .py 파일로 저장하기
# ---------------------------------------------------------
def save_grid_to_py(
    grid: GridType,
    output_path: str,
    var_name: str = "GRID",
):
    """
    2D 리스트 grid 를 파이썬 파일(.py)로 저장한다.
    """
    dirpath = os.path.dirname(output_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Auto-generated grid\n")
        f.write(f"{var_name} = [\n")
        for row in grid:
            f.write("    " + repr(list(row)) + ",\n")
        f.write("]\n")

    print(f"[INFO] Grid saved to: {output_path} (var_name={var_name})")


# ---------------------------------------------------------
# 2) GRID → RGBA 이미지로 변환
#    ※ show_* = False 이면 '검은색(0,0,0)'으로 마스킹
# ---------------------------------------------------------
def grid_to_rgba(
    grid: GridType,
    show_outside: bool = True,       # 0: 현장외곽
    show_inside: bool = True,        # 1: 현장내부
    show_installable: bool = True,   # 2: 설치가능구역
    show_constraint: bool = True,    # 3: 제약조건영역 (색 채움 여부)
) -> np.ndarray:
    """
    2D 정수 grid 를 RGBA 이미지(높이×너비×4, uint8)로 변환.

    - 기본값은 전체를 검은색(0,0,0,255)으로 채움
    - show_* 가 True 인 값만 회색/흰색 등 의미 있는 색으로 덮어씀
      → 즉, show_* = False 면 그 영역은 '검은색으로 가려진 상태'가 됨.
    """
    arr = np.asarray(grid, dtype=int)
    if arr.ndim != 2:
        raise ValueError("grid 는 2차원 리스트/배열이어야 합니다.")

    H, W = arr.shape
    # 기본은 전체 검은색 불투명 (배경 완전 가림)
    img = np.zeros((H, W, 4), dtype=np.uint8)
    img[..., 3] = 255  # alpha 전부 255

    # 0: 현장 외곽 (진회색) - 보이게 하고 싶을 때만 회색으로
    if show_outside:
        mask = (arr == 0)
        img[mask] = [50, 50, 50, 255]

    # 1: 현장 내부 (흰색)
    if show_inside:
        mask = (arr == 1)
        img[mask] = [255, 255, 255, 255]

    # 2: 설치 가능 구역 (흰색 배경)
    if show_installable:
        mask = (arr == 2)
        img[mask] = [255, 255, 255, 255]

    # 3: 제약 조건 영역 (색을 칠할지 여부는 show_constraint로 제어)
    if show_constraint:
        mask = (arr == 3)
        img[mask] = [255, 255, 255, 255]  # 지금은 내부와 동일한 흰색

    return img


# ---------------------------------------------------------
# 3) 설치가능 구역(값 2)에 빨간 테두리 그리기
# ---------------------------------------------------------
def add_installable_borders(
    ax: plt.Axes,
    grid: GridType,
    show_installable: bool = True,
    linewidth: float = 1.0,
):
    """
    값 2(설치가능 구역)에 대해 빨간 테두리를 그려줌.
    """
    if not show_installable:
        return

    arr = np.asarray(grid, dtype=int)
    H, W = arr.shape

    ys, xs = np.where(arr == 2)

    for y, x in zip(ys, xs):
        rect = plt.Rectangle(
            (x, y), 1, 1,
            fill=False,
            edgecolor="red",
            linewidth=linewidth,
        )
        ax.add_patch(rect)


# ---------------------------------------------------------
# 3-1) 현장 내부(값 1) + 제약조건(값 3)에 검은 테두리
# ---------------------------------------------------------
def add_inside_borders(
    ax: plt.Axes,
    grid: GridType,
    show_inside: bool = True,
    linewidth: float = 0.4,
):
    """
    값 1(현장 내부)과 값 3(제약조건)을 검은 테두리로 그려줌.
    """
    if not show_inside:
        return

    arr = np.asarray(grid, dtype=int)
    H, W = arr.shape

    ys, xs = np.where((arr == 1) | (arr == 3))

    for y, x in zip(ys, xs):
        rect = plt.Rectangle(
            (x, y), 1, 1,
            fill=False,
            edgecolor="black",
            linewidth=linewidth,
        )
        ax.add_patch(rect)


# ---------------------------------------------------------
# 3-2) 우측 하단에 길이 축척 바(45m, 5m 눈금) 그리기
# ---------------------------------------------------------
def add_scale_bar(
    ax: plt.Axes,
    H: int,
    W: int,
    cell_size_m: float = 5.0,   # 1셀 = 5m
    length_m: float = 45.0,     # 전체 스케일바 길이 (45m)
    subdiv_m: float = 5.0,      # 보조 눈금 간격 (5m)
    location: str = "lower right",
):
    cells_per_meter = 1.0 / cell_size_m
    bar_len_cells = length_m * cells_per_meter
    subdiv_cells = subdiv_m * cells_per_meter

    margin_x = 1.0
    margin_y = 1.0
    y_base = H - margin_y

    if location == "lower right":
        x_end = W - margin_x
        x_start = x_end - bar_len_cells
    else:
        x_start = margin_x
        x_end = x_start + bar_len_cells

    ax.plot([x_start, x_end], [y_base, y_base],
            color="black", linewidth=1.5)

    n_sub = int(round(length_m / subdiv_m))
    tick_major = 0.6
    tick_minor = 0.3

    for i in range(n_sub + 1):
        x_tick = x_start + i * subdiv_cells
        is_major = (i == 0) or (i == n_sub)
        h = tick_major if is_major else tick_minor
        ax.plot([x_tick, x_tick], [y_base, y_base - h],
                color="black", linewidth=1.0)

    x_text = (x_start + x_end) / 2.0
    y_text = y_base - (tick_major + 0.4)

    ax.text(
        x_text,
        y_text,
        f"{int(length_m)} m",
        ha="center",
        va="top",
        fontsize=8,
        color="black",
        bbox=dict(
            boxstyle="round,pad=0.2",
            facecolor="white",
            edgecolor="none",
            alpha=0.8,
        ),
    )


# ---------------------------------------------------------
# 4) 맵 시각화 (+ PNG 배경, + 선택적 크롭 + 축척바)
# ---------------------------------------------------------
def visualize_map(
    grid: GridType,
    show_outside: bool = True,
    show_inside: bool = True,
    show_installable: bool = True,
    show_constraint: bool = True,
    figsize=(10, 10),
    save_path: Optional[str] = None,
    background_path: Optional[str] = None,
    background_alpha: float = 1.0,
    grid_alpha: float = 1.0,
    crop_x: Optional[Tuple[int, int]] = None,  # (x_start, x_end)
    crop_y: Optional[Tuple[int, int]] = None,  # (y_start, y_end)
    show_scale_bar: bool = True,
    scale_length_m: float = 45.0,
    scale_subdiv_m: float = 5.0,
):
    """
    - crop_x / crop_y 는 "grid 인덱스" 기준
    - show_* = False 인 영역은 투명 X → 검은색으로 마스킹
    """
    arr_full = np.asarray(grid, dtype=int)
    if arr_full.ndim != 2:
        raise ValueError("grid 는 2차원 리스트/배열이어야 합니다.")

    H_full, W_full = arr_full.shape
    arr = arr_full.copy()

    if crop_x is not None and crop_y is not None:
        x_start, x_end = crop_x
        y_start, y_end = crop_y

        x_start = max(0, min(W_full, x_start))
        x_end   = max(0, min(W_full, x_end))
        y_start = max(0, min(H_full, y_start))
        y_end   = max(0, min(H_full, y_end))

        if x_start >= x_end or y_start >= y_end:
            raise ValueError("crop_x / crop_y 범위가 잘못되었습니다.")

        arr = arr[y_start:y_end, x_start:x_end]

    H, W = arr.shape
    grid_cropped: GridType = arr.tolist()

    # ---- 면적 정보 출력 ----
    CELL_SIZE_M = 5.0
    CELL_AREA_M2 = CELL_SIZE_M * CELL_SIZE_M

    nonzero_mask = (arr != 0)
    nonzero_count = int(nonzero_mask.sum())
    nonzero_area_m2 = nonzero_count * CELL_AREA_M2

    map_width_m = W * CELL_SIZE_M
    map_height_m = H * CELL_SIZE_M
    map_area_m2 = map_width_m * map_height_m

    map_width_km = map_width_m / 1000.0
    map_height_km = map_height_m / 1000.0
    map_area_km2 = map_area_m2 / 1_000_000.0
    map_area_ha  = map_area_m2 / 10_000.0

    print("========== MAP INFO (Current View) ==========")
    print(f"- Grid size (cells): width={W}, height={H}")
    print(f"- Physical size (current view):")
    print(f"    • Width : {map_width_m:.2f} m  ({map_width_km:.3f} km)")
    print(f"    • Height: {map_height_m:.2f} m  ({map_height_km:.3f} km)")
    print(f"- Full rectangular area of current view:")
    print(f"    • {map_area_m2:.2f} m^2")
    print(f"    • {map_area_km2:.6f} km^2")
    print(f"    • {map_area_ha:.2f} ha")
    print(f"- Non-zero (1,2,3) cell area in current view:")
    print(f"    • cells: {nonzero_count}")
    print(f"    • area : {nonzero_area_m2:.2f} m^2")
    print("================================")

    # 2) GRID → RGBA 이미지
    rgba = grid_to_rgba(
        grid_cropped,
        show_outside=show_outside,
        show_inside=show_inside,
        show_installable=show_installable,
        show_constraint=show_constraint,
    ).astype(float)

    rgba[..., 3] *= np.clip(grid_alpha, 0.0, 1.0)
    rgba = rgba.astype(np.uint8)

    fig, ax = plt.subplots(figsize=figsize)

    # 3) PNG 배경 먼저 그리기
    if background_path is not None:
        bg_full = plt.imread(background_path)

        if bg_full.ndim == 3 and bg_full.shape[2] == 4:
            bg_full = bg_full[..., :3]

        H_img, W_img = bg_full.shape[:2]
        sx = W_img / W_full
        sy = H_img / H_full

        if crop_x is not None and crop_y is not None:
            x_start, x_end = crop_x
            y_start, y_end = crop_y

            x_start = max(0, min(W_full, x_start))
            x_end   = max(0, min(W_full, x_end))
            y_start = max(0, min(H_full, y_start))
            y_end   = max(0, min(H_full, y_end))

            px0 = int(round(x_start * sx))
            px1 = int(round(x_end   * sx))
            py0 = int(round(y_start * sy))
            py1 = int(round(y_end   * sy))

            bg = bg_full[py0:py1, px0:px1]
        else:
            bg = bg_full

        ax.imshow(
            bg,
            origin="upper",
            extent=(0, W, H, 0),
            alpha=background_alpha,
            interpolation="bilinear",
        )

    # 4) 그리드 RGBA 얹기
    ax.imshow(
        rgba,
        origin="upper",
        extent=(0, W, H, 0),
        interpolation="none",
    )

    # 5) 테두리 + 축척바
    add_inside_borders(ax, grid_cropped, show_inside=show_inside)
    add_installable_borders(ax, grid_cropped, show_installable=show_installable)

    if show_scale_bar:
        add_scale_bar(
            ax,
            H=H,
            W=W,
            cell_size_m=CELL_SIZE_M,
            length_m=scale_length_m,
            subdiv_m=scale_subdiv_m,
            location="lower right",
        )

    # 6) 축/테두리 정리
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)

    if save_path is not None:
        fig.savefig(
            save_path,
            dpi=600,
            bbox_inches="tight",
            pad_inches=0,
        )
        print(f"[INFO] Figure saved to: {save_path}")

    plt.show()
    plt.close(fig)


# ---------------------------------------------------------
# 5) (옵션) 맵 자르기 (크롭된 grid 자체가 필요할 때)
# ---------------------------------------------------------
def crop_grid(
    grid: GridType,
    x_start: int,
    x_end: int,
    y_start: int,
    y_end: int,
) -> Tuple[GridType, Tuple[int, int]]:
    """
    원본 grid 에서 [x_start:x_end], [y_start:y_end] 구간만 잘라낸
    서브 그리드를 반환한다.
    """
    arr = np.asarray(grid)
    if arr.ndim != 2:
        raise ValueError("grid 는 2차원 리스트/배열이어야 합니다.")

    H, W = arr.shape

    x_start_clamped = max(0, min(W, x_start))
    x_end_clamped   = max(0, min(W, x_end))
    y_start_clamped = max(0, min(H, y_start))
    y_end_clamped   = max(0, min(H, y_end))

    if x_start_clamped >= x_end_clamped or y_start_clamped >= y_end_clamped:
        raise ValueError("잘라낼 영역이 없습니다. (start/end 범위를 확인하세요)")

    sub = arr[y_start_clamped:y_end_clamped, x_start_clamped:x_end_clamped]
    cropped_grid: GridType = sub.tolist()

    origin = (x_start_clamped, y_start_clamped)
    return cropped_grid, origin
