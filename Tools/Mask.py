from typing import List, Sequence, Union
import numpy as np

def layer_map(
    map_list: List[List[Union[int, float]]],
    keep_values: Sequence[int],
    *,
    out_type: str = "list"  # "list" or "numpy"
):
    """
    특정 클래스 값들을 하나의 binary layer(0/1)로 변환

    map_list   : 2D python list (H x W)
    keep_values: 유지할 클래스 값들 (ex. [0,2,3])
    return     : 원본 크기의 0/1 binary map
    """
    arr = np.asarray(map_list)
    if arr.ndim != 2:
        raise ValueError(f"map_list must be 2D. Got shape={arr.shape}")

    layer = np.isin(arr, keep_values).astype(np.uint8)

    return layer.tolist() if out_type == "list" else layer
