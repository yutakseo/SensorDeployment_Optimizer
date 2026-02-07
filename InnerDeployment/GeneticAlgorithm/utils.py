from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np

Gene = Tuple[int, int]


def to_int_pairs(points: Iterable[Sequence[int]]) -> List[Gene]:
    return [(int(p[0]), int(p[1])) for p in points]


def to_bool_map(m) -> np.ndarray:
    return (np.asarray(m) > 0)

