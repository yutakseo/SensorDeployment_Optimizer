from __future__ import annotations

import numpy as np


def update_positions(
    positions: np.ndarray,
    velocities: np.ndarray,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    """Move particles and constrain their coordinates to the map bounds."""
    positions += velocities
    positions[:, :, 0] = np.clip(positions[:, :, 0], 0, int(width) - 1)
    positions[:, :, 1] = np.clip(positions[:, :, 1], 0, int(height) - 1)
    return positions
