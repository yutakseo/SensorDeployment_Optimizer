from __future__ import annotations

import numpy as np


def calculate_acceleration(
    positions: np.ndarray,
    personal_best_positions: np.ndarray,
    global_best_position: np.ndarray,
    *,
    cognitive: float,
    social: float,
) -> np.ndarray:
    """Calculate the cognitive and social acceleration for each particle."""
    cognitive_random = np.random.random(size=positions.shape).astype(np.float32)
    social_random = np.random.random(size=positions.shape).astype(np.float32)
    return (
        float(cognitive) * cognitive_random * (personal_best_positions - positions)
        + float(social) * social_random * (global_best_position[None, :, :] - positions)
    )
