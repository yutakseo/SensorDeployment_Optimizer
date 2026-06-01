from __future__ import annotations

import random
from typing import List, Sequence, Tuple

import numpy as np

Gene = Tuple[int, int]
Chromosome = List[Gene]
Generation = List[Chromosome]


def initialize_swarm(
    chromosomes: Generation,
    *,
    max_sensors: int,
    min_sensors: int,
    installable_points: Sequence[Gene],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create fixed-size particle positions, velocities, and active counts."""
    if not installable_points:
        raise ValueError("installable_points must not be empty.")

    particle_count = max(1, len(chromosomes))
    positions = np.zeros((particle_count, int(max_sensors), 2), dtype=np.float32)
    velocities = np.random.uniform(-1.0, 1.0, size=positions.shape).astype(np.float32)
    active_counts = np.zeros(particle_count, dtype=np.int32)

    for index, chromosome in enumerate(chromosomes):
        count = max(int(min_sensors), min(int(max_sensors), len(chromosome)))
        active_counts[index] = count

        fill = list(chromosome[:count])
        while len(fill) < int(max_sensors):
            fill.append(random.choice(installable_points))
        positions[index] = np.asarray(fill[: int(max_sensors)], dtype=np.float32)

    return positions, velocities, active_counts
