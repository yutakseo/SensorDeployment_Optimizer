import random

def initialize_population(
    input_map,
    population_size: int,
    corner_positions: list,
    min_sensors: int = 1,
    max_sensors: int | None = None,
) -> list:
    map_height = len(input_map)
    map_width = len(input_map[0]) if map_height > 0 else 0

    corner_set = set(corner_positions)

    all_positions = [
        (x, y)
        for y in range(map_height)
        for x in range(map_width)
        if (x, y) not in corner_set and int(input_map[y][x]) == 1
    ]

    if not all_positions:
        raise ValueError("No valid sensor positions available")

    if max_sensors is None:
        max_sensors = len(all_positions)

    if min_sensors > max_sensors:
        raise ValueError("min_sensors must be <= max_sensors")

    population = []
    for _ in range(population_size):
        k = random.randint(min_sensors, max_sensors)
        individual = random.sample(all_positions, k)
        population.append(individual)

    return population
