import numpy as np

"""적합도 평가 함수"""
def fitness_function(self, chromosome):
    sensor_map = self.draw_sensor(chromosome)
    num_sensors = len(chromosome) // 2
    coverage_score = np.sum(sensor_map >= 11)
    sensor_counts = (sensor_map - self.map_data) // 10
    overlap_penalty = np.sum(np.maximum(0, sensor_counts - 1)) * 2
    sensor_penalty = num_sensors * 3
    return coverage_score - (sensor_penalty + overlap_penalty), coverage_score