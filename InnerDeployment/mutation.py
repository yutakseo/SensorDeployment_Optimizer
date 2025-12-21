import random

def mutate(self, chromosome, mutation_rate=0.2):
    """돌연변이 연산: 적합도 평가 후 센서 추가 또는 삭제"""
    # ⚠️ 짝수 개수 유지 (좌표 쌍)
    if len(chromosome) % 2 != 0:
        chromosome = chromosome[:-1]

    current_fitness, coverage_score = self.fitness_function(chromosome)
    sensor_map = self.draw_sensor(chromosome)  # 맵 시뮬레이션 실행
    sensor_counts = (sensor_map - self.map_data) // 10

    # ✅ 커버리지가 부족한 영역 찾기
    uncovered_positions = [pos for pos in self.feasible_positions if sensor_map[pos] < 11]

    # ✅ 중첩된 센서 찾기 (인덱스 검사 추가)
    redundant_sensors = []
    for i in range(0, len(chromosome) - 1, 2):  # ✅ 인덱스 초과 방지
        x, y = chromosome[i], chromosome[i + 1]
        if 0 <= x < self.rows and 0 <= y < self.cols:  # ✅ 인덱스 검사 추가
            if sensor_counts[x, y] > 1:
                redundant_sensors.append((x, y))

    if random.random() < mutation_rate:
        # ✅ 1) 센서 추가 (커버되지 않은 영역이 존재할 경우)
        if uncovered_positions:
            chromosome = self.add_sensor(chromosome, uncovered_positions)

        # ✅ 2) 센서 삭제 (중첩된 센서가 많을 경우)
        elif redundant_sensors:
            chromosome = self.remove_sensor(chromosome, redundant_sensors)

    return chromosome