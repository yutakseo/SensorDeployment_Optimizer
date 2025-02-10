import numpy as np
import random
import os
import csv
from datetime import datetime


class SensorGA:
    def __init__(self, map_data, coverage, generations, results_dir,
                 initial_population_size=100, next_population_size=50, candidate_population_size=100):
        """
        SensorGA 클래스: 유전 알고리즘을 기반으로 최적의 센서 배치를 찾는 클래스.

        Parameters:
          - map_data: 2D numpy 배열 (맵 데이터)
          - coverage: 센서 커버리지 (반지름으로 사용)
          - generations: 유전 알고리즘 세대 수
          - results_dir: 결과를 저장할 폴더 경로
          - initial_population_size: 초기 개체군 크기
          - next_population_size: 이후 각 세대에서 선택될 부모 개체 수
          - candidate_population_size: 교배 및 돌연변이를 통해 생성할 후보 개체 수
        """
        self.map_data = np.array(map_data)
        self.coverage = coverage
        self.generations = generations
        self.initial_population_size = initial_population_size
        self.next_population_size = next_population_size
        self.candidate_population_size = candidate_population_size
        self.feasible_positions = set(map(tuple, np.argwhere(self.map_data == 1)))
        self.rows, self.cols = self.map_data.shape

        # 결과 저장 폴더 설정
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

        # CSV 파일 저장 경로 설정
        self.file_path = os.path.join(self.results_dir, "generation_results.csv")
        with open(self.file_path, mode="w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Generation", "Fitness", "Num_Sensors"])

        # 초기 개체군 생성
        self.population = self.initialize_population()

    def initialize_population(self):
        """초기 개체(염색체) 생성"""
        population = []
        for _ in range(self.initial_population_size):
            num_sensors = random.randint(5, 20)
            sensor_positions = random.sample(list(self.feasible_positions), num_sensors)
            chromosome = [coord for pos in sensor_positions for coord in pos]
            population.append(chromosome)
        return population

    def draw_sensor(self, chromosome):
        """센서 커버리지를 적용한 맵을 반환"""
        updated_map = np.array(self.map_data, dtype=int)
        if len(chromosome) % 2 != 0:
            chromosome = chromosome[:-1]
        centers = np.array(chromosome).reshape(-1, 2)
        for center in centers:
            x_center, y_center = center
            x, y = np.ogrid[:self.rows, :self.cols]
            mask = (x - x_center)**2 + (y - y_center)**2 <= self.coverage**2
            updated_map[mask] += 10
        return updated_map

    def fitness_function(self, chromosome):
        """적합도 평가 함수"""
        sensor_map = self.draw_sensor(chromosome)
        num_sensors = len(chromosome) // 2
        coverage_score = np.sum(sensor_map >= 11)
        sensor_counts = (sensor_map - self.map_data) // 10
        overlap_penalty = np.sum(np.maximum(0, sensor_counts - 1)) * 2
        sensor_penalty = num_sensors * 3
        return coverage_score - (sensor_penalty + overlap_penalty)

    def save_generation_results(self, generation, fitness, num_sensors):
        """세대별 적합도 및 센서 개수를 CSV에 저장"""
        with open(self.file_path, mode="a", newline='') as file:
            writer = csv.writer(file)
            writer.writerow([generation, fitness, num_sensors])

    def run(self):
        """유전 알고리즘 실행"""
        population = self.population
        parents = population[:self.next_population_size]
        
        for gen in range(1, self.generations + 1):
            # 새로운 후보 개체 생성
            candidate_offspring = []
            while len(candidate_offspring) < self.candidate_population_size:
                parent1, parent2 = random.sample(parents, 2)
                child = parent1[:len(parent1)//2] + parent2[len(parent2)//2:]  # 간단한 crossover
                candidate_offspring.append(child)
            parents = candidate_offspring[:self.next_population_size]

            best_solution = max(parents, key=self.fitness_function)
            best_fitness = self.fitness_function(best_solution)
            num_sensors = len(best_solution) // 2
            self.save_generation_results(gen, best_fitness, num_sensors)

            # 🔹 **터미널 출력 추가**
            print(f"Generation {gen} | Best Fitness: {best_fitness:.1f} | Num Sensors: {num_sensors}")

        # 최종 결과
        best_solution = max(parents, key=self.fitness_function)

        # 🔹 **최종 결과에서 `feasible_positions` 검토**
        sensor_positions = []
        for i in range(0, len(best_solution), 2):
            if i + 1 < len(best_solution):
                x, y = best_solution[i], best_solution[i + 1]
                if (x, y) in self.feasible_positions:  # ✅ `feasible_positions` 검토 추가
                    sensor_positions.append((y, x))  # (y, x) 순서

        # 🔹 **최종 센서 배치 맵 생성**
        inner_layer = self.map_data.copy()
        for x, y in sensor_positions:
            if (x, y) in self.feasible_positions:  # ✅ `feasible_positions` 검토 추가
                inner_layer[y, x] = 10  # (y, x) 순서로 인덱싱하여 저장

        return inner_layer, sensor_positions
from _VisualModule import VisualTool


if __name__ == "__main__":
    map_data = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ], dtype=int)

    best_sensor_positions = SensorGA(map_data, coverage=5, generations=100).run()
    VisualTool().showJetMap_circle("Final Sensor Placement", map_data, radius=5, sensor_positions=best_sensor_positions)
    print("\n🔍 최종 센서 배치 (좌표 쌍):")
    print(best_sensor_positions)
