import numpy as np
import random
import os
import csv
from datetime import datetime
from _VisualModule import VisualTool

RESULTS_DIR = "__RESULTS__"

class SensorGA:
    def __init__(self, map_data, coverage, generations, 
                 initial_population_size=100, next_population_size=50, candidate_population_size=100):
        """
        SensorGA 클래스: 유전 알고리즘을 기반으로 최적의 센서 배치를 찾는 클래스.

        Parameters:
          - map_data: 2D numpy 배열 (맵 데이터)
          - coverage: 센서 커버리지 (반지름으로 사용)
          - generations: 유전 알고리즘 세대 수
          - initial_population_size: 초기 개체군 크기 (예, 100)
          - next_population_size: 이후 각 세대에서 선택될 부모(다음 세대) 개체 수 (예, 50)
          - candidate_population_size: 교배 및 돌연변이를 통해 생성할 후보 offspring 개체 수 (예, 100)
        """
        self.map_data = np.array(map_data)
        self.coverage = coverage
        self.generations = generations
        self.initial_population_size = initial_population_size
        self.next_population_size = next_population_size
        self.candidate_population_size = candidate_population_size
        self.feasible_positions = np.argwhere(self.map_data == 1)
        self.rows, self.cols = self.map_data.shape

        # 초기 개체군 생성 (크기: initial_population_size)
        self.population = self.initialize_population()

        # 결과 저장 폴더 및 CSV 파일 설정
        now = datetime.now().strftime("%m-%d-%H-%M")
        self.experiment_dir = os.path.join(RESULTS_DIR, now)
        os.makedirs(self.experiment_dir, exist_ok=True)
        self.file_path = os.path.join(self.experiment_dir, "generation_results.csv")
        with open(self.file_path, mode="w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Generation", "Fitness", "Num_Sensors"])

    def initialize_population(self):
        """초기 개체(염색체) 생성: 맵 내부(1)에서 최소 5개 이상의 센서를 배치.
           각 염색체는 (x, y) 좌표 쌍의 1D 리스트입니다.
        """
        population = []
        for _ in range(self.initial_population_size):
            num_sensors = random.randint(5, 20)
            sensor_positions = random.sample(list(map(tuple, self.feasible_positions)), num_sensors)
            chromosome = [coord for pos in sensor_positions for coord in pos]
            population.append(chromosome)
        return population

    def draw_sensor(self, chromosome):
        """
        주어진 개체(chromosome)를 기반으로 센서 커버리지를 적용한 맵을 반환.
        각 센서 중심을 기준으로, 반지름(self.coverage) 내의 영역에 대해 10씩 더합니다.
        """
        updated_map = np.array(self.map_data, dtype=int)
        if len(chromosome) % 2 != 0:
            chromosome = chromosome[:-1]
        centers = np.array(chromosome).reshape(-1, 2)
        feasible_set = set(map(tuple, self.feasible_positions))
        valid_centers = [tuple(c) for c in centers if tuple(c) in feasible_set]
        for center in valid_centers:
            # 센서 설치 시 해당 위치의 커버리지 영역에 +10을 더함
            x_center, y_center = center
            x, y = np.ogrid[:self.rows, :self.cols]
            mask = (x - x_center)**2 + (y - y_center)**2 <= self.coverage**2
            updated_map[mask] += 10
        return updated_map

    def fitness_function(self, chromosome):
        """
        적합도 평가:
          - coverage_score: 센서가 맵 내부(1)에서 덮은 셀의 수 (셀 값이 10 이상이면 덮인 것으로 판단)
          - sensor_penalty: 센서 개수에 대해 3점씩 패널티
          - overlap_penalty: 한 셀에서 센서가 중복 배치된 경우 초과 배치된 횟수당 2점씩 패널티

        최종 적합도 = coverage_score - (sensor_penalty + overlap_penalty)
        """
        sensor_map = self.draw_sensor(chromosome)
        num_sensors = len(chromosome) // 2
        coverage_score = np.sum(sensor_map >= 11)
        sensor_counts = (sensor_map - self.map_data) // 10
        overlap_penalty = np.sum(np.maximum(0, sensor_counts - 1)) * 2
        sensor_penalty = num_sensors * 3
        return coverage_score - (sensor_penalty + overlap_penalty)

    def selection_fixed(self, population, num):
        """
        룰렛 휠 방식을 이용해 주어진 population에서 num개의 개체를 선택합니다.
        """
        fitness_scores = np.array([self.fitness_function(chromo) for chromo in population])
        total_fitness = sum(fitness_scores)
        if total_fitness == 0:
            return random.sample(population, num)
        selection_probs = fitness_scores / total_fitness
        selected_indices = random.choices(range(len(population)), weights=selection_probs, k=num)
        return [population[i] for i in selected_indices]

    def crossover(self, parent1, parent2):
        """
        부모 간 거리 기반 교배:
         - 부모1과 부모2의 센서 좌표 간 거리를 계산하여 새로운 위치를 생성합니다.
         - 각 좌표에 대해, new_coord = x1 + w * (x2 - x1) (w는 베타 분포에 의해 결정)
        """
        min_length = min(len(parent1), len(parent2))
        child = []
        for i in range(0, min_length, 2):
            x1, y1 = parent1[i], parent1[i+1]
            x2, y2 = parent2[i], parent2[i+1]
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            a_x = max(0.1, 2 - dx / (self.rows - 1))
            w_x = random.betavariate(a_x, a_x)
            new_x = int(x1 + w_x * (x2 - x1))
            a_y = max(0.1, 2 - dy / (self.cols - 1))
            w_y = random.betavariate(a_y, a_y)
            new_y = int(y1 + w_y * (y2 - y1))
            child.extend([new_x, new_y])
        while len(child) < 6:
            new_pos = random.choice(self.feasible_positions)
            child.extend([new_pos[0], new_pos[1]])
        return child

    def sensor_add_delete_mutation(self, chromosome, add_prob=0.3, remove_distance_threshold=2):
        """
        센서 추가/삭제 돌연변이:
         - 서로 너무 가까운 센서를 삭제하고,
         - 센서가 맵의 배치 가능한 영역을 충분히 커버하지 못하면, 미커버 영역 중 하나의 센서를 추가합니다.
         - 새 센서를 추가할 때는 염색체의 맨 앞(처음 두 자리)에 삽입합니다.
        """
        # 1. 센서 좌표 추출
        sensors = [(chromosome[i], chromosome[i+1]) for i in range(0, len(chromosome), 2)]
        # 2. 너무 가까운 센서 제거
        new_sensors = []
        for sensor in sensors:
            if not any(((sensor[0]-ns[0])**2 + (sensor[1]-ns[1])**2)**0.5 < remove_distance_threshold for ns in new_sensors):
                new_sensors.append(sensor)
        # 3. 현재 센서들로 구성된 염색체 및 센서 맵 계산
        new_chromosome = [coord for sensor in new_sensors for coord in sensor]
        sensor_map = self.draw_sensor(new_chromosome)
        # 4. 미커버 영역 탐색
        uncovered_positions = []
        for pos in self.feasible_positions:
            if sensor_map[pos[0], pos[1]] < 11:
                uncovered_positions.append((pos[0], pos[1]))
        # 5. 미커버 영역이 존재하면 add_prob 확률에 따라 새 센서를 염색체 맨 앞에 추가
        if uncovered_positions and random.random() < add_prob:
            new_sensor = random.choice(uncovered_positions)
            new_sensors.insert(0, new_sensor)
        # 6. 새로운 센서 리스트를 1차원 염색체로 변환하여 반환
        new_chromosome = [coord for sensor in new_sensors for coord in sensor]
        return new_chromosome

    def mutation(self, chromosome, population_fitness):
        """
        돌연변이 연산:
         - 적합도가 하위 50%에 해당하는 개체에 대해서만 센서 추가/삭제 돌연변이를 적용하고,
         - 상위 50% 개체는 변화 없이 그대로 반환합니다.
        """
        median_fitness = np.median(population_fitness)
        if self.fitness_function(chromosome) < median_fitness:
            return self.sensor_add_delete_mutation(chromosome, add_prob=0.3, remove_distance_threshold=2)
        else:
            return chromosome

    def save_generation_results(self, generation, fitness, num_sensors):
        """세대별 적합도 및 센서 개수를 CSV에 저장"""
        with open(self.file_path, mode="a", newline='') as file:
            writer = csv.writer(file)
            writer.writerow([generation, fitness, num_sensors])

    def run(self):
        """
        알고리즘 진행:
         - 초기 세대는 100개 개체로 생성됩니다.
         - 그 후, selection_fixed()를 통해 부모(다음 세대) 개체를 50개로 줄입니다.
         - 각 세대마다 현재 부모 개체(50개)로부터 후보 offspring 100개를 생성한 후,
           selection_fixed()를 통해 50개 개체를 다음 세대 부모로 선정합니다.
         - 최종적으로, 최종 부모 집단 중 최고 해를 기준으로 내부 layer와 inner points를 계산하여 반환합니다.
           - inner points는 (y, x) 순서의 센서 좌표입니다.
           - inner layer는 원본 맵 데이터에 각 센서 위치에 대해 +10을 적용한 결과입니다.
        """
        population = self.population  # 초기 100개
        parents = self.selection_fixed(population, self.next_population_size)
        for gen in range(1, self.generations):
            candidate_offspring = []
            while len(candidate_offspring) < self.candidate_population_size:
                parent1, parent2 = random.sample(parents, 2)
                child = self.crossover(parent1, parent2)
                # 돌연변이 적용 (후보 offspring에 대해 fitness 계산)
                child = self.mutation(child, [self.fitness_function(ch) for ch in candidate_offspring] if candidate_offspring else [0])
                candidate_offspring.append(child)
            parents = self.selection_fixed(candidate_offspring, self.next_population_size)
            best_solution = max(parents, key=self.fitness_function)
            best_fitness = self.fitness_function(best_solution)
            num_sensors = len(best_solution) // 2
            print(f"Generation {gen+1} | Best Fitness: {best_fitness} | Num Sensors: {num_sensors}")
            self.save_generation_results(gen+1, best_fitness, num_sensors)
        
        # 최종 부모 집단 중 최고 해를 구함
        best_solution = max(parents, key=self.fitness_function)
        sensor_positions = []
        for i in range(0, len(best_solution), 2):
            if i+1 < len(best_solution):
                x, y = best_solution[i], best_solution[i+1]
                if 0 <= x < self.map_data.shape[0] and 0 <= y < self.map_data.shape[1]:
                    # 센서 좌표를 (y, x) 순서로 저장
                    sensor_positions.append((y, x))
        # inner layer: 원본 맵 데이터를 복사한 후, 센서 설치 위치에 +10 적용
        inner_layer = self.map_data.copy().astype(int)
        for pos in sensor_positions:
            inner_layer[pos[0], pos[1]] = 10
        return inner_layer, sensor_positions

