import numpy as np
import random
import copy
import os
import csv
from datetime import datetime

RESULTS_DIR = "__RESULTS__"

class SensorGA:
    def __init__(self, map_data, coverage, generations, pop_size=50):
        """
        SensorGA 클래스: 유전 알고리즘을 기반으로 최적의 센서 배치를 찾는 클래스.

        Parameters:
        - map_data: 2D numpy 배열 (맵 데이터)
        - coverage: 센서 커버리지 (반지름으로 사용)
        - generations: 유전 알고리즘 세대 수
        - pop_size: 초기 개체 수
        """
        self.map_data = np.array(map_data)
        self.coverage = coverage  # ✅ coverage 값을 radius로 사용
        self.generations = generations
        self.population_size = pop_size
        self.feasible_positions = np.argwhere(self.map_data == 1)  # ✅ 맵 내부(1)만 센서 배치 가능
        self.population = self.initialize_population()

        # ✅ 결과 저장 폴더 생성
        now = datetime.now().strftime("%m-%d-%H-%M")
        self.experiment_dir = os.path.join(RESULTS_DIR, now)
        os.makedirs(self.experiment_dir, exist_ok=True)
        self.file_path = os.path.join(self.experiment_dir, "generation_results.csv")

        # ✅ CSV 파일 초기화 (헤더 추가)
        with open(self.file_path, mode="w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Generation", "Fitness", "Num_Sensors"])

    def initialize_population(self):
        """초기 개체(염색체) 생성: 맵 내부(1)에 최소 5개 이상의 센서 배치"""
        population = []
        for _ in range(self.population_size):
            num_sensors = random.randint(5, 20)
            sensor_positions = random.sample(list(map(tuple, self.feasible_positions)), num_sensors)
            chromosome = [coord for pos in sensor_positions for coord in pos]
            population.append(chromosome)

        return population

    def draw_sensor(self, pop):
        """
        센서를 배치하고 중첩된 영역을 카운트하여 맵을 반환
        - pop: 개체 (센서 좌표 리스트)
        """
        updated_map = np.array(self.map_data, dtype=int)  # 기존 맵 복사
        rows, cols = updated_map.shape  

        if len(pop) % 2 != 0:
            pop = pop[:-1]

        centers = np.array(pop).reshape(-1, 2) 
        feasible_set = set(map(tuple, self.feasible_positions)) 
        valid_centers = [c for c in centers if tuple(c) in feasible_set]
        valid_centers = np.array(valid_centers)

        if len(valid_centers) == 0:
            return updated_map

        x, y = np.ogrid[:rows, :cols]
        for cx, cy in valid_centers:
            mask = (x - cx) ** 2 + (y - cy) ** 2 <= self.coverage ** 2 
            updated_map[mask] += 10
        print(updated_map)
        return updated_map

    def fitness_function(self, chromosome):
        """
        적합도 평가:
        - 맵 내부(1) 영역 커버 최대화
        - 센서 개수 최소화 (패널티 적용)
        - 중첩 패널티 적용
        """
        sensor_map = self.draw_sensor(chromosome)  # ✅ 개체의 센서 배치를 시뮬레이션
        num_sensors = len(chromosome) // 2

        coverage_score = np.sum(sensor_map > 0)  # ✅ 커버된 맵 내부 영역 수
        overlap_penalty = np.sum(sensor_map > 10) * 2  # ✅ 중복 센서 패널티
        sensor_penalty = num_sensors * 3  # ✅ 센서 개수 패널티

        fitness = coverage_score - (sensor_penalty + overlap_penalty)

        # ✅ 적합도가 음수면 랜덤하게 센서 추가
        while fitness < 0:
            new_pos = random.choice(self.feasible_positions)
            if new_pos[0] not in chromosome or new_pos[1] not in chromosome:
                chromosome.extend([new_pos[0], new_pos[1]])
                num_sensors += 1
                fitness = coverage_score - (num_sensors * 3 + overlap_penalty)  # 다시 적합도 계산
        print(fitness)
        return fitness

    def run(self):
        """
        ✅ 전체 실행 흐름:
        1. 초기 해(50개 개체) 생성
        2. 각 개체의 적합도를 평가
        3. 세대별 진화 진행
        4. 최적 해 반환
        """
        for gen in range(self.generations):
            fitness_scores = [self.fitness_function(chromosome) for chromosome in self.population]

            # ✅ 상위 개체 선택 (룰렛 휠 방식)
            sorted_population = [x for _, x in sorted(zip(fitness_scores, self.population), reverse=True)]
            top_half = sorted_population[:self.population_size // 2]

            new_population = []

            # ✅ 교배 과정 (랜덤으로 두 부모 선택)
            while len(new_population) < self.population_size:
                parent1, parent2 = random.sample(top_half, 2)
                min_length = min(len(parent1), len(parent2))
                child = []
                for i in range(0, min_length, 2):
                    x_choice = random.choice([parent1[i], parent2[i]])
                    y_choice = random.choice([parent1[i+1], parent2[i+1]])
                    if (x_choice, y_choice) in self.feasible_positions:
                        child.extend([x_choice, y_choice])

                new_population.append(child)

            # ✅ 돌연변이 (30% 확률로 센서 추가/변경)
            for i in range(len(new_population)):
                if random.random() < 0.3:
                    new_pos = random.choice(self.feasible_positions)
                    new_population[i].extend([new_pos[0], new_pos[1]])

            self.population = new_population

            # ✅ 최고 적합도 출력 및 저장
            best_solution = max(self.population, key=self.fitness_function)
            best_fitness = self.fitness_function(best_solution)
            num_sensors = len(best_solution) // 2

            print(f"Generation {gen+1} | Fitness: {best_fitness} | Num Sensors: {num_sensors}")

            # ✅ CSV 파일 저장
            with open(self.file_path, mode="a", newline='') as file:
                writer = csv.writer(file)
                writer.writerow([gen+1, best_fitness, num_sensors])

        return best_solution




# ✅ 테스트용 맵 (22x22 크기)
map_data = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
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
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
])



# ✅ 센서 배치 테스트
ga = SensorGA(map_data, coverage=10, generations=10, pop_size=10)
ga.run()