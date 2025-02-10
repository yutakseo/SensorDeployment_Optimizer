import numpy as np
import random
import copy
import os
import csv
from datetime import datetime

RESULTS_DIR = "__RESULTS__"

class SensorGA:
    def __init__(self, map_data, coverage, generations, pop_size=50):
        self.map_data = np.array(map_data)
        self.coverage = coverage
        self.generations = generations
        self.feasible_positions = np.argwhere(self.map_data == 1)  # ✅ 맵 내부(1)만 센서 배치 가능
        self.population_size = pop_size
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
        """초기 개체(염색체) 생성: 최소 5개 이상의 센서 포함"""
        population = []
        for _ in range(self.population_size):
            num_sensors = random.randint(5, 20)  # ✅ 최소 5개 이상의 센서 배치
            sensor_positions = random.sample(self.feasible_positions.tolist(), num_sensors)
            chromosome = [coord for pos in sensor_positions for coord in pos]  # (x1, y1, x2, y2, ...)
            population.append(chromosome)
        return population

    def fitness_function(self, chromosome):
        """적합도 평가: 맵 내부(1) 영역 커버 최대화 + 센서 개수 최소화 + 중복 패널티 적용"""
        sensor_coverage = np.zeros_like(self.map_data)
        num_sensors = len(chromosome) // 2

        for i in range(0, len(chromosome), 2):
            x, y = chromosome[i], chromosome[i + 1]
            if 0 <= x < self.map_data.shape[0] and 0 <= y < self.map_data.shape[1]:
                sensor_coverage[x, y] += 1  # ✅ 센서 배치 (겹치는 경우 카운트 증가)

        coverage_score = np.sum(sensor_coverage > 0)  # ✅ 커버된 맵 내부 영역 수
        overlap_penalty = np.sum(sensor_coverage > 1) * 2  # ✅ 중복 센서 패널티
        sensor_penalty = num_sensors * 3  # ✅ 센서 개수 패널티

        fitness = coverage_score - (sensor_penalty + overlap_penalty)

        # ✅ 적합도가 음수면 랜덤하게 센서 추가
        while fitness < 0:
            new_pos = random.choice(self.feasible_positions)
            if new_pos[0] not in chromosome or new_pos[1] not in chromosome:
                chromosome.extend([new_pos[0], new_pos[1]])
                num_sensors += 1
                fitness = coverage_score - (num_sensors * 3 + overlap_penalty)  # 다시 적합도 계산

        return fitness

    def selection(self):
        """휠 룰렛 방식으로 부모 선택 (적합도가 높을수록 선택 확률 증가)"""
        fitness_scores = np.array([self.fitness_function(chromosome) for chromosome in self.population])
        fitness_scores = fitness_scores - np.min(fitness_scores) + 1  # ✅ 음수 보정하여 확률 계산 가능하도록 조정
        selection_prob = fitness_scores / np.sum(fitness_scores)
        selected_indices = np.random.choice(len(self.population), size=self.population_size // 2, p=selection_prob)
        return [self.population[i] for i in selected_indices]

    def crossover(self, parent1, parent2):
        """부모 개체 두 개를 교배하여 새로운 자식 개체 생성"""
        min_length = min(len(parent1), len(parent2))  
        child = []

        for i in range(0, min_length, 2):
            x_choice = random.choice([parent1[i], parent2[i]])
            y_choice = random.choice([parent1[i+1], parent2[i+1]])

            # ✅ 맵을 벗어나지 않는 센서만 추가
            if 0 <= x_choice < self.map_data.shape[0] and 0 <= y_choice < self.map_data.shape[1]:
                if self.map_data[x_choice, y_choice] == 1:
                    child.extend([x_choice, y_choice])

        # ✅ 최소한 3개의 센서 유지
        while len(child) < 6:  
            new_pos = random.choice(self.feasible_positions)
            child.extend([new_pos[0], new_pos[1]])

        return child

    def mutation(self, chromosome):
        """30% 확률로 맵 내부에 센서를 랜덤하게 추가하는 돌연변이 과정"""
        mutation_prob = 0.3  
        max_additional_sensors = 3  

        if random.random() < mutation_prob:  
            num_sensors = len(chromosome) // 2
            num_to_add = random.randint(1, max_additional_sensors)  
            new_positions = random.sample(self.feasible_positions.tolist(), num_to_add)

            # ✅ 기존 센서 위치와 중복되지 않도록 필터링
            existing_positions = {(chromosome[i], chromosome[i+1]) for i in range(0, len(chromosome), 2)}
            new_positions = [pos for pos in new_positions if tuple(pos) not in existing_positions]

            # ✅ 새로운 센서 추가
            for pos in new_positions:
                chromosome.extend([pos[0], pos[1]])

        return chromosome

    def evolve(self):
        """유전 알고리즘 실행"""
        for gen in range(self.generations):
            new_population = []
            selected_parents = self.selection()

            # ✅ 교배 과정 (자식 생성)
            while len(new_population) < self.population_size:
                parent1, parent2 = random.sample(selected_parents, 2)
                child = self.crossover(parent1, parent2)
                mutated_child = self.mutation(child)
                new_population.append(mutated_child)

            self.population = new_population
            best_solution = max(self.population, key=self.fitness_function)
            best_fitness = self.fitness_function(best_solution)
            num_sensors = len(best_solution) // 2

            print(f"Generation {gen+1} | Fitness: {best_fitness} | Num Sensors: {num_sensors}")

            # ✅ 세대 결과 CSV 파일에 저장
            self.save_generation_results(gen+1, best_fitness, num_sensors)

        return best_solution

    def save_generation_results(self, generation, fitness, num_sensors):
        """세대별 적합도 및 센서 개수를 CSV에 저장"""
        with open(self.file_path, mode="a", newline='') as file:
            writer = csv.writer(file)
            writer.writerow([generation, fitness, num_sensors])

    def run(self):
        """GA 실행 및 최적해 반환"""
        return self.evolve()
