import numpy as np
import random
import os, sys, csv
import torch

from SensorModule.Sensor_v2 import Sensor
from Tools.FitnessFunction import SensorEvaluator


class SensorGA:
    def __init__(
                 self, 
                 input_map,
                 coverage:int, 
                 generations:int, 
                 results_dir:str,
                 corner_positions:list,
                 initial_population_size:int=100, 
                 next_population_size:int=50, 
                 candidate_population_size:int=100,
                 ):
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
        self.base_map = np.array(input_map)
        self.coverage = int(coverage/5)
        self.generations = generations
        self.corner_positions = corner_positions
        self.initial_population_size = initial_population_size
        self.next_population_size = next_population_size
        self.candidate_population_size = candidate_population_size
        self.feasible_positions = set(map(tuple, np.argwhere(self.base_map == 1)))
        self.rows, self.cols = self.base_map.shape
        # 결과 저장 폴더 설정
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        # CSV 파일 저장 경로 설정
        self.file_path = os.path.join(self.results_dir, "generation_results.csv")
        with open(self.file_path, mode="w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Generation", "Fitness", "Num_Sensors", "Coverage_Score"])
        # 초기 개체군 생성
        self.population = {}
        self.best_solution = None
        self.min_sensor_count = float('inf')
        
        #적합도평가함수 인스턴스 생성
        self.fitnessFunc = SensorEvaluator(map=self.base_map, corner_points=self.corner_positions, coverage=self.coverage)


    #개별 염색체의 적합도를 출력
    def _fitness_func(self, chromosome) -> float:
        # [x1, y1, x2, y2, ...] → [(x1, y1), (x2, y2), ...]
        sensor_list = [(chromosome[i], chromosome[i+1]) for i in range(0, len(chromosome), 2)]
        return self.fitnessFunc(inner_positions=sensor_list)
    #한 세대의 염색체 집합에 대한 적합도 채점
    def fitness_func(self, population) -> list:
        idx = 0
        for chromosome in population:
            fitness_score = self._fitness_func(chromosome=chromosome)
            self.population[idx] = (chromosome, fitness_score)
            idx += 1
    
    
    #초기 염색체 생성(랜덤하게 생성 가능한 그리드에서 유전자 생성)
    def init_population(self, min_numb, max_numb):
        feasible_positions = list(self.feasible_positions)
        for idx in range(self.initial_population_size):
            num_genes = random.randint(min_numb, max_numb)
            chromosome = []
            for _ in range(num_genes):
                x, y = random.choice(feasible_positions)
                chromosome.extend([x, y])
            fitness_score = self._fitness_func(chromosome=chromosome)
            self.population[idx] = (chromosome, fitness_score)
            
            
    def elite_selection(self, next_generation) -> list:
        top_k = next_generation if next_generation is not None else self.initial_population_size

        # 적합도 기준 내림차순 정렬 → (idx, (chromosome, fitness)) 형태 유지
        sorted_population = sorted(self.population.items(), key=lambda item: item[1][1], reverse=True)
        # 상위 top_k 개체 인덱스만 추출
        selected_chromosome = [idx for idx, _ in sorted_population[:top_k]]
        return selected_chromosome
    
       
    def _crossover(self, p1: list, p2: list) -> list:
        p1 = p1.copy()
        p2 = p2.copy()
        # 유효한 (x, y) 쌍 유지
        if len(p1) % 2 != 0:
            p1 = p1[:-1]
        if len(p2) % 2 != 0:
            p2 = p2[:-1]
        child_genes = []
        # 교차: 각 좌표를 부모 범위 내 랜덤 선택
        for i in range(0, min(len(p1), len(p2)), 2):
            x1, y1 = p1[i], p1[i + 1]
            x2, y2 = p2[i], p2[i + 1]
            x = random.randint(min(x1, x2), max(x1, x2))
            y = random.randint(min(y1, y2), max(y1, y2))
            child_genes.append((int(x), int(y)))
            
        # 중복 좌표 제거
        child_genes = list(set(child_genes))
        # CNN 기반 적합도 맵으로 센서 우선순위 정렬
        ranked = self.fitnessFunc.rankSensors(sensor_points=child_genes)
        # 정렬된 센서를 평탄화하여 최종 자식 염색체 구성
        sorted_child = []
        for (x, y), _ in ranked:
            sorted_child.extend([x, y])
        return sorted_child
        
        
    def crossover(self, selected_idx: list) -> None:
        new_population = {}
        next_idx = 0

        # 1. 엘리트 개체 상위 N개 보존
        num_elites = min(self.next_population_size, len(selected_idx))
        elite_indices = sorted(selected_idx, key=lambda idx: self.population[idx][1], reverse=True)[:num_elites]
        for idx in elite_indices:
            new_population[next_idx] = self.population[idx]
            next_idx += 1

        # 2. 엘리트 외 나머지는 교배로 생성
        while next_idx < self.candidate_population_size:
            mom_idx, dad_idx = random.sample(selected_idx, 2)
            mom, _ = self.population[mom_idx]
            dad, _ = self.population[dad_idx]
            child = self._crossover(mom, dad)
            fitness_score = self._fitness_func(child)
            new_population[next_idx] = (child, fitness_score)
            next_idx += 1

        # 새로운 세대로 population 교체
        self.population = new_population

        
    
    def mutation(self, chromosome) -> list:
        genes, score = chromosome  # chromosome: (genes, fitness_score)
        genes = genes.copy()  # 원본 보호

        if score >= 100:
            # 높은 성능이면 센서 하나 제거
            if len(genes) >= 2:
                genes = genes[:-2]
        else:
            # 성능 낮으면 미커버 영역에서 센서 추가
            sensor_pos_list = [(genes[i], genes[i+1]) for i in range(0, len(genes), 2)]
            positions = self.fitnessFunc.extractUncovered(corner_positions=self.corner_positions, inner_positions=sensor_pos_list)
            if positions:
                x, y = random.choice(positions)
                genes.extend([int(x), int(y)])
        return genes
    
    
    def loop(self, min_genes=20, max_genes=30) -> None:
        self.init_population(min_genes, max_genes)  # 초기 해 생성

        for gen in range(self.generations):
            print(f"[Generation : {gen}]")

            # 1. 엘리트 선택
            selected_idx = self.elite_selection(next_generation=self.next_population_size)

            # 2. 교배
            self.crossover(selected_idx)

            # 3. 돌연변이 적용
            for idx in self.population:
                mutated_genes = self.mutation(self.population[idx])
                mutated_fitness = self._fitness_func(mutated_genes)
                self.population[idx] = (mutated_genes, mutated_fitness)

            # 4. 베스트 결과 저장
            best_idx, (best_chromosome, best_score) = max(self.population.items(), key=lambda x: x[1][1])
            num_sensors = len(best_chromosome) // 2
            coverage_score = best_score  # 너의 fitness 자체가 coverage 기준이니까
            
            if best_score == 100.0 and num_sensors < self.min_sensor_count:
                self.best_solution = (best_chromosome.copy(), best_score)
                self.min_sensor_count = num_sensors

            # 5. CSV 저장
            with open(self.file_path, mode="a", newline='') as file:
                writer = csv.writer(file)
                writer.writerow([gen, best_score, num_sensors, coverage_score])

            print(f"[Generation {gen}] Best Fitness: {best_score:.4f} | Sensors: {num_sensors}")
        print("Genetic Algorithm Complete")

    
    
    def run(self):
        self.loop()
        
        # 최종 세대의 최고 해를 리턴
        best_idx, (best_chromosome, best_score) = max(self.population.items(), key=lambda x: x[1][1])
        return {
            "best_chromosome": best_chromosome,
            "fitness_score": best_score,
            "num_sensors": len(best_chromosome) // 2
        }
        
