import numpy as np
import os, sys, csv, random
import torch
from Tools.FitnessFunction import SensorEvaluator
from Tools.SensorModule import Sensor


class SensorGA:
    def __init__(
                 self, 
                 map_data:np.ndarray,
                 coverage:int, 
                 generations:int, 
                 results_dir:str,
                 corner_positions:list,
                 initial_population_size:int=100, 
                 next_population_size:int=50, 
                 candidate_population_size:int=100
                 ):
        
        self.base_map = np.array(map_data)
        self.coverage = int(coverage/5)
        self.corner_positions = corner_positions
        self.generations = generations
        self.initial_population_size = initial_population_size
        self.next_population_size = next_population_size
        self.candidate_population_size = candidate_population_size
        self.feasible_positions = set(map(tuple, np.argwhere(self.base_map == 1)))
        self.rows, self.cols = self.base_map.shape
        
        #log data setting
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        self.file_path = os.path.join(self.results_dir, "generation_results.csv")
        with open(self.file_path, mode="w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Generation", "Fitness", "Num_Sensors", "Coverage_Score"])
        
        #init fitness function instance
        self.fitnessFunc = SensorEvaluator(map=self.base_map, corner_points=self.corner_positions, coverage=self.coverage)
        
        #init first generation population(chromosomes)
        self.population = []
        self.init_population()
        
        #init best score and chromosome
        self.best_score = 0
        self.best_chromosome = []
        for chromosome in self.population:
            if chromosome[1] >= 0:
                self.best_score = chromosome[1]
                self.best_chromosome = chromosome[0]
        
        self.min_sensor_count = float('inf')
        
    # ============================= initialize population ======================================
    #초기 염색체 생성(생성 가능한 그리드에서 랜덤한 염색체 생성)
    def init_population(self):
        feasible_positions = list(self.feasible_positions)
        for idx in range(self.initial_population_size):
            chromosome = []
            for _ in range(random.randrange(10, 100)):
                x, y = random.choice(feasible_positions)
                chromosome.extend([x, y])
            fitness_score = self.evaluate_genes(chromosome=chromosome)
            self.population.append([chromosome, fitness_score])
    # ==========================================================================================


    # ==================================== FitnessFuction ======================================
    #개별 염색체의 적합도를 출력
    def evaluate_genes(self, chromosome:list) -> float:
        # [x1, y1, x2, y2, ...] → [(x1, y1), (x2, y2), ...]
        sensor_list = [(chromosome[i], chromosome[i+1]) for i in range(0, len(chromosome), 2)]
        return self.fitnessFunc(inner_positions=sensor_list)
    
    #한 세대의 염색체 집합에 대한 적합도 채점
    def fitness_func(self, population:list):
        for chromosome in population:
            fitness_score = self.evaluate_genes(chromosome=chromosome)
            chromosome[1] = fitness_score
    # ==========================================================================================
            
            
            
    # ================================ Chromosome Selection ====================================    


    from typing import List, Tuple
    def elite_selection_list(population: List[Chromosome],
        Coord = Tuple[int, int]
        Chromosome = List[Coord]
                            fitness: List[float],
                            top_k: int,
                            unique: bool = True,
                            prefer_fewer_sensors: bool = True) -> List[int]:
        """
        반환: 상위 top_k의 '인덱스' 리스트 (내림차순)
        - population[i]    : i번째 염색체 ([(x,y), ...])
        - fitness[i]       : i번째 염색체 적합도
        - unique=True      : 좌표 집합이 동일한 염색체 중복 제거
        - prefer_fewer_sensors=True : 동점일 때 센서 수가 적은 해 우선
        """
        n = len(population)
        if n == 0 or top_k <= 0:
            return []
        top_k = min(top_k, n)

        def norm_key(ch: Chromosome) -> Tuple[Coord, ...]:
            # 순서/중복 무의미 처리: 집합화 후 정렬 → 해시 가능한 키
            genes = set(map(tuple, ch))
            return tuple(sorted(genes))

        def num_genes(ch: Chromosome) -> int:
            return len(set(map(tuple, ch)))

        order = sorted(
            range(n),
            key=lambda i: (
                -fitness[i],                                # 1) 적합도 내림차순
                num_genes(population[i]) if prefer_fewer_sensors else 0,  # 2) 센서 수 적게
                i                                           # 3) 인덱스(안정성)
            )
        )

        selected = []
        seen = set()
        for i in order:
            if unique:
                k = norm_key(population[i])
                if k in seen:
                    continue
                seen.add(k)
            selected.append(i)
            if len(selected) == top_k:
                break
        return selected
        
    # ==========================================================================================
    
    
    
    
    # =================================== Crossover Method =====================================  
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
    # ==========================================================================================
        
    
    

    # =================================== Mutation Method =====================================
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
    # ==========================================================================================
    
    
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
            coverage_score = best_score  # 너의 fitness 자체가 coverage 기준이니까...>_*찡긋
            if best_score == 100.0 and num_sensors < self.min_sensor_count:
                self.best_chromosome = (best_chromosome.copy(), best_score)
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
        
