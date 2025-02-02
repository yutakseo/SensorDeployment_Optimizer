import pygad
import os, csv, time
import numpy as np
from _SensorModule.Sensor import Sensor

RESULTS_DIR = "__RESULTS__"

class SensorGA:
    """GA 기반 센서 배치 최적화 클래스"""
    
    def __init__(self, map_data, coverage, generations):
        self.map_data = np.array(map_data)
        self.coverage = coverage
        self.generations = generations
        self.feasible_positions = np.argwhere(self.map_data == 1)
        self.num_genes = len(self.feasible_positions)
        self.last_fitness = None  # 이전 적합도 저장
        self.stagnation_counter = 0  # 정체 탐지 변수
        
        # ✅ 초기 해에서 센서 개수를 30~50개 사이로 랜덤 설정
        self.init_chromosome = np.zeros(self.num_genes, dtype=int)
        num_sensors_init = np.random.randint(30, 50)
        sensor_indices = np.random.choice(self.num_genes, size=num_sensors_init, replace=False)
        self.init_chromosome[sensor_indices] = 1

        self.range_ben = [{"low": 0, "high": 1.1} for _ in range(self.num_genes)]

        # GA 인스턴스 생성
        self.ga_instance = pygad.GA(
            num_generations=self.generations,
            num_parents_mating=40,
            sol_per_pop=100,
            num_genes=self.num_genes,
            gene_type=int,
            gene_space=self.range_ben,
            initial_population=np.tile(self.init_chromosome, (100, 1)),  
            fitness_func=self.fitness_function,
            parent_selection_type="rws",  # ✅ 부모 선택 방식을 Roulette Wheel Selection으로 변경
            crossover_type="uniform",
            mutation_type="adaptive",
            mutation_probability=[0.8, 0.5],  
            on_generation=self.on_generation_callback,  # ✅ 함수 참조 수정
            stop_criteria=["saturate_250"],
            parallel_processing=None
        )

    def deploy_simulation(self, solution):
        """센서 배치 시뮬레이션"""
        sensor_instance = Sensor(self.map_data)
        for i in range(self.num_genes):
            if solution[i] == 1:
                sensor_instance.deploy(sensor_position=self.feasible_positions[i], coverage=self.coverage)
        return sensor_instance.result()

    def fitness_function(self, ga_instance, solution, solution_idx):
        """적합도 함수: 센서 개수 최소화 + 겹침 방지 + 강제 센서 개수 제한"""
        dst = self.deploy_simulation(solution)

        numb_of_sensor = np.sum(solution == 1)  # 배치된 센서 개수
        feasible_grid = np.sum(dst >= 1)  # 커버된 영역 개수
        uncover = 1 if (np.sum(dst == 1)) == 0 else 0

        overlap_grid = np.sum(dst > 1)  # 중복 커버된 영역 개수
        overlap_penalty = (overlap_grid / feasible_grid) * 200  

        if numb_of_sensor <= 30:
            sensor_bonus = 50  
        else:
            sensor_bonus = 0

        if numb_of_sensor > 150:
            sensor_penalty = (numb_of_sensor - 150) * 3  
        else:
            sensor_penalty = 0

        fitness_score = (100 - numb_of_sensor * 0.4 - overlap_penalty - sensor_penalty + sensor_bonus) * uncover

        return fitness_score

    def on_generation_callback(self, ga_instance):
        """세대별 콜백 함수 (50세대마다 체크포인트 기록)"""
        generation = ga_instance.generations_completed
        fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]
        
        if self.last_fitness is not None and abs(fitness - self.last_fitness) < 1e-5:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0  

        print(f"\nGeneration = {generation}")
        print(f"Fitness    = {fitness}")
        print(f"Stagnation Counter = {self.stagnation_counter}")

        self.last_fitness = fitness  

        if generation % 50 == 0:
            solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
            num_sensors = np.sum(solution == 1)

            print(f"중간 세대 {generation}의 최적 해 저장")
            print(f"중간 세대 {generation}의 적합도: {solution_fitness}")
            print(f"센서의 수 (1의 개수): {num_sensors}")

            self.save_checkpoint(generation, solution_fitness, num_sensors)

    def save_checkpoint(self, generation, solution_fitness, num_sensors):
        """CSV 파일로 중간 결과 저장"""
        os.makedirs(RESULTS_DIR, exist_ok=True)
        file_path = os.path.join(RESULTS_DIR, "generation_results.csv")

        with open(file_path, mode="a", newline='') as file:
            writer = csv.writer(file)
            writer.writerow([generation, solution_fitness, num_sensors])

    def run(self):
        """GA 실행"""
        self.ga_instance.run()

        solution, solution_fitness, solution_idx = self.ga_instance.best_solution(self.ga_instance.last_generation_fitness)
        print(f"최종 최적 해: {solution}")
        print(f"최종 적합도 값: {solution_fitness}")
        print(f"최종 최적 해의 인덱스: {solution_idx}")

        indices = np.where(solution == 1)[0]
        result_list = [self.feasible_positions[i] for i in indices]
        result_list = [tuple(arr.tolist()) for arr in result_list]

        self.ga_instance.plot_fitness()
        print(solution)

        return result_list
