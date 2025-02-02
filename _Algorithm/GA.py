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
        
        # ✅ 초기 해에서 센서 개수를 30~50개 사이에서 랜덤 설정
        self.init_chromosome = np.zeros(self.num_genes, dtype=int)
        num_sensors_init = np.random.randint(30, 50)
        sensor_indices = np.random.choice(self.num_genes, size=num_sensors_init, replace=False)
        self.init_chromosome[sensor_indices] = 1

        self.range_ben = [{"low": 0, "high": 1.1} for _ in range(self.num_genes)]

        # GA 인스턴스 생성
        self.ga_instance = pygad.GA(
            num_generations=self.generations,
            num_parents_mating=50,  
            sol_per_pop=150,  
            num_genes=self.num_genes,
            gene_type=int,
            gene_space=self.range_ben,
            initial_population=np.tile(self.init_chromosome, (150, 1)),  
            fitness_func=self.fitness_function,
            parent_selection_type="sus",  
            crossover_type="two_points",  
            mutation_type="adaptive",
            mutation_probability=self.adaptive_mutation(),  # ✅ 동적 변이율 적용
            on_generation=self.on_generation_callback,
            stop_criteria=["saturate_500"],  
            parallel_processing=None,
            keep_elitism=2  # ✅ 가장 적합한 2개의 개체 유지
        )

    def adaptive_mutation(self):
        """적응형 변이율 설정: 탐색이 정체될 경우 변이율 증가"""
        if self.stagnation_counter >= 10:  # 10세대 연속 변화 없으면 변이율 100%
            return [1.0, 1.0]
        elif self.stagnation_counter >= 5:  # 5세대 연속 변화 없으면 변이율 증가
            return [0.9, 0.7]
        return [0.7, 0.5]  # 기본 변이율

    def deploy_simulation(self, solution):
        """센서 배치 시뮬레이션"""
        sensor_instance = Sensor(self.map_data)
        for i in range(self.num_genes):
            if solution[i] == 1:
                sensor_instance.deploy(sensor_position=self.feasible_positions[i], coverage=self.coverage)
        return sensor_instance.result()

    def fitness_function(self, ga_instance, solution, solution_idx):
        """적합도 함수: 센서 개수 최소화 + 중복 커버리지 최소화 + 센서 배치 균형 유지"""
        dst = self.deploy_simulation(solution)

        numb_of_sensor = np.sum(solution == 1)  # 배치된 센서 개수
        feasible_grid = np.sum(dst >= 1)  # 커버된 영역 개수
        uncover = 1 if (np.sum(dst == 1)) == 0 else 0

        # ✅ 중복 커버리지 개수별 계산
        num_overlap_2 = np.sum(dst == 2)  # 2개 센서가 겹친 영역 개수
        num_overlap_3 = np.sum(dst == 3)  # 3개 센서가 겹친 영역 개수
        num_overlap_4 = np.sum(dst >= 4)  # 4개 이상 센서가 겹친 영역 개수

        # ✅ 가중치 적용 (더 많이 겹칠수록 패널티 증가)
        overlap_penalty = (num_overlap_2 * 1 + num_overlap_3 * 3 + num_overlap_4 * 5) / feasible_grid * 100

        # ✅ 센서 개수가 150개 이상이면 패널티 적용 (기존보다 완화)
        sensor_penalty = max(0, (numb_of_sensor - 150) * 1.5)  

        # ✅ 센서 배치 개수가 30~60개 사이면 보너스 적용
        sensor_bonus = max(0, 50 - abs(numb_of_sensor - 45))  

        # ✅ 새로운 적합도 공식
        fitness_score = (100 - numb_of_sensor * 0.3 - overlap_penalty - sensor_penalty + sensor_bonus) * uncover

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
