import pygad
import os, sys, random, copy, time, csv
import numpy as np
from datetime import datetime

__file__ = os.getcwd()
__root__ = os.path.dirname(__file__)
sys.path.append(os.path.join(__file__,"SensorModule"))
from Sensor import *

class sensor_GA:
    # 중간 결과를 출력할 세대 리스트
    checkpoints = [10,20,30,50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000]
    generation_results = []
    def __init__(self, map, coverage, generation):
        self.map_data = np.array(map)
        self.coverage = coverage
        self.generations = generation
        self.feasible_positions = np.argwhere(self.map_data == 1)
        self.__init__chromsome__ = np.random.choice([0,1], size=self.feasible_positions.shape[0], p=[0.9, 0.1])
        self.num_of_parents_mating = 60
        self.solutions_per_pop = 120
        self.num_of_genes = len(self.feasible_positions)
        self.last_fitness = 100
        
        # 초기 염색체 생성 함수
        function_inputs = self.__init__chromsome__
        # 기대값 설정
        desired_output = 95
        # 유전자 해 범위 설정
        self.range_ben = [{"low": 0, "high": 1.1} for i in range(self.num_of_genes)]
         
    def deploy_simulation(self, solution):
        self.sensor_instance = Sensor(self.map_data)
        for i in range(self.num_of_genes):
            if solution[i] == 1:
                self.sensor_instance.deploy(sensor_position=self.feasible_positions[i], coverage=self.coverage)
        return self.sensor_instance.result()
         
    #새 적합도 함수
    def fitness_func(self, ga_instance, solution, solution_idx):
        # 적합도 함수는 센서 개수 최소화(목적), 제약조건: 모든 현장 커버리지 커버
        self.dst = self.deploy_simulation(solution=solution)
        
        numb_of_sensor = np.sum(solution == 1)
        feasible_grid = np.sum(self.dst >=1)
        uncover = 1 if (np.sum(self.dst == 1)) == 0 else 0
        overlap_grid = np.sum(self.dst >= 20)
        
        # Objective Function
        Minimize = round((self.num_of_genes - numb_of_sensor) / self.num_of_genes * 100, 3)
        Minimize_overlap = overlap_grid / feasible_grid * 100
        
        return min(Minimize, Minimize_overlap)* uncover

    def on_generation(self, ga_instance):
        generation = ga_instance.generations_completed
        print("\nGeneration = {generation}".format(generation=generation))
        print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]))
        print("Change     = {change}".format(change=ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1] - self.last_fitness))
        self.last_fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]
        
        # 중간 결과를 저장
        if generation in self.checkpoints:
            solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
            num_sensors = np.sum(solution == 1)  # 해에서 1의 개수(센서의 수) 계산
            print(f"중간 세대 {generation}의 최적 해: {solution}")
            print(f"중간 세대 {generation}의 적합도: {solution_fitness}")
            print(f"센서의 수 (1의 개수): {num_sensors}")
            # generation_results 리스트에 저장
            self.generation_results.append((generation, solution, solution_fitness, num_sensors))
            # CSV 파일로 저장 (세대, 최적 해, 적합도, 1의 개수)
            with open("generation_results.csv", mode="a", newline='') as file:
                writer = csv.writer(file)
                #solution 잠깐 제거했음
                writer.writerow([generation, solution, solution_fitness, num_sensors])
        return self.last_fitness
        
    def run(self):
        ga_instance = pygad.GA(
                        num_generations=self.generations,
                        num_parents_mating=self.num_of_parents_mating,
                        sol_per_pop=self.solutions_per_pop,
                        num_genes=self.num_of_genes,
                        gene_type=int,
                        gene_space=self.range_ben,
                        fitness_func=self.fitness_func,
                        parent_selection_type="sss",
                        crossover_type="two_points",
                        mutation_type="adaptive",
                        mutation_probability=[1.0, 0.7],
                        on_generation=self.on_generation,
                        stop_criteria=["saturate_500"],
                        parallel_processing=24)
        ga_instance.run()
        
        solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
        print("최종 최적 해: {solution}".format(solution=solution))
        print("최종 적합도 값: {solution_fitness}".format(solution_fitness=solution_fitness))
        print("최종 최적 해의 인덱스: {solution_idx}".format(solution_idx=solution_idx))

        indices = np.where(solution == 1)[0]
        result_list = [self.feasible_positions[i] for i in indices]
        result_list = [tuple(arr.tolist()) for arr in result_list]
        ga_instance.plot_fitness()
        print(solution)
        
        return result_list
