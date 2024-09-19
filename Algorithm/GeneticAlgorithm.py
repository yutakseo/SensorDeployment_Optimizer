import pygad
import os, sys, random, copy, time
import numpy as np
from datetime import datetime

__file__ = os.getcwd()
__root__ = os.path.dirname(__file__)
sys.path.append(os.path.join(__file__,"SensorModule"))



class sensor_GA:
    def __init__(self, map, coverage, generation):
        self.map_data = np.array(map)
        self.coverage = coverage
        self.generations = generation
        self.feasible_positions = np.argwhere(self.map_data == 1)
        
        
        self.__init__chromsome__ = np.random.choice([0,1], size=self.feasible_positions.shape[0], p=[0.1, 0.9])
        self.num_of_parents_mating = 120
        self.solutions_per_pop = 240
        self.num_of_genes = len(self.feasible_positions)
        self.last_fitness = 0
        
        #초기 염색체 생성 함수
        function_inputs = self.__init__chromsome__
        #기대값 설정
        desired_output = 95
        #유전자 해범위 설정
        self.range_ben = [{"low": 0,"high":1.1} for i in range(self.num_of_genes)]
        
    def deploy_simulation(self, solution):
        self.sensor_instance = Sensor(self.map_data)
        for i in range(self.num_of_genes):
            if solution[i] == 1:
                self.sensor_instance.deploy(sensor_position=self.feasible_positions[i], coverage=self.coverage)
        return self.sensor_instance.result()
        
    def check_cover_restriction(self, simulation):
        if np.sum(simulation == 1) == 0:
            return 1
        else:
            return 0
        
    def fitness_func(self, ga_instance, solution, solution_idx):
        #적합도함수는 #센서개수 최소화(목적) #제약조건: 모든 현장 커버리지 커버
        simulation = self.deploy_simulation(solution=solution)
        numb_of_sensor = np.sum(solution == 1)
        
        #Objective Function
        Minimize = self.num_of_genes - numb_of_sensor
        
        #Constraint
        cond1 = self.check_cover_restriction(simulation=simulation)

        return round((Minimize*cond1)/self.num_of_genes*100,3)
        
    def on_generation(self, ga_instance):
        print("\nGeneration = {generation}".format(generation=ga_instance.generations_completed))
        print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]))
        print("Change     = {change}".format(change=ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1] - self.last_fitness))
        self.last_fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]
        return self.last_fitness
        
    def run(self):
        ga_instance = pygad.GA(
                        num_generations = self.generations,
                        num_parents_mating = self.num_of_parents_mating,
                        sol_per_pop = self.solutions_per_pop,
                        num_genes = self.num_of_genes,
                        gene_type = int,
                        gene_space = self.range_ben,
                        fitness_func = self.fitness_func,
                        parent_selection_type="sss",
                        crossover_type="two_points",
                        mutation_type="adaptive",
                        mutation_probability=[1.0, 0.7],
                        on_generation = self.on_generation,
                        stop_criteria=["saturate_1000"],
                        parallel_processing=24)
        
        ga_instance.run()
        
        solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
        print("Parameters of the best solution : {solution}".format(solution=solution))
        print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
        print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))


        indices = np.where(solution == 1)[0]
        result_list = [self.feasible_positions[i] for i in indices]
        result_list = [tuple(arr.tolist()) for arr in result_list]
        ga_instance.plot_fitness()
        print(solution)
        
        return result_list