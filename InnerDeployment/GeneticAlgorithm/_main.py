import random
from initializer import initialize_population
from FitnessFunction import fitnessFunc
from crossover import crossover

class SensorGA:
    def __init__(
                self, 
                input_map,
                coverage: int, 
                generations: int, 
                corner_positions: list,
                initial_size: int = 100, 
                selection_size: int = 50, 
                child_chromo_size: int = 100,
            ):
        self.map = input_map
        self.coverage = coverage
        self.generations = generations
        self.corner_positions = corner_positions
        self.generation_size = initial_size
        self.selection_size = selection_size
        self.child_size = child_chromo_size


        self.population = initialize_population(
                                                input_map=self.map,
                                                population_size=self.generation_size,
                                                corner_positions=self.corner_positions,
                                                min_sensors=70,
                                                max_sensors=100,
                                            )
        
    def fitness(self, individual):
        score = fitnessFunc(self.map, self.corner_positions, individual)
        return score
    
    def selection(self, generation):
        return None

    
    def crossover(self, parent1, parent2):
        child = crossover(parent1, parent2)
        return child
      
      
    def mutation(self, chromosome):
        return chromosome
        
        
    def run(self):
        for i in range(self.generations):
            # Selection, Crossover, Mutation would be implemented here
            selected_parents = self.selection()
            children = []
            for j in range(0, len(selected_parents), 2):
                if j + 1 < len(selected_parents):
                    child = self.crossover(
                        self.population[selected_parents[j]],
                        self.population[selected_parents[j + 1]]
                    )
                    children.append(child)
            self.population = children

            pass
        return self.population
        