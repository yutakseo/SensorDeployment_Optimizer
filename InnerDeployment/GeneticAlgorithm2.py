from selection import *
#chromosome == [[x1,y1], [x2,y2], ...],
#chromosome2 == [[x1,y1], [x2,y2], ...],
# ....
#generation == [chromosome, chromosome2]
class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate):
        self.pop_size = population_size
        self.mutation_rate = mutation_rate
        generation = init_generation(self.pop_size)

    def selection(self, generation, n_perents:int):
        
        return

def geneticAlgorithm():
    #initialize population
    
    #evaluate fitness
    
    #perform selection
    
    #perform crossover
    
    #perform mutation
    return