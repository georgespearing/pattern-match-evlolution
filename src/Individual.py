import numpy as np

class Individual:

    def __init__(self, genome_length):
        self.genome = np.random.randint(low=0, high=2, size=genome_length)
        self.fitness_function = fitness_function
        self.fitness = 0
        
    # def eval_fitness(self):
    #     self.fitness= self.fitness_function(self.genome)


if __name__=='__main__':
    print("nope, import only")