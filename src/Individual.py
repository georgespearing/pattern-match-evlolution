import numpy as np

class Individual:

    def __init__(self, genome_length, upper_limit):
        self.genome = np.random.randint(low=0, high=upper_limit, size=genome_length)
        self.fitness = 0
        self.novelty = 0
        self.age = 0
        
        # know which indexes match the predator
        self.match_indexes = [0,0]
    # def eval_fitness(self):
    #     self.fitness= self.fitness_function(self.genome)


if __name__=='__main__':
    print("nope, import only")