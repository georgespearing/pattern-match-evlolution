import numpy as np

class Individual:

    def __init__(self, genome_length, upper_limit):
        # self.genome = (np.random.randint(low=0, high=upper_limit, size=genome_length)).atype(str)
        self.genome = np.random.randint(low=0, high=upper_limit, size=genome_length)
        # self.genome.astype(str)
        self.fitness = 0
        self.novelty = 0
    
    # def __init__(self, genome_length):
    #     self.genome = (np.random.randint(low=0, high=35, size=genome_length)).astype(str) # assign random numbers
    #     # self.genome.astype(str)
    #     self.addLetters()
    #     self.fitness = 0
    #     self.novelty = 0
        
    #     # know which indexes match the predator
    #     self.match_indexes = [0,0]
    # # def eval_fitness(self):
    # #     self.fitness= self.fitness_function(self.genome)

    # def addLetters(self):
    #     letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    #     for index, alleal in enumerate(self.genome):
    #         if int(alleal) > 9: 
    #             self.genome[index] = letters[int(alleal)-10]

    # def mutate(self, elements_to_mutate):
    #     for index in elements_to_mutate:
    #         self.genome[index] = np.random.randint(0,35)


if __name__=='__main__':
    print("nope, import only")