import numpy as np

class Individual:

    # an individual is represented by an array of ints. for example: [1, 6, 2, 9, 0]
    # it will have some fitness, novelty, and age. 
    # fitness is assesssed in comparision to an external predator
    # novelty represented 'uniqueness' of solution compared to an archive
    # age is number of generations it has been 'alive'

    def __init__(self, genome_length, upper_limit):
        self.genome = np.random.randint(low=0, high=upper_limit, size=genome_length)
        self.fitness = 0
        self.novelty = 0
        self.age = 0
        self.match_indexes = [0,genome_length]

if __name__=='__main__':
    print("nope, import only")