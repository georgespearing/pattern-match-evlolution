'''
George Spearing 
November 2021
CS352

main.py

Evolutionary algorithm developed to match a string pattern of an individual using group dynamics. 
For each population, the highest fit individual (measured by how long of a pattern they match the "predator")
will do crossover in attempt to combine pattern fitness. 

'''

################################
#      IMPORTS     #
################################

import numpy as np
import copy
import matplotlib.pyplot as plt
plt.style.use('seaborn')

import scikits.bootstrap as bootstrap
import warnings
warnings.filterwarnings('ignore') # Danger, Will Robinson! (not a scalable hack, and may surpress other helpful warning other than for ill-conditioned bootstrapped CI distributions)

import scipy.stats # for finding statistical significance

import time

import Landscape
import Individual

################################
#      MAIN FUNCTION CALL      #
################################

def main():

    num_runs = 20
    total_generations = 500
    num_elements_to_mutate = 1
    bit_string_length = 40
    num_parents = 20
    num_children = 20
    upper_limit = 10
    max_age = total_generations/10 # age before individual dies, don't do this for parent testing

    # adding novelty 
    novelty_k = 5
    novelty_selection_prop = 0.1 # lower number means more novelty. .3 or .4 works the best
    max_archive_length = 50

    num_random_parents = [10] # with replacement

    experiment_results = {}
    solutions_results = {}
    diversity_results = {}

    run_names = ["mutation only", "2 point random crossover", "pattern matching"]
    modifications = [["None", "random"],["2Point", "random"],['Combinational', "random"]]

    for parents in num_random_parents: # run a bunch of different parent combindations

        for run_index, run_name in enumerate(run_names):

            experiment_results[run_name] = np.zeros((num_runs, total_generations))
            solutions_results[run_name] = np.zeros((num_runs, total_generations, bit_string_length), dtype=int)
            diversity_results[run_name] = np.zeros((num_runs, total_generations))

            for run_num in range(num_runs):
                print(f'Working on run name: {run_name} num: {run_num}')

                # run the algorithm
                fitness, solutions, diversity = evolutionary_algorithm(total_generations=total_generations, \
                        num_parents=num_parents, num_children=num_children, bit_string_length=bit_string_length, \
                        num_elements_to_mutate=num_elements_to_mutate, crossover=modifications[run_index][0], mutation=modifications[run_index][1], \
                        novelty_k = novelty_k, novelty_selection_prop = novelty_selection_prop, max_archive_length = max_archive_length, \
                        upper_limit=upper_limit, num_random_parents=parents, max_age=max_age)
                
                # save the results
                experiment_results[run_name][run_num] = fitness
                solutions_results[run_name][run_num] = solutions
                diversity_results[run_name][run_num] = diversity
                print(f'run name: {run_name} | num: {run_num} | fit: {fitness[-1]}\n')

        # plotting
        data_names = run_names

        plot_mean_and_bootstrapped_ci_over_time(input_data = experiment_results, name = data_names, title=f'combination methods', x_label = "Generation", y_label = "Fitness", y_limit = [0,bit_string_length], plot_bootstrap = False)
        plot_mean_and_bootstrapped_ci_over_time(input_data = diversity_results, name = data_names, title=f'combination methods', x_label = "Generation", y_label = "Diversity", plot_bootstrap = False)


################################
#           ALGORITHM          #
################################

def evolutionary_algorithm(total_generations=100, num_parents=10, num_children=10, bit_string_length=10, num_elements_to_mutate=1, \
    crossover=True, mutation=True, novelty_k = 0, novelty_selection_prop = 0, max_archive_length = 100, upper_limit = 3, num_random_parents = 4, max_age=10):
    """
        parameters: 
        fitness_funciton: (callable function) that return the fitness of a genome 
                           given the genome as an input parameter (e.g. as defined in Landscape)
        total_generations: (int) number of total iterations for stopping condition
        num_parents: (int) the number of parents we downselect to at each generation (mu)
        num_childre: (int) the number of children (note: parents not included in this count) that we baloon to each generation (lambda)
        bit_string_length: (int) length of bit string genome to be evoloved
        num_elements_to_mutate: (int) number of alleles to modify during mutation (0 = no mutation)
        crossover (bool): whether to perform crossover when generating children 
        
        returns:
        fitness_over_time: (numpy array) track record of the top fitness value at each generation
    """

    # initialize record keeping
    solution = None # best genome so far
    solution_fitness = -99999 # fitness of best genome so far
    best_accuracy = -99999 # fitness of best genome so far
    fitness_over_time = np.zeros(total_generations)
    solutions_over_time = np.zeros((total_generations,bit_string_length))
    diversity_over_time = np.zeros(total_generations)
    solution_archive = []

    change_in_fitness = 0
    last_fitness = 0
    last_last_fitness = 0
    
    ################################
    #   INITILIZATING POPULATION   #
    ################################

    # create the predator (random instantiation)
    predator = Individual.Individual(bit_string_length, upper_limit)

    # create an initial population 
    population = [] # keep population of individuals in a list
    for i in range(num_parents): # only create parents for initialization (the mu in mu+lambda)
        population.append(Individual.Individual(bit_string_length, upper_limit)) # generate new random individuals as parents
    # print(f'population 0 genome:     {population[0].genome}')

    # get population fitness
    for i in range(len(population)):
        population[i].fitness, population[i].match_indexes = get_fitness(predator, population[i]) # evaluate the fitness of each parent

    # add population to solution archive to initialize it
    for i in range(len(population)):
        solution_archive.append(population[i])
    

    ################################
    #    MODIFICATION PROCEDURE    #
    ################################

    for generation_num in range(total_generations): # repeat

        if generation_num % 500 == 0:
            print(generation_num)
        
        # the modification procedure
        new_children = [] # keep children separate for now (lambda in mu+lambda)
        while len(new_children) < num_children:
            
            # inheretance
            random_parents = np.random.choice(population, size=num_random_parents) # pick 5 random parents
            new_child = (copy.deepcopy(random_parents[0])) # initialize children as perfect copies of their parents
            new_child.age = 0

            # crossover -- intelligently combine the best parts of many parents
            if crossover=='Combinational':
                for parent in random_parents[1:]: # combine the information from multiple parents into one child.
                    new_child.genome[parent.match_indexes[0]:parent.match_indexes[1]] = parent.genome[parent.match_indexes[0]:parent.match_indexes[1]] # for replacing patterns that match 
            
            # crossover/mutation
            if crossover=='2Point':
                [crossover_point1, crossover_point2] = sorted(np.random.randint(0,bit_string_length,2)) # crossover points for 2-point crossover (sorted to make indexing easier in the next step)
                # uses the second random choice for the random parents (NOTE: due to np random choice, this could be the same parent)
                new_child.genome[crossover_point1:crossover_point2+1] = random_parents[1].genome[crossover_point1:crossover_point2+1] # take the point between the crossover points and swap in the genes from the other parent

            # mutation - random bit changes
            if mutation=="random":
                # for this_child in [child1,child2]:
                elements_to_mutate = set() 
                while len(elements_to_mutate)<num_elements_to_mutate:
                    elements_to_mutate.add(np.random.randint(bit_string_length)) # randomly select the location in the child bit string to mutate
                for this_element_to_mutate in elements_to_mutate:
                    # new_child.genome[this_element_to_mutate] = np.random.randint(0,np.max(predator.genome)+1)
                    new_child.genome[this_element_to_mutate] = np.random.randint(0,np.max(predator.genome)+1)

            # mutation - intelligent bit changes
            if mutation=="intelligent":
                # for this_child in [child1,child2]:
                elements_to_mutate = set() 
                elements_to_ignore = np.arange(new_child.match_indexes[0], new_child.match_indexes[1])
                while len(elements_to_mutate)<num_elements_to_mutate:
                    indexes_to_mutate = np.random.choice(np.delete(np.arange(0,bit_string_length), elements_to_ignore), size=1)
                    # print(indexes_to_mutate)
                    elements_to_mutate.add(indexes_to_mutate[0]) # randomly select the location in the child bit string to mutate
                for this_element_to_mutate in elements_to_mutate:
                    # new_child.genome[this_element_to_mutate] = np.random.randint(0,np.max(predator.genome)+1)
                    new_child.genome[this_element_to_mutate] = np.random.randint(0,np.max(predator.genome)+1)

            new_children.append(new_child) # add children to the new_children list

        ################################
        #     ASSESSMENT PROCEDURE     #
        ################################

        # assign fitness to each child 
        for i in range(len(new_children)):
            new_children[i].fitness, new_children[i].match_indexes = get_fitness(predator, new_children[i]) 

        # assign novelty to each child 
        for i in range(len(new_children)):
            new_children[i].novelty = get_novelty(solution_archive, new_children[i], novelty_k)            
            solution_archive = update_archive(solution_archive, new_children[i], max_archive_length)

        ################################
        #      SELECTION PROCEDURE     #
        ################################

        # combine parents with new children (the + in mu+lambda)
        population += new_children  

        # age the entire population
        for ind in population: 
            ind.age += 1
            if ind.age >= max_age:
                population.remove(ind)

        # Get the top __ fit invididuals
        population = sorted(population, key=lambda individual: individual.fitness, reverse=True) 
        # perform truncation selection for top % fitt individuals
        new_population = population[:int(num_parents*(1-novelty_selection_prop))] 

        # fill in the rest of next generation with novel solutions
        population = sorted(population, key=lambda individual: individual.novelty, reverse=True) # sort the full population by each individual's fitness (from highers to lowest)
        while len(new_population) < num_parents:
            this_ind = population.pop()
            if not this_ind in new_population:
                new_population.append(this_ind)   

        ################################
        #         RECORD KEEPING       #
        ################################

        # sort by fitness again
        population = sorted(new_population, key=lambda individual: individual.fitness, reverse=True)
        if population[0].fitness > solution_fitness: # if the new parent is the best found so far
            solution = population[0].genome                 # update best solution records
            # solution_fitness = population[0].fitness

            # prioritize group fitness by setting solution fitness to the mean of the group
            solution_fitness = int(np.mean([population[i].fitness for i in range(len(population))])) 

        # record the fitness of the current best over evolutionary time
        fitness_over_time[generation_num] = solution_fitness 
        solutions_over_time[generation_num] = solution
        
        genome_list = np.array([individual.genome for individual in population])
        diversity = np.mean(genome_list.std(axis=0))
        diversity_over_time[generation_num] = diversity
    
    return fitness_over_time, solutions_over_time, diversity_over_time 

################################
#     FITNESS ASSESSMENT       #
################################

def get_fitness(predator, prey):
    '''
    Compare two individuals based on pattern matching, searching substrings
    Fitness is based on the number of elements that match in each place. 
    '''

    # convert individuals to strings
    str_predator = "".join(predator.genome.astype(str))
    str_prey = "".join(prey.genome.astype(str))

    # find the longest substring 
    substring = ""
    match_start_index = 0
    match_end_index = 0
    for start_index in range(len(str_prey)):
        # for end_index in range(1,len(str_prey)):
        for end_index in range(start_index+1, len(str_prey)):
            prey_substring = str_prey[start_index:end_index]
            # is_substring = prey_substring in str_predator
            is_substring = prey_substring == str_predator[start_index:end_index]
            if (is_substring and len(prey_substring)>len(substring)):
                substring = prey_substring
                match_start_index = start_index
                match_end_index = end_index

    fitness = len(substring)
    
    # return the fitness based on the length of the substring found
    return fitness, [match_start_index, match_end_index]
    # return len(substring), prey.genome==predator.genome


################################
#             ARCHIVE          #
################################

def update_archive(solution_archive, individual, max_archive_length):
    
    solution_archive = sorted(solution_archive, key=lambda individual: individual.novelty)
#     print ([i.novelty for i in solution_archive[0:5]])
    
    if len(solution_archive) < max_archive_length:
        solution_archive.append(individual)
    elif solution_archive[0].novelty < individual.novelty:
        solution_archive.pop(0)
        solution_archive.append(individual)
    return solution_archive


################################
#           NOVELTY            #
################################

def get_novelty(solution_archive, individual, k):
    
    archive_size = len(solution_archive)
    
    if archive_size < k: k = archive_size-1
    distance_to_new_genome = np.zeros(archive_size)
    for i in range(archive_size):
        distance_to_new_genome[i] = np.sum(np.abs(solution_archive[i].genome - individual.genome))
    distance_to_new_genome = sorted(distance_to_new_genome)
#     print(distance_to_new_genome[1:k+1])
#     print("novelty:", np.mean(distance_to_new_genome[1:k+1]))
    return np.mean(distance_to_new_genome[1:k+1])


################################
#       PLOTTING RESULTS       #
################################

def plot_mean_and_bootstrapped_ci_over_time(input_data = None, name = "change me", title = "change me", x_label = "change me", y_label="change me", y_limit = None, plot_bootstrap = True):
    """
    
    parameters: 
    input_data: (numpy array of shape (max_k, num_repitions)) solution metric to plot
    name: (string) name for legend
    x_label: (string) x axis label
    y_label: (string) y axis label
    
    returns:
    None
    """

    fig, ax = plt.subplots() # generate figure and axes

    if isinstance(name, str): name = [name]; input_data = [input_data]

    # for this_input_data, this_name in zip(input_data, name):
    for this_name in name:
        print(f"plotting {this_name}...")
        this_input_data = input_data[this_name]
        total_generations = this_input_data.shape[1]

        if plot_bootstrap:
            boostrap_ci_generation_found = np.zeros((2,total_generations))
            for this_gen in range(total_generations):
#                 if this_gen%10==0: print(this_gen)
                boostrap_ci_generation_found[:,this_gen] = bootstrap.ci(this_input_data[:,this_gen], np.mean, alpha=0.05)

        ax.set_title(f'{y_label} over generations {title}')
        ax.plot(np.arange(total_generations), np.mean(this_input_data,axis=0), label = this_name) # plot the fitness over time
        if plot_bootstrap:
            ax.fill_between(np.arange(total_generations), boostrap_ci_generation_found[0,:], boostrap_ci_generation_found[1,:],alpha=0.3) # plot, and fill, the confidence interval for fitness over time
        ax.set_xlabel(x_label) # add axes labels
        ax.set_ylabel(y_label)
        if y_limit: ax.set_ylim(y_limit[0],y_limit[1])
        plt.legend(loc='best'); # add legend

    # plt.show()
    plt.savefig(f"{y_label}_over_generations_{title}.png")
        

if __name__ == '__main__':
    main()