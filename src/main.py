# imports
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


def main():

    alg_landscape = Landscape.Landscape(n=10)
    fitness, solutions, diversity = evolutionary_algorithm(fitness_function=alg_landscape.get_fitness, total_generations=100, \
                num_parents=10, num_children=10, bit_string_length=10, num_elements_to_mutate=1, crossover=False)
    print(f'fit: {fitness[-1]} | sol: {solutions[-1]} | div: {diversity[-1]}')


    experiment_results = {}
    solutions_results = {}
    diversity_results = {}


    num_runs = 20
    total_generations = 100
    num_elements_to_mutate = 1
    bit_string_length = 15
    num_parents = 20
    num_children = 20

    n = bit_string_length
    k = bit_string_length - 1

    run_name = "baseline"

    experiment_results[run_name] = np.zeros((num_runs, total_generations))
    solutions_results[run_name] = np.zeros((num_runs, total_generations, bit_string_length))
    diversity_results[run_name] = np.zeros((num_runs, total_generations))



    for run_num in range(num_runs):
        start_time = time.time()
        # run the algorithm
        alg_landscape = Landscape.Landscape(n=n, k=k)
        fitness, solutions, diversity = evolutionary_algorithm(fitness_function=alg_landscape.get_fitness, total_generations=total_generations, \
                num_parents=num_parents, num_children=num_children, bit_string_length=bit_string_length, num_elements_to_mutate=num_elements_to_mutate, crossover=False)
        
        # save the results
        experiment_results[run_name][run_num] = fitness
        solutions_results[run_name][run_num] = solutions
        diversity_results[run_name][run_num] = diversity
        print(run_name, run_num, time.time()-start_time, fitness[-1])

    # plotting
    data_names = ["baseline"]

    plot_mean_and_bootstrapped_ci_over_time(input_data = experiment_results, name = data_names, x_label = "Generation", y_label = "Fitness", plot_bootstrap = True)
    plot_mean_and_bootstrapped_ci_over_time(input_data = diversity_results, name = data_names, x_label = "Generation", y_label = "Diversity", plot_bootstrap = True)




def evolutionary_algorithm(fitness_function=None, total_generations=100, num_parents=10, num_children=10, bit_string_length=10, num_elements_to_mutate=1, crossover=True):
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
    
    ################################
    ### INITILIZATING POPULATION ###
    ################################

    # create the predator 
    predator = Individual.Individual

    # create an initial population 
    population = [] # keep population of individuals in a list
    for i in range(num_parents): # only create parents for initialization (the mu in mu+lambda)
        population.append(Individual.Individual(fitness_function,bit_string_length)) # generate new random individuals as parents
    
    # get population fitness
    for i in range(len(population)):
        population[i].eval_fitness() # evaluate the fitness of each parent
    
    for generation_num in range(total_generations): # repeat
        
        # the modification procedure
        new_children = [] # keep children separate for now (lambda in mu+lambda)
        while len(new_children) < num_children:
            
            # inheretance
            [parent1, parent2] = np.random.choice(population, size=2) # pick 2 random parents
            child1 = copy.deepcopy(parent1) # initialize children as perfect copies of their parents
            child2 = copy.deepcopy(parent2)
            
            # crossover
            if crossover:
                [crossover_point1, crossover_point2] = sorted(np.random.randint(0,bit_string_length,2)) # crossover points for 2-point crossover (sorted to make indexing easier in the next step)
                child1.genome[crossover_point1:crossover_point2+1] = parent2.genome[crossover_point1:crossover_point2+1] # take the point between the crossover points and swap in the genes from the other parent
                child2.genome[crossover_point1:crossover_point2+1] = parent1.genome[crossover_point1:crossover_point2+1]

            # mutation
            for this_child in [child1,child2]:
                elements_to_mutate = set()
                while len(elements_to_mutate)<num_elements_to_mutate:
                    elements_to_mutate.add(np.random.randint(bit_string_length)) # randomly select the location in the child bit string to mutate
                for this_element_to_mutate in elements_to_mutate:
                    this_child.genome[this_element_to_mutate] = (this_child.genome[this_element_to_mutate] + 1) % 2 # flip the bit at the chosen location
            
            new_children.extend((child1,child2)) # add children to the new_children list
            
        # the assessement procedure
        for i in range(len(new_children)):
            new_children[i].eval_fitness() # assign fitness to each child 

        # selection procedure
        population += new_children # combine parents with new children (the + in mu+lambda)
        population = sorted(population, key=lambda individual: individual.fitness, reverse=True) # sort the full population by each individual's fitness (from highers to lowest)
        population = population[:num_parents] # perform truncation selection (keep just top mu individuals to become next set of parents)
        
        # record keeping
        
        if population[0].fitness > solution_fitness: # if the new parent is the best found so far
            solution = population[0].genome                 # update best solution records
            solution_fitness = population[0].fitness
            solution_generation = generation_num
        fitness_over_time[generation_num] = solution_fitness # record the fitness of the current best over evolutionary time
        solutions_over_time[generation_num,:] = solution
        
        genome_list = np.array([individual.genome for individual in population])
        diversity = np.mean(genome_list.std(axis=0))
        diversity_over_time[generation_num] = diversity
        
        
    return fitness_over_time, solutions_over_time, diversity_over_time 




def plot_mean_and_bootstrapped_ci_over_time(input_data = None, name = "change me", x_label = "change me", y_label="change me", y_limit = None, plot_bootstrap = True):
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


        ax.plot(np.arange(total_generations), np.mean(this_input_data,axis=0), label = this_name) # plot the fitness over time
        if plot_bootstrap:
            ax.fill_between(np.arange(total_generations), boostrap_ci_generation_found[0,:], boostrap_ci_generation_found[1,:],alpha=0.3) # plot, and fill, the confidence interval for fitness over time
        ax.set_xlabel(x_label) # add axes labels
        ax.set_ylabel(y_label)
        if y_limit: ax.set_ylim(y_limit[0],y_limit[1])
        plt.legend(loc='best'); # add legend
        

if __name__ == '__main__':
    main()