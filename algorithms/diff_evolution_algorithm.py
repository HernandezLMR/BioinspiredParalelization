import numpy as np
import random
import sympy as sp
from utils import clip_individual, check_restrictions, evaluate_fitness, generate_valid_individual, EarlyStopping


def create_population(POPSIZE, RESTRICTIONS, preloaded_ranges, VARIABLES):
    population = []

    for _ in range(POPSIZE):
        individual = generate_valid_individual(preloaded_ranges,RESTRICTIONS, VARIABLES)
        population.append(individual)
    return population

def process_ind(target, child, EQUATION, RESTRICTIONS, OBJECTIVE,VARIABLES):
    #Check if child is valid
    
    if not check_restrictions(child, RESTRICTIONS, VARIABLES):
        return False

    child_value = evaluate_fitness(child, EQUATION, VARIABLES)
    target_value = evaluate_fitness(target, EQUATION, VARIABLES)

    win = (child_value > target_value) if OBJECTIVE == 'MAX' else (child_value < target_value)

    return win


def diff_ev(compact, ranges, population = None):
    GENERATIONS, POPSIZE, EQUATION, VARIABLES, VARNUM, RESTRICTIONS, OBJECTIVE, MUTATIONF, RECOMBCONST  = compact
    preloaded_ranges = ranges

    
    
    #Create initial population
    if population == None:
        population = create_population(POPSIZE,RESTRICTIONS,preloaded_ranges,VARIABLES)            
    


    #Main algorithm
    early_stop = EarlyStopping()
    for n in range(GENERATIONS):
        for i, target in enumerate(population):
            populationEX = [ind for j, ind in enumerate(population) if j != i]
            values = random.sample(populationEX, 3)
            child = np.add(values[0],(MUTATIONF * (np.subtract(values[1],values[2]))))
            recombined = np.where(np.random.rand(VARNUM) < RECOMBCONST, child, target)
            recombined = clip_individual(recombined, preloaded_ranges)

            if (process_ind(target, recombined, EQUATION, RESTRICTIONS, OBJECTIVE, VARIABLES)):

                population[i] = recombined
        best_fitness = min(evaluate_fitness(ind, EQUATION, VARIABLES) for ind in population) if OBJECTIVE == 'MIN' else \
                   max(evaluate_fitness(ind, EQUATION, VARIABLES) for ind in population)

        if early_stop.stopper(best_fitness):
            #print(f"Early stopping at generation {n}")
            break

                
    best_individual = min(population, key=lambda ind: evaluate_fitness(ind, EQUATION, VARIABLES)) if OBJECTIVE == 'MIN' else max(population, key=lambda ind: evaluate_fitness(ind, EQUATION, VARIABLES))
    best_value = evaluate_fitness(best_individual, EQUATION, VARIABLES)

    
    return best_individual, best_value, population


                




if __name__ == "__main__":
    diff_ev("./saved_exp/Sample.json")