import re
import numpy as np
import random
import json
import sympy as sp
from utils import clip_individual, get_variables, check_restrictions, evaluate_fitness, generate_valid_individual


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


def diff_ev(config_path):
    with open(config_path,'r') as file:
        data = json.load(file)
        GENERATIONS = data['generations']
        POPSIZE = data['pop_size']
        EQUATION = sp.sympify(data['task_config']['eq']) 

        VARIABLES = get_variables(EQUATION)
        VARNUM = len(VARIABLES)

        OBJECTIVE = data['task_config']['obj'] 
        RANGES = data['task_config']['ranges'] #dict type
        preloaded_ranges = [(RANGES[f'x{i}']['min'], RANGES[f'x{i}']['max']) for i in range(VARNUM)] #Store in memory to avoid constat lookup
        RESTRICTIONS = data['task_config']['restrictions'] #array type
        #Parameters for differential evolution algorithm
        MUTATIONF = data['alg_config']['diff_ev']['mutation_f']
        RECOMBCONST = data['alg_config']['diff_ev']['recomb_const']

        
        
        #Create initial population
        population = create_population(POPSIZE,RESTRICTIONS,preloaded_ranges,VARIABLES)            
        


        #Main algorithm
        for n in range(GENERATIONS):
            for i, target in enumerate(population):
                populationEX = [ind for j, ind in enumerate(population) if j != i]
                values = random.sample(populationEX, 3)
                child = np.add(values[0],(MUTATIONF * (np.subtract(values[1],values[2]))))
                recombined = np.where(np.random.rand(VARNUM) < RECOMBCONST, child, target)
                recombined = clip_individual(recombined, preloaded_ranges)

                if (process_ind(target, child, EQUATION, RESTRICTIONS, OBJECTIVE, VARIABLES)):

                    population[i] = recombined

                    
    #Old logging feature, will replace in later version
        minimun = float('inf') if OBJECTIVE == 'MIN' else float('-inf')  # Initialize for MIN or MAX
        best_individual = None

        for individual in population:
            value = evaluate_fitness(individual, EQUATION, VARIABLES)
            
            if OBJECTIVE == 'MIN' and value < minimun:
                minimun = value
                best_individual = individual
            elif OBJECTIVE == 'MAX' and value > minimun:
                minimun = value
                best_individual = individual

    print(f"El valor {'minimo' if OBJECTIVE == 'MIN' else 'maximo'} encontrado es: {minimun}")
    print(f"El individuo con el valor {'minimo' if OBJECTIVE == 'MIN' else 'maximo'} es: {best_individual}")


                




if __name__ == "__main__":
    diff_ev("./saved_exp/Sample.json")