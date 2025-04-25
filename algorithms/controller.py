import json
import sympy as sp
import numpy as np
from copy import deepcopy
from multiprocessing import Pool
from utils import get_variables, EarlyStopping
from genetic_algorithm import genetic
from diff_evolution_algorithm import diff_ev
from particle_algorithm import particle_swarm

def range_difference(r):
    return r[1] - r[0]

def run_genetic(args):
    comp_args, local_ranges, *optional = args
    if optional:
        population = optional[0]
        return genetic(comp_args, local_ranges, population)
    return genetic(comp_args, local_ranges)

def run_ev(args):
    comp_args, local_ranges, *optional = args
    if optional:
        population = optional[0]
        return diff_ev(comp_args, local_ranges, population)
    return diff_ev(comp_args, local_ranges)

def run_particles(args):
    comp_args, local_ranges, *optional = args
    if optional:
        population = optional[0]
        return particle_swarm(comp_args, local_ranges, population)
    return particle_swarm(comp_args, local_ranges)

def create_subarrays(ranges, n_processors):
    max_index = max(range(len(ranges)), key=lambda i: range_difference(ranges[i]))
    sectioned_ranges = np.linspace(ranges[max_index][0], ranges[max_index][1], n_processors+1)

    return max_index, sectioned_ranges

def get_best_pop(results, OBJECTIVE):
    if OBJECTIVE == "MIN":
        return min(range(len(results)), key=lambda i: results[i][1])
    else:
        return max(range(len(results)), key=lambda i: results[i][1])



def main(alg_type, config_path, n_processors, n_repeats):
    with open(config_path,'r') as file:
        data = json.load(file)
    GENERATIONS = data['generations']
    POPSIZE = data['pop_size']
    EQUATION = sp.sympify(data['task_config']['eq']) 
    VARIABLES = get_variables(EQUATION)
    VARNUM = len(VARIABLES)
    RESTRICTIONS = data['task_config']['restrictions']
    OBJECTIVE = data['task_config']['obj'] 
    RANGES = data['task_config']['ranges'] #dict type
    preloaded_ranges = [(RANGES[f'x{i}']['min'], RANGES[f'x{i}']['max']) for i in range(VARNUM)]

    comp = [GENERATIONS, POPSIZE, EQUATION, VARIABLES, VARNUM, RESTRICTIONS, OBJECTIVE]
    population = None

    early_stop = EarlyStopping()
    if alg_type == 0:
        MUTATIONP = data['alg_config']['genetic']['mutation_p']
        comp.append(MUTATIONP)
        for e in range(n_repeats):
            max_index, sectioned_ranges = create_subarrays(preloaded_ranges, n_processors)

            tasks = []

            for i in range(n_processors):
                comp_copy = deepcopy(comp)
                ranges_copy = preloaded_ranges.copy()
                replace = (sectioned_ranges[i], sectioned_ranges[i+1])
                ranges_copy[max_index] = replace
                if population is not None:
                    tasks.append((comp_copy, ranges_copy, population))
                else:
                    tasks.append((comp_copy, ranges_copy))
                
            
            with Pool(n_processors) as pool:
                results = pool.map(run_genetic, tasks)

            best_population_idx = get_best_pop(results, OBJECTIVE)

            print(f"Epoch {e+1}")
            print("Best Individual:", results[best_population_idx][0])
            print("Best Value:", results[best_population_idx][1])
            print("\n")
            population = results[best_population_idx][2]
            if early_stop.stopper(results[best_population_idx][1]):
                print(f"Early stopping at epoch {e}")
                break
    

    elif alg_type == 1:
        MUTATIONF = data['alg_config']['diff_ev']['mutation_f']
        RECOMBCONST = data['alg_config']['diff_ev']['recomb_const']
        comp.append(MUTATIONF)
        comp.append(RECOMBCONST)


        for e in range(n_repeats):
            max_index, sectioned_ranges = create_subarrays(preloaded_ranges, n_processors)

            tasks = []

            for i in range(n_processors):
                comp_copy = deepcopy(comp)
                ranges_copy = preloaded_ranges.copy()
                replace = (sectioned_ranges[i], sectioned_ranges[i+1])
                ranges_copy[max_index] = replace
                if population is not None:
                    tasks.append((comp_copy, ranges_copy, population))
                else:
                    tasks.append((comp_copy, ranges_copy))
                
            
            with Pool(n_processors) as pool:
                results = pool.map(run_ev, tasks)

            best_population_idx = get_best_pop(results, OBJECTIVE)

            print(f"Epoch {e+1}")
            print("Best Individual:", results[best_population_idx][0])
            print("Best Value:", results[best_population_idx][1])
            print("\n")
            population = results[best_population_idx][2]
            if early_stop.stopper(results[best_population_idx][1]):
                print(f"Early stopping at epoch {e}")
                break
        

    elif alg_type == 2:
        W = data['alg_config']['particle']['w']
        C1 = data['alg_config']['particle']['c1']
        C2 = data['alg_config']['particle']['c2']
        comp.append(W)
        comp.append(C1)
        comp.append(C2)

        for e in range(n_repeats):
            max_index, sectioned_ranges = create_subarrays(preloaded_ranges, n_processors)

            tasks = []

            for i in range(n_processors):
                comp_copy = deepcopy(comp)
                ranges_copy = preloaded_ranges.copy()
                replace = (sectioned_ranges[i], sectioned_ranges[i+1])
                ranges_copy[max_index] = replace
                if population is not None:
                    tasks.append((comp_copy, ranges_copy, population))
                else:
                    tasks.append((comp_copy, ranges_copy))
                
            
            with Pool(n_processors) as pool:
                results = pool.map(run_particles, tasks)

            best_population_idx = get_best_pop(results, OBJECTIVE)

            print(f"Epoch {e+1}")
            print("Best Individual:", results[best_population_idx][0])
            print("Best Value:", results[best_population_idx][1])
            print("\n")
            population = results[best_population_idx][2]
            if early_stop.stopper(results[best_population_idx][1]):
                print(f"Early stopping at epoch {e}")
                break
            


if __name__ == "__main__":
    main(1, "./saved_exp/Sample.json", 8, 10)