import json
import time
import sympy as sp
import numpy as np
from pathlib import Path
from copy import deepcopy
from multiprocessing import Pool
from algorithms.utils import get_variables, EarlyStopping
from algorithms.genetic_algorithm import genetic
from algorithms.diff_evolution_algorithm import diff_ev
from algorithms.particle_algorithm import particle_swarm

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

def create_tasks(i, comp, preloaded_ranges, sectioned_ranges, max_index, population = None):
    comp_copy = deepcopy(comp)
    ranges_copy = preloaded_ranges.copy()
    replace = (sectioned_ranges[i], sectioned_ranges[i+1])
    ranges_copy[max_index] = replace
    if population is not None:
        return((comp_copy, ranges_copy, population[i]))
    else:
        return((comp_copy, ranges_copy))
    


def run_algs(alg_type, config_path, n_processors, n_repeats):
    base_dir = Path(__file__).resolve().parent.parent
    json_path = base_dir / "saved_exp" / config_path
    with open(json_path,'r') as file:
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
    comp_seq = [(GENERATIONS*n_repeats), (POPSIZE*(n_processors-1)*n_repeats), EQUATION, VARIABLES, VARNUM, RESTRICTIONS, OBJECTIVE]
    population = None
    task_populations = None

    results_log = {
        "algorithm": ["genetic", "diff_ev", "particle"][alg_type],
        "sequential": {},
        "parallel": {
            "epochs": []
        },
        "n_processors":None,
        "n_epochs":None
    }

    early_stop = EarlyStopping()
    if alg_type == 0:
        MUTATIONP = data['alg_config']['genetic']['mutation_p']
        comp.append(MUTATIONP)
        comp_seq.append(MUTATIONP)
        #print("Running sequential version")
        start_time = time.time()
        res = genetic(comp_seq, preloaded_ranges)
        #print(f"Best individual found: {res[0]}")
        #print(f"With result {res[1]}")
        seq_time = time.time() - start_time
        #print(f"Time taken: {seq_time}")

        results_log["sequential"] = {
            "best_individual": res[0],
            "best_value": res[1],
            "time_taken": seq_time
        }
        
        #print("Running parallel version")
        start_time = time.time()
        for e in range(n_repeats):
            max_index, sectioned_ranges = create_subarrays(preloaded_ranges, n_processors)
            if population:
                task_populations = []
                population = np.stack(population)
                keys = population[:, max_index]
                indices = np.digitize(keys, sectioned_ranges, right=True) - 1 
                indices = np.clip(indices, 0, len(sectioned_ranges) - 2)
                task_populations = [population[indices == i] for i in range(len(sectioned_ranges)-1)]

            tasks = []

            for i in range(n_processors):
                tasks.append(create_tasks(i,comp, preloaded_ranges, sectioned_ranges, max_index, task_populations))
                
            
            with Pool(n_processors) as pool:
                results = pool.map(run_genetic, tasks)

            best_population_idx = get_best_pop(results, OBJECTIVE)
            best_ind, best_val = results[best_population_idx][:2]

            results_log["parallel"]["epochs"].append({
                "epoch": e + 1,
                "best_individual": best_ind,
                "best_value": best_val
            })

            #print(f"Epoch {e+1}")
            #print("Best Individual:", results[best_population_idx][0])
            #print("Best Value:", results[best_population_idx][1])
            #print("\n")
            population = results[best_population_idx][2]
            if early_stop.stopper(results[best_population_idx][1]):
                results_log["parallel"]["early_stopping_epoch"] = e
                #print(f"Early stopping at epoch {e}")
                break
        parallel_time = time.time() - start_time
        speed_up = seq_time - parallel_time
        efficiency = speed_up / n_processors

        results_log["parallel"]["time_taken"] = parallel_time
        results_log["parallel"]["speed_up"] = speed_up
        results_log["parallel"]["efficiency"] = efficiency
    

    elif alg_type == 1:
        MUTATIONF = data['alg_config']['diff_ev']['mutation_f']
        RECOMBCONST = data['alg_config']['diff_ev']['recomb_const']
        comp.append(MUTATIONF)
        comp.append(RECOMBCONST)
        comp_seq.append(MUTATIONF)
        comp_seq.append(RECOMBCONST)

        #print("Running sequential version")
        start_time = time.time()
        res = diff_ev(comp_seq, preloaded_ranges)
        #print(f"Best individual found: {res[0]}")
        #print(f"With result {res[1]}")
        seq_time = time.time() - start_time
        #print(f"Time taken: {seq_time}")

        results_log["sequential"] = {
            "best_individual": res[0],
            "best_value": res[1],
            "time_taken": seq_time
        }
        
        #print("Running parallel version")
        start_time = time.time()
        
        for e in range(n_repeats):
            max_index, sectioned_ranges = create_subarrays(preloaded_ranges, n_processors)
            #If a population exists, section into sub-ranges to avoid introducing invalid individuals
            if population is not None:
                population = np.stack(population)
                keys = population[:, max_index]
                indices = np.digitize(keys, sectioned_ranges, right=True) - 1 
                indices = np.clip(indices, 0, len(sectioned_ranges) - 2)
                task_populations = [population[indices == i] for i in range(len(sectioned_ranges)-1)]
                    


            tasks = []

            for i in range(n_processors):
                tasks.append(create_tasks(i,comp, preloaded_ranges, sectioned_ranges, max_index, task_populations))     
            
            with Pool(n_processors) as pool:
                results = pool.map(run_ev, tasks)

            best_population_idx = get_best_pop(results, OBJECTIVE)
            best_ind, best_val = results[best_population_idx][:2]

            results_log["parallel"]["epochs"].append({
                "epoch": e + 1,
                "best_individual": best_ind,
                "best_value": best_val
            })

            #print(f"Epoch {e+1}")
            #print("Best Individual:", results[best_population_idx][0])
            #print("Best Value:", results[best_population_idx][1])
            #print("\n")
            population = results[best_population_idx][2]
            if early_stop.stopper(results[best_population_idx][1]):
                results_log["parallel"]["early_stopping_epoch"] = e
                #print(f"Early stopping at epoch {e}")
                break
        parallel_time = time.time() - start_time
        speed_up = seq_time - parallel_time
        efficiency = speed_up / n_processors

        results_log["parallel"]["time_taken"] = parallel_time
        results_log["parallel"]["speed_up"] = speed_up
        results_log["parallel"]["efficiency"] = efficiency
        

    elif alg_type == 2:
        W = data['alg_config']['particle']['w']
        C1 = data['alg_config']['particle']['c1']
        C2 = data['alg_config']['particle']['c2']
        comp.append(W)
        comp.append(C1)
        comp.append(C2)
        comp_seq.append(W)
        comp_seq.append(C1)
        comp_seq.append(C2)

        #print("Running sequential version")
        start_time = time.time()
        res = particle_swarm(comp_seq, preloaded_ranges)
        #print(f"Best individual found: {res[0]}")
        #print(f"With result {res[1]}")
        seq_time = time.time() - start_time
        #print(f"Time taken: {seq_time}")

        results_log["sequential"] = {
            "best_individual": res[0],
            "best_value": res[1],
            "time_taken": seq_time
        }

        #print("Running parallel version")
        start_time = time.time()

        for e in range(n_repeats):
            max_index, sectioned_ranges = create_subarrays(preloaded_ranges, n_processors)

            if population:
                population = np.stack(population)
                extract = [i.current_position for i in population] #as in vanilla extract
                extract = np.stack(extract)
                keys = extract[:, max_index]
                indices = np.digitize(keys, sectioned_ranges, right=True) - 1
                indices = np.clip(indices, 0, len(sectioned_ranges) - 2)
                task_populations = [population[indices == i] for i in range(len(sectioned_ranges)-1)]


            tasks = []

            for i in range(n_processors):
                tasks.append(create_tasks(i,comp, preloaded_ranges, sectioned_ranges, max_index, task_populations))
                
            with Pool(n_processors) as pool:
                results = pool.map(run_particles, tasks)

            best_population_idx = get_best_pop(results, OBJECTIVE)
            best_ind, best_val = results[best_population_idx][:2]

            results_log["parallel"]["epochs"].append({
                "epoch": e + 1,
                "best_individual": best_ind,
                "best_value": best_val
            })

            #print(f"Epoch {e+1}")
            #print("Best Individual:", results[best_population_idx][0])
            #print("Best Value:", results[best_population_idx][1])
            #print("\n")
            population = results[best_population_idx][2]
            if early_stop.stopper(results[best_population_idx][1]):
                results_log["parallel"]["early_stopping_epoch"] = e
               # print(f"Early stopping at epoch {e}")
                break
            
        parallel_time = time.time() - start_time
        speed_up = seq_time - parallel_time
        efficiency = speed_up / n_processors

        results_log["parallel"]["time_taken"] = parallel_time
        results_log["parallel"]["speed_up"] = speed_up
        results_log["parallel"]["efficiency"] = efficiency
    
    return results_log


if __name__ == "__main__":
    run_algs(1, "./saved_exp/Sample.json", 8, 5)