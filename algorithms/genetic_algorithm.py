import sympy as sp
import numpy as np
from utils import clip_individual, check_restrictions, evaluate_fitness, generate_valid_individual, EarlyStopping


def create_population(POPSIZE, RESTRICTIONS, preloaded_ranges, VARIABLES):
    population = []

    for _ in range(POPSIZE):
        individual = generate_valid_individual(preloaded_ranges, RESTRICTIONS, VARIABLES)

        population.append(individual)
    return population


def get_winners(population, pop_fitness, NWINNERS, OBJECTIVE):
    if len(population) == 0 or len(pop_fitness) == 0:
        print("Warning: Population or fitness array is empty when calling get_winners.")
        return [], population

    pop_fitness = np.array(pop_fitness)
    population = np.array(population)

    if OBJECTIVE == 'MAX':
        winner_indices = np.argpartition(-pop_fitness, range(NWINNERS))[:NWINNERS]
    else:
        winner_indices = np.argpartition(pop_fitness, range(NWINNERS))[:NWINNERS]

    winners = population[winner_indices].tolist()

    # Create mask to remove winners from original population
    mask = np.ones(len(population), dtype=bool)
    mask[winner_indices] = False
    remaining_population = population[mask].tolist()

    return winners, remaining_population

def mutate(population, MUTATIONP, preloaded_ranges, RESTRICTIONS, VARIABLES):
    max_attempts = 10
    for i in range(len(population)):
        attempts = 0
        safety = population[i].copy()#Make a safety copy to avoid introducing bad individuals into population
        while True: 
            attempts += 1
            if attempts > max_attempts:
                population[i] = safety
                break
            
            for j, _ in enumerate(population[i]):
                if np.random.rand() < MUTATIONP:
                    min_bound, max_bound = preloaded_ranges[j]
                    #Could probably make this section more efficient by storing all the ranges in memory
                    #Full reset mutation rarely ever worked
                    #Using creep mutation to explore already confirmed stable regions
                    creep_range = 0.1 * (max_bound - min_bound)
                    mutation = np.random.uniform(-creep_range, creep_range)
                    population[i][j] += mutation
                    population[i][j] = np.clip(population[i][j], min_bound, max_bound)

            population[i] = clip_individual(population[i], preloaded_ranges)    
            if(check_restrictions(population[i],RESTRICTIONS,VARIABLES)):
                break
    return population

def create_children(parents, VARNUM, RESTRICTIONS, preloaded_ranges, VARIABLES, max_regenerations=10):
    children = []
    parent_array = np.array(parents)
    np.random.shuffle(parent_array)  # ensures full coverage with no repeats

    num_pairs = len(parent_array) // 2

    for i in range(num_pairs):
        parent1 = parent_array[2 * i]
        parent2 = parent_array[2 * i + 1]

        # Create two children with at least one differing gene (index)
        swap_indices = np.random.rand(VARNUM) < 0.5
        if not np.any(swap_indices):  # ensure at least one swap
            swap_indices[np.random.randint(0, VARNUM)] = True

        child1 = np.where(swap_indices, parent2, parent1).copy()
        child1 = clip_individual(child1, preloaded_ranges)
        child2 = np.where(swap_indices, parent1, parent2).copy()
        child2 = clip_individual(child2, preloaded_ranges)

        for child, parent in zip([child1, child2], [parent1, parent2]):
            attempts = 0
            while not (check_restrictions(child.tolist(), RESTRICTIONS, VARIABLES)):
                attempts += 1
                if attempts >= max_regenerations:
                    child[:] = parent  # fall back to parent if stuck
                    break

                for j in range(VARNUM):
                    min_bound, max_bound = preloaded_ranges[j]
                    #If generated child is not valid, try and mutate it to see if it becomes valid
                    #This really shows that this particular algorithm was not made for continuous value problems
                    creep_range = 0.1 * (max_bound - min_bound)
                    mutation = np.random.uniform(-creep_range, creep_range)
                    child[j] += mutation
                child[:] = clip_individual(child, preloaded_ranges)


        children.append(child1.tolist())
        children.append(child2.tolist())

    return children

                    
                    
                    
def genetic(compact, ranges, population = None):
    GENERATIONS, POPSIZE, EQUATION, VARIABLES, VARNUM, RESTRICTIONS, OBJECTIVE, MUTATIONP  = compact
    preloaded_ranges = ranges
    
    NWINNERS = int(np.floor(POPSIZE/2))
    #Population ratios
    NELITE = int(POPSIZE * 0.2)
    NCHILDREN = int(POPSIZE * 0.6)
    NMUTATE = POPSIZE - NELITE - NCHILDREN

    if population == None:
        population = create_population(POPSIZE,RESTRICTIONS,preloaded_ranges,VARIABLES)

    early_stop = EarlyStopping()
    for g in range(GENERATIONS):
        pop_fitness = []
        for individual in population:
                pop_fitness.append(evaluate_fitness(individual, EQUATION, VARIABLES))
        elites, population = get_winners(population, pop_fitness, NWINNERS, OBJECTIVE)
        #Mutate population after extracting winners
        population = mutate(population, MUTATIONP, preloaded_ranges, RESTRICTIONS, VARIABLES)
        population = population[:NMUTATE]
        #Generate children
        children = create_children(elites, VARNUM, RESTRICTIONS, preloaded_ranges, VARIABLES)
        children = children[:NCHILDREN]
        elites = elites[:NELITE] #Clip all population sections to avoid bloat
        #Form final population for epoch
        population.extend(children)
        population.extend(elites)
        best_fitness = min(evaluate_fitness(ind, EQUATION, VARIABLES) for ind in population) if OBJECTIVE == 'MIN' else \
                   max(evaluate_fitness(ind, EQUATION, VARIABLES) for ind in population)
        if early_stop.stopper(best_fitness):
            #print(f"Early stopping at generation {g}")
            break

    best_individual = min(population, key=lambda ind: evaluate_fitness(ind, EQUATION, VARIABLES)) if OBJECTIVE == 'MIN' else max(population, key=lambda ind: evaluate_fitness(ind, EQUATION, VARIABLES))
    best_value = evaluate_fitness(best_individual, EQUATION, VARIABLES)


    return best_individual, best_value, population



if __name__ == "__main__":
    genetic("./saved_exp/Sample.json")
