import numpy as np
import json
import sympy as sp
from utils import clip_individual, check_restrictions, evaluate_fitness, generate_valid_individual, EarlyStopping

class particle:
    def __init__(self, preloaded_ranges, VARIABLES, EQUATION, RESTRICTIONS):
        self.current_position = generate_valid_individual( preloaded_ranges, RESTRICTIONS, VARIABLES)
            
        self.direction = np.array([(np.random.uniform(min_bound, max_bound) * 0.1) for (min_bound, max_bound) in preloaded_ranges])
        self.current_value = evaluate_fitness(self.current_position, EQUATION, VARIABLES)
        self.best_value = self.current_value
        self.best_position = self.current_position.copy()
    
    def update(self, best_global_value, best_global_position, OBJECTIVE, preloaded_ranges, EQUATION, RESTRICTIONS, VARIABLES, W, C1, C2):
        #Update position
        self.current_position = np.add(self.current_position, self.direction)
        # Clamp each dimension of the position within its interval
        self.current_position = clip_individual(self.current_position, preloaded_ranges)
        #Check if answer is valid
        valid = check_restrictions(self.current_position, RESTRICTIONS, VARIABLES)


        #Update best values if necessary
        if valid:
            #Update value
            self.current_value = evaluate_fitness(self.current_position, EQUATION, VARIABLES)
            if OBJECTIVE == 'MIN' and self.current_value < self.best_value:
                self.best_value = self.current_value
                self.best_position = self.current_position.copy()
            elif OBJECTIVE == 'MAX' and self.current_value > self.best_value:
                self.best_value = self.current_value
                self.best_position = self.current_position.copy()

            # Update global best only if the particle is valid
            if OBJECTIVE == 'MIN' and self.best_value < best_global_value:
                best_global_value = self.best_value
                best_global_position = self.best_position.copy()
            elif OBJECTIVE == 'MAX' and self.best_value > best_global_value:
                best_global_value = self.best_value
                best_global_position = self.best_position.copy()
        #Get new velocity
        r1 = np.random.uniform(0,1)
        r2 = np.random.uniform(0,1)
        if best_global_position is None:
            self.direction = W * self.direction
        else:
            self.direction = (W * self.direction) + (C1*r1*(np.subtract(self.best_position, self.current_position))) + (C2*r2*(np.subtract(best_global_position, self.current_position)))
        return best_global_value, best_global_position
    
def particle_swarm(compact, ranges, population = None):
    GENERATIONS, POPSIZE, EQUATION, VARIABLES, VARNUM, RESTRICTIONS, OBJECTIVE, W, C1, C2  = compact
    preloaded_ranges = ranges
    

    best_global_value = None
    best_global_position = None
    if population is None:
        particles = [particle(preloaded_ranges, VARIABLES, EQUATION, RESTRICTIONS) for _ in range(POPSIZE)]
    else:
        particles = population.tolist()
        if len(particles) < POPSIZE:
            particles.extend(particle(preloaded_ranges, VARIABLES, EQUATION, RESTRICTIONS) for _ in range(POPSIZE-len(population)))
    
    if OBJECTIVE == 'MIN':
        best_global_value = np.inf
    else:
        best_global_value = -np.inf

    #Get initial best global value
    for p in particles:

        if OBJECTIVE == 'MIN' and p.current_value < best_global_value:
            best_global_value = p.current_value
            best_global_position = p.current_position.copy()
        elif OBJECTIVE == 'MAX' and p.current_value > best_global_value:
            best_global_value = p.current_value
            best_global_position = p.current_position.copy()

    early_stop = EarlyStopping()
    for gen in range(1, GENERATIONS+1):
        for p in particles:
            best_global_value, best_global_position = p.update(best_global_value, best_global_position, OBJECTIVE, preloaded_ranges, EQUATION, RESTRICTIONS, VARIABLES, W, C1, C2)
        if early_stop.stopper(best_global_value):
            #print(f"Early stopping at generation {gen}")
            break
    return best_global_position, best_global_value, particles
    
    
if __name__ == '__main__':
    particle_swarm("./saved_exp/Sample.json")

