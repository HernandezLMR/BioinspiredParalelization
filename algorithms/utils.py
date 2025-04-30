import re
import sympy as sp
import numpy as np
#Old method of checking if and individual had its values within acceptable ranges, no longer used
#Now all values are preemtively clipped to avoid excessive checking and trashing of possibly better answers
def check_ranges(individual, preloaded_ranges):
    for i, val in enumerate(individual):
        min_val, max_val = preloaded_ranges[i]
        if val < min_val or val > max_val:
            return False
    return True

def get_variables(EQUATION):
    variables = EQUATION.free_symbols
    def sort_key(symbol):
        match = re.match(r"x(\d+)", str(symbol))
        return int(match.group(1)) if match else float('inf')
    sorted_vars = sorted(variables, key=sort_key)
    return sorted_vars

def check_restrictions(individual, RESTRICTIONS, VARIABLES):
    valid = True
    for restriction in RESTRICTIONS:
        expr = sp.sympify(restriction)
        sub_dict = {
            var : individual[i]
            for i,var in enumerate(VARIABLES) 
        }
        if not expr.subs(sub_dict):
            valid = False
            break
    return valid

def evaluate_fitness(individual, FUNCTION, VARIABLES):
    sub_dict = {
            var : individual[i]
            for i,var in enumerate(VARIABLES) 
        }
    return FUNCTION.subs(sub_dict).evalf()

def generate_valid_individual(preloaded_ranges, RESTRICTIONS, VARIABLES):
    while True:
        individual = np.array([
            np.random.uniform(min_bound, max_bound) for (min_bound, max_bound) in preloaded_ranges
        ])
        if check_restrictions(individual, RESTRICTIONS, VARIABLES):
            return individual

def clip_individual(individual, preloaded_ranges):
    clip_min = np.array([low for (low, _) in preloaded_ranges])
    clip_max = np.array([high for (_, high) in preloaded_ranges])
    return np.clip(individual, clip_min, clip_max)


class EarlyStopping:
    def __init__(self, patience=10, epsilon=1e-6):
        self.patience = patience
        self.epsilon = epsilon
        self.counter = 0
        self.best_score = None

    def stopper(self, current_score):
        if self.best_score is None or abs(current_score - self.best_score) > self.epsilon:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience
