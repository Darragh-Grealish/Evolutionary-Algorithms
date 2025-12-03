import random
import numpy as np
from src.grammar import genome_to_expression

# --- GLOBAL CACHES ---
expr_cache = {}
fitness_cache = {}

def safe_div(a, b):
    return a / b if b != 0 else 0.0

def evaluate_expression(expr_str, sample):
    """
    Evaluate an expression with a single sample
    """
    safe_dict = {k: sample[i] for i, k in enumerate([
        'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
        'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement',
        'yr_built', 'yr_renovated', 'city_num', 'statezip_num', 'country_num'
    ])}

    safe_dict["__builtins__"] = None
    safe_dict["sd"] = safe_div

    # replace `/` with `sd(...)` form
    expr_str = expr_str.replace('/', ' sd ')
    
    try:
        result = eval(expr_str, {}, safe_dict)
        return float(result)
    except Exception:
        return 0.0 # populate with blank flaot

def get_expr(genome, max_depth):
    key = tuple(genome)
    if key not in expr_cache: # new genome, need to built expression from scratch
        expr_cache[key] = genome_to_expression(genome, max_depth)
    return expr_cache[key]

def fitness(genome, X, y, max_depth):
    key = tuple(genome) # convert list to immutable tuple so it can be used as dict key
    if key not in fitness_cache: # new genome, not evaluated before
        expr_str = get_expr(genome, max_depth)
        preds = [evaluate_expression(expr_str, f) for f in X]
        preds = np.array(preds)
        fitness_cache[key] = np.sqrt(np.mean((preds - y)**2))
    return fitness_cache[key]

def random_genome(length=20):
    return [random.randint(0, 255) for _ in range(length)]

def mutate(genome, mutation_rate=0.1):
    return [g if random.random() > mutation_rate else random.randint(0, 255) for g in genome]

def crossover(parent1, parent2):
    point = random.randint(1, len(parent1)-1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2