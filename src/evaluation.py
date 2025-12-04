import random, numpy as np, time

fitness_cache = {} # map expression to fitness

def safe_div(a, b):
    return a / b if b != 0 else 0.0

def evaluate_population(population, X, y, cfg):
    for genome in population:
        start = time.time()

        # Use existing phenotype if available
        expr = genome.get('phenotype')
        if expr is None:
            print("Error: Phenotype missing")

        key = tuple(genome['genotype'])

        # --- Fitness (cached) ---
        fit = fitness_cache.get(expr)
        if fit is None:
            preds = np.array([evaluate_expression(expr, s) for s in X])
            fit = np.sqrt(np.mean((preds - y) ** 2))
            fitness_cache[expr] = fit

        genome['fitness'] = fit
        genome['eval_time'] = time.time() - start

    return population

def evaluate_expression(expr_str, sample):
    """
    Evaluate an expression with a single sample
    """
    safe_dict = {k: sample[i] for i, k in enumerate([
        'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
        'view', 'condition', 'sqft_above', 'sqft_basement',
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
        # print("Error evaluating expression:", expr_str)
        return 0.0 # populate with blank flaot

def get_expr(genome, max_depth):
    key = tuple(genome)
    if key not in expr_cache: # new genome, need to build expression from scratch
        expr_cache[key] = genome_to_expression(genome, max_depth)
    return expr_cache[key]

def fitness(genome, X, y, max_depth):
    # expr = get_expr(genome, max_depth)
    expr = genome_to_expression(genome, max_depth)
    print(f"Expr {expr}")
    if expr not in fitness_cache: # new expression, not evaluated before
        preds = [evaluate_expression(expr, f) for f in X]
        preds = np.array(preds)
        print(f"Predictions: {preds[:20]}")
        fitness_cache[expr] = np.sqrt(np.mean((preds - y)**2))
    return fitness_cache[expr]

def random_genome(length):
    return [random.randint(0, 255) for _ in range(length)]

def get_dynamic_mutation_rate(best_fitness, worst_fitness, min_rate, max_rate):
    # avoid division by zero or negative scales
    if worst_fitness <= 0:
        return max_rate

    # progress score: 0 = no progress, 1 = big improvement
    s = (worst_fitness - best_fitness) / worst_fitness
    s = max(0.0, min(1.0, s))  # clamp to [0, 1]

    # interpolate: when s=0 -> max_rate, when s=1 -> min_rate
    return max_rate - s * (max_rate - min_rate)

def mutate(genome, mutation_rate):
    return [g if random.random() > mutation_rate else random.randint(0, 255) for g in genome]

# uniform crossover
def crossover(parent1, parent2, crossover_rate):
    child1 = []
    child2 = []
    
    for g1, g2 in zip(parent1, parent2):
        if random.random() < crossover_rate:
            # parent1->child1, parent2->child2
            child1.append(g1)
            child2.append(g2)
        else:
            # parent2->child1, parent1->child2
            child1.append(g2)
            child2.append(g1)
    
    return child1, child2