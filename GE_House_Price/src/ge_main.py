import numpy as np, random, time, secrets
from src.grammar import genome_to_expression
from src.evaluation import fitness, random_genome, crossover, mutate, get_dynamic_mutation_rate, expr_cache, fitness_cache
from src.visualisation import plot_generation_times

def run_ge(X, y, cfg):
    generation_times = []
    population = [random_genome(cfg.genome_length) for _ in range(cfg.population_size)] # initial population
    for gen in range(cfg.generations):
        start = time.perf_counter() # start timer
        fitnesses = [fitness(g, X, y, cfg.max_depth) for g in population] # build expression and evaluate fitness of each genome
        idxs = np.argsort(fitnesses) # get sorted indexes
        population = [population[i] for i in idxs] # sort by fitness
        fitnesses = [fitnesses[i] for i in idxs] # apply sorting to fitness to match population
        best_f = fitnesses[0]
        worst_f = fitnesses[-1]

        best_expr_str = genome_to_expression(population[0], cfg.max_depth)
        print(f"Gen {gen}: Best Fitness {fitnesses[idxs[0]]:.2f} Expr: {best_expr_str}")


        mutation_rate = get_dynamic_mutation_rate(best_fitness=best_f, worst_fitness=worst_f, min_rate=cfg.min_mutation_rate, max_rate=cfg.max_mutation_rate)
        print(f"  Dynamic mutation rate this gen: {mutation_rate:.4f}")
        
        # Elitism
        new_pop = population[:cfg.elitism_count] 
        parents = population[:cfg.top_parents_count]
        # Generate rest of new pop with crossover/mutation
        while len(new_pop) < cfg.population_size:
            p1 = random.choice(parents) # select from top parents
            p2 = random.choice(parents)
            c1, c2 = crossover(p1, p2, cfg.crossover_rate)
            c1 = mutate(c1, mutation_rate)
            c2 = mutate(c2, mutation_rate)
            new_pop.extend([c1, c2])
        population = new_pop[:cfg.population_size]
        print(f"Generation {gen} completed in {time.perf_counter() - start:.4f}s")
        generation_times.append(time.perf_counter() - start)
    # Return best genome & expression
    best_g = population[0]
    best_expr = genome_to_expression(best_g, cfg.max_depth)    
    print(f"Expr cache size: {len(expr_cache)}, Fitness cache size: {len(fitness_cache)}")
    
    plot_generation_times(generation_times)
    # Clear caches as next step will be evaluating fitness on test set
    expr_cache.clear()
    fitness_cache.clear()
    return best_g, best_expr
