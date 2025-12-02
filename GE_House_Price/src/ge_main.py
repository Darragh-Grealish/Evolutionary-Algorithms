import numpy as np
import random
from src.grammar import genome_to_expression
from src.evaluation import fitness, random_genome, crossover, mutate


def run_ge(X, y, cfg):
    population = [random_genome(cfg.genome_length) for _ in range(cfg.population_size)] # genome is random list of 100 integers
    for gen in range(cfg.generations):
        fitnesses = [fitness(g, X, y, cfg.max_depth) for g in population]
        idxs = np.argsort(fitnesses)
        population = [population[i] for i in idxs] # sort by fitness

        best_expr_str = genome_to_expression(population[0], cfg.max_depth)
        print(f"Gen {gen}: Best Fitness {fitnesses[idxs[0]]:.2f} Expr: {best_expr_str}")
        
        # Elitism
        new_pop = population[:cfg.elitism_count]
        # Generate rest of new pop with crossover/mutation
        while len(new_pop) < cfg.population_size:
            p1 = random.choice(population[:cfg.top_parents_count]) # select from top parents
            p2 = random.choice(population[:cfg.top_parents_count])
            c1, c2 = crossover(p1, p2)
            c1 = mutate(c1)
            c2 = mutate(c2)
            new_pop.extend([c1, c2])
        population = new_pop[:cfg.population_size]
    # Return best genome & expression
    best_g = population[0]
    best_expr = genome_to_expression(best_g, cfg.max_depth)    
    return best_g, best_expr
