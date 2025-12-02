import numpy as np
import random
from src.grammar import parse_bnf, genome_to_expression
from src.evaluation import fitness, random_genome, crossover, mutate


def run_ge(X, y, pop_size, generations, genome_len, max_depth):
    grammar = parse_bnf()
    population = [random_genome(genome_len) for _ in range(pop_size)] # genome is random list of 100 integers
    for gen in range(generations):
        fitnesses = [fitness(g, grammar, X, y, max_depth) for g in population]
        idxs = np.argsort(fitnesses)
        population = [population[i] for i in idxs] # sort by fitness
        print(f"Gen {gen}: Best Fitness {fitnesses[idxs[0]]:.2f} Expr: {genome_to_expression(population[0], grammar, max_depth)}")
        
        # Elitism
        new_pop = population[:2]
        # Generate rest of new pop with crossover/mutation
        while len(new_pop) < pop_size:
            p1 = random.choice(population[:10]) # select from top 10
            p2 = random.choice(population[:10])
            c1, c2 = crossover(p1, p2)
            c1 = mutate(c1)
            c2 = mutate(c2)
            new_pop.extend([c1, c2])
        population = new_pop[:pop_size]
    # Return best genome & expression
    best_g = population[0]
    best_expr = genome_to_expression(best_g, grammar, max_depth)
    return best_g, best_expr
