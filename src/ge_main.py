import numpy as np, random, time, secrets
from src.genome import Genome
from src.grammar import genome_to_expression
from src.evaluation import evaluate_expression, fitness, random_genome, crossover, mutate, get_dynamic_mutation_rate, expr_cache, fitness_cache
from src.visualisation import plot_generation_times

def run_ge(X, y, cfg):
    generation_times = []
    population = [Genome(random_genome(cfg.genome_length)) for _ in range(cfg.population_size)] # initial population
    for gen in range(cfg.generations):
        start = time.perf_counter() # start timer

        for ind in population:
            # compute fitness on genotype
            ind.fitness = fitness(ind.genotype, X, y, cfg.max_depth)
            # cache phenotype
            ind.phenotype = genome_to_expression(ind.genotype, cfg.max_depth)

        population.sort(key=lambda g: g.fitness) # sort by fitness
        best_f = population[0].fitness
        worst_f = population[-1].fitness
        best_expr_str = population[0].phenotype
        print(f"Gen {gen}: Best Fitness {best_f:.2f} Expr: {best_expr_str}")


        # for i in enumerate(population):
        #     print("EXPR:", i[1].phenotype)
        #     print("FITNESS:", i[1].fitness)
        #     print("PREDS:", [evaluate_expression(i[1].phenotype, f) for f in X[:5]])
        #     print("\n")


        mutation_rate = get_dynamic_mutation_rate(best_fitness=best_f, worst_fitness=worst_f, min_rate=cfg.min_mutation_rate, max_rate=cfg.max_mutation_rate)
        print(f"  Dynamic mutation rate this gen: {mutation_rate:.4f}")
        
        # Elitism
        new_pop = [Genome(ind.genotype[:]) for ind in population[:cfg.elitism_count]]
        parents = population[:cfg.top_parents_count]
        # Generate rest of new pop with crossover/mutation
        while len(new_pop) < cfg.population_size:
            p1 = random.choice(parents) # select from top parents
            p2 = random.choice(parents)
            c1, c2 = crossover(p1.genotype, p2.genotype, cfg.crossover_rate)
            c1 = mutate(c1, mutation_rate)
            c2 = mutate(c2, mutation_rate)
            new_pop.append(Genome(c1))
            if len(new_pop) < cfg.population_size: # avoid overfill when pop size is odd
                new_pop.append(Genome(c2))

        population = new_pop
        timer = time.perf_counter() - start
        print(f"Generation {gen} completed in {timer:.4f}s")
        generation_times.append(timer)

    for ind in population: # evaluate children fitness/phenotype at end of final generation
        if ind.fitness is None:
            ind.fitness = fitness(ind.genotype, X, y, cfg.max_depth)
            ind.phenotype = genome_to_expression(ind.genotype, cfg.max_depth)

    # Return best genome & expression
    population.sort(key=lambda g: g.fitness) # ensure sorted
    best_g = population[0].genotype
    best_expr = population[0].phenotype 


    print(f"Expr cache size: {len(expr_cache)}, Fitness cache size: {len(fitness_cache)}")
    
    plot_generation_times(generation_times)
    # Clear caches as next step will be evaluating fitness on test set
    expr_cache.clear()
    fitness_cache.clear()
    return best_g, best_expr
