import numpy as np, random, time
from src.grammar import initialise_population
from src.evaluation import evaluate_population,fitness_cache
from src.genetic_operators import crossover_individuals, mutate_genotype
from src.visualisation import plot_generation_times

def run_ge(X, y, cfg):
    generation_times = []

    # --- Initial population ---
    population = initialise_population(cfg)  # list of dicts

    for gen in range(cfg.generations):
        start = time.perf_counter()
        population = evaluate_population(population, X, y, cfg)

        # --- Sort by fitness ---
        population.sort(key=lambda g: g['fitness'])
        best = population[0]
        best_f, worst_f = best['fitness'], population[-1]['fitness']
        print(f"Gen {gen}: Best Fitness {best_f:.4f} Expr: {best['phenotype']}")

        # --- Elitism ---
        new_pop = [dict(genome) for genome in population[:cfg.elitism_count]]
        parents = population[:cfg.top_parents_count]

        # --- Fill remaining population ---
        start_fill = time.perf_counter()
        while len(new_pop) < cfg.population_size:
            p1, p2 = random.choice(parents), random.choice(parents)

            # DSGE-style gene-level uniform crossover (whole-gene swap via mask)
            c1_ind, c2_ind = crossover_individuals(p1, p2)

            c1g = mutate_genotype(c1_ind['genotype'], max_depth=cfg.max_depth)
            c2g = mutate_genotype(c2_ind['genotype'], max_depth=cfg.max_depth)

            new_pop.append({'genotype': c1g, 'phenotype': None, 'fitness': None})
            if len(new_pop) < cfg.population_size:
                new_pop.append({'genotype': c2g, 'phenotype': None, 'fitness': None})
        print(f"Filling new population time: {time.perf_counter() - start_fill:.4f}s")

        # --- Evaluate new population ---
        start_eval = time.perf_counter()
        population = evaluate_population(new_pop, X, y, cfg)
        print(f"Evaluation time: {time.perf_counter() - start_eval:.4f}s")

        elapsed = time.perf_counter() - start
        generation_times.append(elapsed)
        print(f"Generation {gen} completed in {elapsed:.4f}s")

    # --- Final sort & best genome ---
    population.sort(key=lambda g: g['fitness'])
    best = population[0]

    print(f"Fitness cache size: {len(fitness_cache)}")
    plot_generation_times(generation_times)

    # Clear caches for next step
    fitness_cache.clear()

    return best['genotype'], best['phenotype']
