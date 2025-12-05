import numpy as np, random, time
from src.grammar import initialise_population
from src.evaluation import (
    evaluate_population,
    fitness_cache
)
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
        while len(new_pop) < cfg.population_size:
            p1, p2 = random.choice(parents), random.choice(parents)

            c1, c2 = crossover(p1['genotype'], p2['genotype'], cfg.crossover_rate)
            c1 = mutate(c1, mutation_rate)
            c2 = mutate(c2, mutation_rate)

            new_pop.append({'genotype': c1, 'phenotype': None, 'fitness': None})
            if len(new_pop) < cfg.population_size:
                new_pop.append({'genotype': c2, 'phenotype': None, 'fitness': None})

        # --- Evaluate new population ---
        population = evaluate_population(new_pop, X, y, cfg)

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
