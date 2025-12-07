import numpy as np, random, time, logging
from src.grammar import initialise_population, genome_to_expression_cache
from src.evaluation import evaluate_population, fitness_cache
from src.genetic_operators import crossover_individuals, mutate_genotype
from src.visualisation import plot_generation_times

logger = logging.getLogger(__name__)

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
        logger.info("Gen %d: Best Fitness %.4f Expr: %s", gen, best_f, best['phenotype'])

        # --- Elitism ---
        new_pop = [dict(genome) for genome in population[:cfg.elitism_count]]
        parents = population[:cfg.top_parents_count]

        # --- Fill remaining population ---
        while len(new_pop) < cfg.population_size:
            p1, p2 = random.choice(parents), random.choice(parents)

            # DSGE-style gene-level uniform crossover (whole-gene swap via mask)
            c1_ind, c2_ind = crossover_individuals(p1, p2)

            c1g = mutate_genotype(c1_ind['genotype'], max_depth=cfg.max_depth)
            c2g = mutate_genotype(c2_ind['genotype'], max_depth=cfg.max_depth)

            new_pop.append({'genotype': c1g, 'phenotype': None, 'fitness': None})
            if len(new_pop) < cfg.population_size:
                new_pop.append({'genotype': c2g, 'phenotype': None, 'fitness': None})

        # --- Evaluate new population ---
        start_eval = time.perf_counter()
        population = evaluate_population(new_pop, X, y, cfg)
        logger.info("Evaluation time: %.4fs", time.perf_counter() - start_eval)
        elapsed = time.perf_counter() - start
        generation_times.append(elapsed)
        logger.info("Generation %d completed in %.4fs", gen, elapsed)

    # --- Final sort & best genome ---
    population.sort(key=lambda g: g['fitness'])
    best = population[0:10] # return top 10 genomes
    logger.info("Best 10 Genomes:")
    for i, genome in enumerate(best):
        logger.info("Rank %d: Fitness %.4f Expr: %s", i+1, genome['fitness'], genome['phenotype'])

    # --- Log Caches To Show One:One Mapping ---
    logger.info("Fitness cache size: %d", len(fitness_cache))
    logger.info("Genome to expression cache size: %d", len(genome_to_expression_cache))
    plot_generation_times(generation_times)

    # Clear caches for next step
    fitness_cache.clear()
    genome_to_expression_cache.clear()

    return best
