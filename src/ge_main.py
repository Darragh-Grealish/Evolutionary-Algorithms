import numpy as np, random, time, logging
from src.grammar import initialise_population, genome_to_expression_cache
from src.evaluation import evaluate_population, fitness_cache
from src.genetic_operators import crossover_individuals, mutate_genotype
from src.visualisation import plot_generation_times

logger = logging.getLogger(__name__)

def run_ge(X, y, cfg):
    generation_times = []

    # --- Initial population ---
    population = initialise_population(cfg)

    for gen in range(cfg.generations):

        # --- Evaluate current population ---
        population = evaluate_population(population, X, y, cfg)

        # --- Sort by fitness ---
        population.sort(key=lambda g: g['fitness'])
        best = population[0]
        logger.info("Gen %d: Best Fitness %.4f Expr: %s",
                    gen, best['fitness'], best['phenotype'])

        # --- Elitism ---
        new_pop = [dict(ind) for ind in population[:cfg.elitism_count]]
        parents = population[:cfg.top_parents_count]

        # --- Reproduction ---
        while len(new_pop) < cfg.population_size:
            p1, p2 = random.choice(parents), random.choice(parents)

            c1, c2 = crossover_individuals(p1, p2)
            c1g = mutate_genotype(c1['genotype'], max_depth=cfg.max_depth)
            c2g = mutate_genotype(c2['genotype'], max_depth=cfg.max_depth)

            new_pop.append({'genotype': c1g, 'phenotype': None, 'fitness': None})
            if len(new_pop) < cfg.population_size:
                new_pop.append({'genotype': c2g, 'phenotype': None, 'fitness': None})
        population = new_pop

    # --- Final sort & best genome ---
    population = evaluate_population(population, X, y, cfg)
    population.sort(key=lambda g: g['fitness'])
    best = population[0:10] # return top 10 genomes
    logger.info("Best 10 Genomes:")
    for i, genome in enumerate(best):
        logger.info("Rank %d: Fitness %.4f Expr: %s", i+1, genome['fitness'], genome['phenotype'])
    logger.info("\n")

    # --- Log Caches To Show One:One Mapping ---
    if (len(fitness_cache)) != (len(genome_to_expression_cache)):
        logger.warning("Warning: Fitness cache size (%d) does not match genome to expression cache size (%d)", len(fitness_cache), len(genome_to_expression_cache))
    else: # 1:1 mapping - DSGE working as intended
        logger.info("Fitness cache size: %d", len(fitness_cache))
        logger.info("Genome to expression cache size: %d", len(genome_to_expression_cache))

    # Clear caches for next step
    fitness_cache.clear()
    genome_to_expression_cache.clear()

    return best
