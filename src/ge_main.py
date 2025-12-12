import numpy as np, random, logging
from multiprocessing import Manager, Pool, cpu_count
from src.population import initialise_population
from src.evaluation import evaluate_population
from src.genetic_operators import crossover_individuals, mutate_genotype

logger = logging.getLogger(__name__)

def genome_to_tuple(genotype):
    """Convert genotype dict to a hashable tuple for uniqueness checking."""
    return tuple(sorted((k, tuple(v)) for k, v in genotype.items()))

def run_ge(X, y, cfg):
    with Manager() as manager, Pool(processes=cpu_count()) as pool:
        # Create shared process-safe dictionaries
        fitness_cache = manager.dict()
        genome_to_expression_cache = manager.dict()

        generation_times = []

        # --- Initial population ---
        population = initialise_population(cfg)

        for gen in range(cfg.generations):

            # --- Evaluate current population ---
            population = evaluate_population(
                population, X, y, cfg, 
                pool, fitness_cache, genome_to_expression_cache
            )

            # --- Sort by fitness ---
            population.sort(key=lambda g: g['fitness'])
            best = population[0]
            logger.info("Gen %d: Best Fitness %.4f Expr: %s",
                        gen, best['fitness'], best['phenotype'])

            # --- Elitism ---
            new_pop = [dict(ind) for ind in population[:cfg.elitism_count]]
            parents = population[:cfg.top_parents_count]
        
            # Create set to track unique genomes in new population
            genome_set = {genome_to_tuple(ind['genotype']) for ind in new_pop}

            # --- Reproduction ---
            while len(new_pop) < cfg.population_size:
                p1, p2 = random.choice(parents), random.choice(parents)

                c1, c2 = crossover_individuals(p1, p2)
                c1g = mutate_genotype(c1['genotype'], max_depth=cfg.max_depth)
                c2g = mutate_genotype(c2['genotype'], max_depth=cfg.max_depth)

                while genome_to_tuple(c1g) in genome_set:
                    c1g = mutate_genotype(c1g, max_depth=cfg.max_depth)
                genome_set.add(genome_to_tuple(c1g))
                new_pop.append({'genotype': c1g, 'phenotype': None, 'fitness': None})
                
                if len(new_pop) < cfg.population_size:
                    # Ensure c2g is unique - keep mutating until it is
                    while genome_to_tuple(c2g) in genome_set:
                        c2g = mutate_genotype(c2g, max_depth=cfg.max_depth)
                    
                    genome_set.add(genome_to_tuple(c2g))
                    new_pop.append({'genotype': c2g, 'phenotype': None, 'fitness': None})
            population = new_pop

        # --- Final sort & best genome ---
        population = evaluate_population(
            population, X, y, cfg, 
            pool, fitness_cache, genome_to_expression_cache
        )
        population.sort(key=lambda g: g['fitness'])
        best_ten = population[0:10]  # return top 10 genomes
        logger.info("Best 10 Genomes:")
        for i, genome in enumerate(best_ten):
            logger.info("Rank %d: Fitness %.4f Expr: %s", i+1, genome['fitness'], genome['phenotype'])
        logger.info("\n")

        # --- Log Caches To Show One:One Mapping ---
        if (len(fitness_cache)) != (len(genome_to_expression_cache)):
            logger.warning("Warning: Fitness cache: %d, Expr cache: %d", 
                           len(fitness_cache), len(genome_to_expression_cache))
        else:
            logger.info("Fitness cache size: %d", len(fitness_cache))
            logger.info("Genome to expression cache size: %d", len(genome_to_expression_cache))

        return best_ten