import random, numpy as np, time, math, logging, os
from multiprocessing import Pool, cpu_count
from src.models import TreeNode
from src.grammar import map_genotype, GRAMMAR

fitness_cache = {} # map expression to fitness
logger = logging.getLogger(__name__)

EPS = 1e-10
MAX_MAG = 1e6

def clamp(v, m=MAX_MAG): # max magnitude in either direction
    return max(min(v, m), -m)

PRE_OPS = {
    'sin': lambda x: math.sin(x),
    'cos': lambda x: math.cos(x),
    'exp': lambda x: clamp(math.exp(min(x, 70))), # min x, 70 to avoid overflow
    'log': lambda x: math.log(max(x, EPS)), # log undefined for <=0, use EPS
    'inv': lambda x: clamp(1.0 / max(abs(x), EPS) * (1 if x >= 0 else -1)) # safe inverse (avoid div by 0, but maintain sign)
}

def safe_div(a, b):
    return clamp(a / max(abs(b), EPS) * (1 if b >= 0 else -1))

def _eval_individual_wrapper(args):
    """Wrapper function for multiprocessing that unpacks arguments."""
    individual, X, y, cfg = args
    return eval_individual(individual, X, y, cfg)

def evaluate_population(population, X, y, cfg):
    start = time.perf_counter()
    with Pool(processes=cpu_count()) as pool:
        # Create argument tuples for each individual
        args_list = [(ind, X, y, cfg) for ind in population]
        results = pool.map(_eval_individual_wrapper, args_list)
    logger.info("Evaluation time: %.4fs", time.perf_counter() - start)
    return results

def eval_individual(individual, X, y, cfg):
    phenotype = map_genotype(
        grammar=GRAMMAR,
        genotype=individual['genotype'],
        start_nt="start",
        max_depth=cfg.max_depth,
    )

    # cache key
    key = tuple((nt, tuple(individual['genotype'].get(nt, []))) 
                for nt in sorted(GRAMMAR.keys()))

    fit = fitness_cache.get(key) # check if already evaluated
    if fit is None:
        preds = np.array([
            eval_tree(phenotype, {k: s[i] for i, k in enumerate(cfg.feature_names)})
            for s in X
        ])
        fit = np.sqrt(np.mean((preds - y) ** 2))
        fitness_cache[key] = fit

    return {
        "genotype": individual["genotype"],
        "phenotype": phenotype,
        "fitness": fit,
    }

def eval_tree(node, sample):
    """
    Recursively evaluate a TreeNode using a sample row.
    Handles pre_ops, safe division, arithmetic, and variables/literals.
    """
    if isinstance(node, TreeNode):
        sym = node.symbol

        # Pre-ops
        if sym in PRE_OPS:
            return PRE_OPS[sym](eval_tree(node.children[0], sample))

        # Arithmetic ops
        if sym in '+-*':
            a = eval_tree(node.children[0], sample)
            b = eval_tree(node.children[1], sample)
            return {'+': a+b, '-': a-b, '*': a*b}[sym]

        if sym == '/':
            a = eval_tree(node.children[0], sample)
            b = eval_tree(node.children[1], sample)
            return safe_div(a, b)

        # Structural helpers: parentheses or sequence nodes
        if sym == '(':
            return eval_tree(node.children[0], sample) if node.children else 0.0
        if sym == ')':
            return eval_tree(node.children[0], sample) if node.children else 0.0
        if sym == 'seq':
            # Evaluate the last meaningful child
            for child in reversed(node.children):
                val = eval_tree(child, sample)
                return val
            return 0.0

        if sym == 'start':
            # Unwrap start node to its child expression if present
            return eval_tree(node.children[0], sample) if node.children else 0.0

        # Terminal: variable or literal
        if sym in sample:
            return sample[sym]
        try:
            return float(sym)
        except ValueError:
            logger.error("Failed to parse tree %s", node)
            return 0.0  # fallback
    else:
        return float(node)  # node is already terminal

def get_expr(genome, max_depth):
    key = tuple(genome)
    if key not in expr_cache: # new genome, need to build expression from scratch
        expr_cache[key] = genome_to_expression(genome, max_depth)
    return expr_cache[key]

def evaluate_top_individuals_on_test(best_individuals, X_test, y_test, cfg):
    logger.info("=" * 80)
    logger.info("EVALUATING TOP 10 INDIVIDUALS ON TEST DATASET")
    logger.info("=" * 80)

    results = []
    
    for rank, individual in enumerate(best_individuals, 1):
        # Make predictions on test set
        y_pred = np.array([
            eval_tree(
                individual['phenotype'],
                {k: row[i] for i, k in enumerate(cfg.feature_names)}
            )
            for row in X_test
        ])
        
        rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
        avg_absolute_error = np.mean(np.abs(y_pred - y_test)) # average prediction error
        
        # Store results
        result = {
            'rank': rank,
            'genotype': individual['genotype'],
            'phenotype': individual['phenotype'],
            'train_fitness': individual['fitness'],
            'test_rmse': rmse,
            'avg_absolute_error': avg_absolute_error
        }
        results.append(result)
    
    logger.info("\n" + "=" * 80)
    logger.info("TEST SET EVALUATION SUMMARY")
    logger.info("=" * 80)
    logger.info("  Test samples: %d", len(y_test))
    
    best_on_test = min(results, key=lambda x: x['test_rmse']) # Find best performing on test set
    logger.info("  Best individual on test set: Rank %d", best_on_test['rank'])
    logger.info("  Test RMSE: %.4f", best_on_test['test_rmse'])
    logger.info("  Average Absolute Error: $%.2f", best_on_test['avg_absolute_error'])
    
    return results
