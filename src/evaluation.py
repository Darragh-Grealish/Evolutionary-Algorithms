import random, numpy as np, time, math, logging, os
from concurrent.futures import ThreadPoolExecutor
from src.models import TreeNode
from src.grammar import map_genotype, GRAMMAR

fitness_cache = {} # map expression to fitness
logger = logging.getLogger(__name__)

PRE_OPS = {
    'sin': lambda x: math.sin(x),
    'cos': lambda x: math.cos(x),
    'exp': lambda x: math.exp(min(x, 70)),  # prevent overflow
    'log': lambda x: math.log(x) if x > 1e-10 else 0.0,
    'inv': lambda x: 1.0 / x if abs(x) > 1e-10 else 0.0
}

def safe_div(a, b):
    return a / b if abs(b) > 1e-10 else 0.0

def evaluate_population(population, X, y, cfg):
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:
        results = list(
            ex.map(
                lambda ind: eval_individual(ind, X, y, cfg),
                population
            )
        )
    logger.info("Evaluation time: %.4fs", time.perf_counter() - start)
    return results

def eval_individual(individual, X, y, cfg):
    # phenotype
    phenotype = map_genotype(
        grammar=GRAMMAR,
        genotype=individual['genotype'],
        axiom="start",
        max_depth=cfg.max_depth,
    )

    # cache key
    key = tuple((nt, tuple(individual['genotype'].get(nt, []))) 
                for nt in sorted(GRAMMAR.keys()))

    fit = fitness_cache.get(key)
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
