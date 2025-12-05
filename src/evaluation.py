import random, numpy as np, time, math
from src.models import TreeNode
from src.grammar import map_genotype, GRAMMAR

fitness_cache = {} # map expression to fitness

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
    for genome in population:
        start = time.time()

        tree = genome.get('phenotype')
        if tree is None:
            # Map genotype to phenotype tree if missing
            print("Error - phenotype missing")
            genotype, tree = map_genotype(
                grammar=GRAMMAR,
                genotype=genome['genotype'],
                axiom="start",
                max_depth=cfg.max_depth,
            )
            genome['genotype'] = genotype
            genome['phenotype'] = tree

        # Build a hashable key from structured genes
        key = tuple((nt, tuple(genome['genotype'].get(nt, []))) for nt in sorted(GRAMMAR.keys()))

        # --- Fitness (cached) ---
        fit = fitness_cache.get(key)
        if fit is None:
            preds = np.array([eval_tree(tree, {k: s[i] for i, k in enumerate(cfg.feature_names)}) for s in X])
            fit = np.sqrt(np.mean((preds - y) ** 2))
            fitness_cache[key] = fit
        
        genome['fitness'] = fit
        genome['eval_time'] = time.time() - start

    return population

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
            print(f"Failed to parse tree {node}")
            return 0.0  # fallback
    else:
        return float(node)  # node is already terminal

def get_expr(genome, max_depth):
    key = tuple(genome)
    if key not in expr_cache: # new genome, need to build expression from scratch
        expr_cache[key] = genome_to_expression(genome, max_depth)
    return expr_cache[key]
