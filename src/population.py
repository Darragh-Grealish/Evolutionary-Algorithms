import random, re, os
from typing import List, Dict
from src.models import TreeNode, Grammar

# Initialize grammar from BNF file
GRAMMAR = Grammar(os.path.join(os.path.dirname(__file__), "grammar.bnf"))

def is_recursive(nt, prod):
    return nt in prod # return true is LHS is in RHS

def choose_production(grammar, nt, depth, max_depth):
    """
    Choose a production rule index for a non-terminal
    """
    prods = grammar[nt] # get all posible productions for this non-terminal
    if depth >= max_depth: # at max depth, avoid recursive productions
        nonrec = [i for i,p in enumerate(prods) if not is_recursive(nt, p)]
        return random.choice(nonrec) if nonrec else random.randrange(len(prods))
    return random.randrange(len(prods)) # otherwise choose any production

def initialise_individual(grammar, start_nt, max_depth, rng=random):
    """
    Create a structured genotype: dict mapping non-terminals to lists of chosen productions.
    This aligns with DSGE where each non-terminal has its own gene list.
    """
    genotype = {nt: [] for nt in grammar.keys()}
    def expand(nt, depth):
        idx = choose_production(grammar, nt, depth, max_depth)
        genotype[nt].append(idx) # append index of rule chosen
        prod = grammar[nt][idx]
        for sym in prod:
            if sym in grammar: # recursively built out non-terminals
                expand(sym, depth+1)
    expand(start_nt, 0)
    return genotype # genotype is dict of lists of production indices for each non-terminal

def map_genotype(grammar, genotype, start_nt, max_depth, expression_cache=None, rng=random):
    """
    Map a genotype (list of production indices) to a phenotype tree (TreeNode).
    The grammar determines arity: 'op' is binary, 'pre_op' is unary, 'var' and literals are terminals.
    """
    # read positions index for each non-terminal's gene list
    cursors = {nt: 0 for nt in grammar.keys()}

    # key for cache is tuple of all non-terminals and their production indices
    key = tuple((nt, tuple(genotype.get(nt, []))) for nt in sorted(grammar.keys()))
    if expression_cache is not None and key in expression_cache: # we have already mapped this genotype
        return expression_cache[key]

    def expand(nt, depth):
        prods = grammar[nt] # get productions for this non-terminal
        cur = cursors[nt]   # current read position in gene list for this non-terminal (this tells us what rule to choose)
        if cur < len(genotype.get(nt, [])):
            idx = genotype[nt][cur] % len(prods)
            cursors[nt] += 1
        else:
            # Dynamically extend gene list for this non-terminal when exhausted (DSGE behavior)
            idx = choose_production(grammar, nt, depth, max_depth)
            genotype.setdefault(nt, []).append(idx)
            cursors[nt] += 1

        prod = prods[idx]

        # Build subtree according to the production
        # Productions are sequences of symbols; we recursively expand non-terminals
        nodes = []
        for sym in prod:
            if sym in grammar: # non-terminal
                nodes.append(expand(sym, depth+1))
            else: # terminal
                nodes.append(TreeNode(sym))

        # Collapse parentheses patterns like ["(", expr, op, expr, ")"] into op node
        if nt == "expr":
            # Handle parenthesized binary
            if len(nodes) == 5 and nodes[0].symbol == "(" and nodes[4].symbol == ")":
                return TreeNode(nodes[2].symbol, [nodes[1], nodes[3]])
            if len(nodes) == 4 and nodes[1].symbol == "(" and nodes[3].symbol == ")":
                return TreeNode(nodes[0].symbol, [nodes[2]])
            if len(nodes) == 3 and nodes[1].symbol in ['+', '-', '*', '/']:
                return TreeNode(nodes[1].symbol, [nodes[0], nodes[2]])
            if len(nodes) == 1:
                return nodes[0]
            return TreeNode("seq", nodes)

        if nt == "op" or nt == "pre_op" or nt == "var":
            return nodes[0]
        if nt == "start":
            return nodes[0] if nodes else TreeNode("seq", [])

        return TreeNode(nt, nodes)

    tree = expand(start_nt, 0)
    
    # Update shared cache
    final_key = tuple((nt, tuple(genotype.get(nt, []))) for nt in sorted(grammar.keys()))
    if expression_cache is not None:
        expression_cache[final_key] = tree
        
    return tree

def initialise_population(config, start_nt="start", rng=random):
   """
    Create a list of individuals, each a dict:
        { 'genotype': [...], 'phenotype': [...], 'fitness': None }
    """
    population = []

    for _ in range(config.population_size):
        genotype = initialise_individual(
            grammar=GRAMMAR,
            start_nt=start_nt,
            max_depth=config.max_depth,
            rng=rng
        )

        individual = {
            "genotype": genotype,
            "phenotype": None,
            "fitness": None
        }

        population.append(individual)

    return population