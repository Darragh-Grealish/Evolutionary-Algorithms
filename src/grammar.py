import random, re, os
from typing import List, Dict
from src.models import TreeNode, Grammar

# Initialize grammar from BNF file
GRAMMAR = Grammar(os.path.join(os.path.dirname(__file__), "grammar.bnf"))

genome_to_expression_cache = {}

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
        genotype[nt].append(idx)
        prod = grammar[nt][idx]
        for sym in prod:
            if sym in grammar: # recursively built out non-terminals
                expand(sym, depth+1)

    expand(start_nt, 0)
    return genotype # genotype is dict of lists of production indices for each non-terminal

def map_genotype(grammar, genotype, start_nt, max_depth, rng=random):
    """
    Map a genotype (list of production indices) to a phenotype tree (TreeNode).
    The grammar determines arity: 'op' is binary, 'pre_op' is unary, 'var' and literals are terminals.
    """
    # read positions index for each non-terminal's gene list
    cursors = {nt: 0 for nt in grammar.keys()}

    # key for cache is tuple of all non-terminals and their production indices
    key = tuple((nt, tuple(genotype.get(nt, []))) for nt in sorted(grammar.keys()))
    if key in genome_to_expression_cache: # we have already mapped this genotype
        return genome_to_expression_cache[key]

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
                left = nodes[1]
                op = nodes[2]
                right = nodes[3]
                return TreeNode(op.symbol, [left, right])

            # Handle unary pre_op with parentheses
            if len(nodes) == 4 and nodes[1].symbol == "(" and nodes[3].symbol == ")":
                pre = nodes[0]
                arg = nodes[2]
                return TreeNode(pre.symbol, [arg])

            # Handle direct binary: expr op expr
            if len(nodes) == 3 and nodes[1].symbol in ['+', '-', '*', '/']:
                left = nodes[0]
                op = nodes[1]
                right = nodes[2]
                return TreeNode(op.symbol, [left, right])

            # Handle var
            if len(nodes) == 1:
                return nodes[0]

            # Fallback: create a sequence node
            return TreeNode("seq", nodes)

        if nt == "op":
            # op produces a single terminal operator
            return nodes[0]

        if nt == "pre_op":
            # pre_op produces a single terminal pre-operator
            return nodes[0]

        if nt == "var":
            # var produces a single terminal variable/literal
            return nodes[0]

        if nt == "start":
            # start -> expr, unwrap to return the expression tree directly
            return nodes[0] if nodes else TreeNode("seq", [])

        # Default: pack nodes as sequence
        return TreeNode(nt, nodes)

    tree = expand(start_nt, 0)
    # cache using the final, possibly-extended genotype
    final_key = tuple((nt, tuple(genotype.get(nt, []))) for nt in sorted(grammar.keys()))
    genome_to_expression_cache[final_key] = tree
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
