import random, re, os
from typing import List, Dict

Symbol = str
Production = List[Symbol]
Grammar = Dict[Symbol, List[Production]]

GRAMMAR = {
    "start": [["expr"]],
    "expr": [
        ["expr", "op", "expr"],
        ["(", "expr", "op", "expr", ")"],
        ["pre_op", "(", "expr", ")"],
        ["var"]
    ],
    "op": [["+"], ["-"], ["*"], ["/"]],
    "pre_op": [["sin"], ["cos"], ["exp"], ["inv"], ["log"]],
    "var": [["x"], ["1.0"]]
}

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

def initialise_individual(grammar, axiom, max_depth, rng=random):
    """
    Create a random genotype by expanding the grammar from the axiom
    """
    genotype = []

    def expand(nt, depth):
        idx = choose_production(grammar, nt, depth, max_depth)
        genotype.append(idx) # append index of chosen production
        prod = grammar[nt][idx]
        for sym in prod:
            if sym in grammar: # recursively built out non-terminals
                expand(sym, depth+1)

    expand(axiom, 0)
    return genotype # genotype is list of production indices

def map_genotype(grammar, genotype, axiom, max_depth, rng=random):
    """
    Map a genotype (list of production indices) to a phenotype (expression)
    """
    out = []
    read_pos = 0

    if tuple(genotype) in genome_to_expression_cache:
        return genotype, genome_to_expression_cache[tuple(genotype)]

    def expand(nt, depth):
        nonlocal read_pos

        prods = grammar[nt]

        if read_pos < len(genotype):
            idx = genotype[read_pos] % len(prods)
            read_pos += 1
        else:
            idx = choose_production(grammar, nt, depth, max_depth)
            genotype.append(idx)
            read_pos += 1

        prod = prods[idx]
        for sym in prod:
            if sym in grammar:
                expand(sym, depth+1)
            else:
                out.append(sym)

    expand(axiom, 0)
    genome_to_expression_cache[tuple(genotype)] = ''.join(out) # add mapping to cache
    return genotype, ''.join(out)

def initialise_population(config, axiom="start", rng=random):
    """
    Create a list of individuals, each a dict:
        { 'genotype': [...], 'phenotype': [...], 'fitness': None }
    """
    population = []

    for _ in range(config.population_size):
        genotype = initialise_individual(
            grammar=GRAMMAR,
            axiom=axiom,
            max_depth=config.max_depth,
            rng=rng
        )

        genotype, phenotype = map_genotype(
            grammar=GRAMMAR,
            genotype=genotype,
            axiom=axiom,
            max_depth=config.max_depth,
            rng=rng
        )

        individual = {
            "genotype": genotype,
            "phenotype": phenotype,
            "fitness": None
        }

        population.append(individual)

    return population

