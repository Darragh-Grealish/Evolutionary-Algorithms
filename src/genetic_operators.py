import random
from src.grammar import choose_production, GRAMMAR

def crossover_genotypes(parent1, parent2, rng=random):
    """
    DSGE-style gene-level uniform crossover.
    - Build a random binary mask over the gene keys (non-terminals).
    - For each key, offspring take the entire gene (list of productions)
      from one parent or the other, according to the mask.
    """
    keys = sorted(set(parent1.keys()) | set(parent2.keys()))

    child1 = {}
    child2 = {}

    if parent1 == None or parent2 == None:
        print("Error : One of the parents is None")

    for k in keys:
        take_from_p2 = rng.random() < 0.5  # mask bit for this gene
        g1 = parent1.get(k, [])
        g2 = parent2.get(k, [])

        if take_from_p2:
            # child1 takes P2's whole gene, child2 takes P1's
            child1[k] = list(g2)
            child2[k] = list(g1)
        else:
            # child1 takes P1's whole gene, child2 takes P2's
            child1[k] = list(g1)
            child2[k] = list(g2)

    return child1, child2

def crossover_individuals(ind1, ind2, rng=random):
    """
    Convenience wrapper to crossover two individuals with structured genotypes.
    Returns two new individuals with phenotypes cleared for remapping.
    """
    c1g, c2g = crossover_genotypes(ind1.get('genotype', {}), ind2.get('genotype', {}), rng=rng)
    child1 = {"genotype": c1g, "phenotype": None, "fitness": None}
    child2 = {"genotype": c2g, "phenotype": None, "fitness": None}
    return child1, child2
    
def mutate_genotype(genotype, max_depth, rng=random):
    """
    One-step mutation: choose a random gene (non-terminal) and a random
    position, then assign a random valid production index.
    Assumes all gene lists are non-empty.
    """
    new_genotype = {nt: list(genes) for nt, genes in genotype.items()}

    # Choose a non-empty gene; if selected gene is empty, fallback to a non-empty one or seed
    candidates = [k for k, v in new_genotype.items() if len(v) > 0]

    nt = rng.choice(candidates)
    genes = new_genotype[nt]

    pos = rng.randrange(len(genes))
    
    max_choices = len(GRAMMAR[nt])

    old_idx = genes[pos]
    new_idx = choose_production(GRAMMAR, nt, 0, max_depth)
    if new_idx == old_idx and max_choices > 1:
        new_idx = (old_idx + 1) % max_choices

    genes[pos] = new_idx
    return new_genotype
