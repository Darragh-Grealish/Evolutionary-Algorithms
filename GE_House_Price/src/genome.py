class Genome:
    def __init__(self, genotype):
        self.genotype = genotype   # list[int]
        self.phenotype = None      # expression string
        self.fitness = None        # float

    def __repr__(self):
        return f"Genome(f={self.fitness}, expr={self.phenotype})"
