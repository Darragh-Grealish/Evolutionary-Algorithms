import json, logging
from math import ceil

logger = logging.getLogger(__name__)

class EvolutionConfig:
    def __init__(self, filepath="./config.json"):
        with open(filepath, "r") as f:
            data = json.load(f)

        # Basic params
        self.generations = data.get("generations", 10)
        self.population_size = data.get("population_size", 100)
        self.genome_length = data.get("genome_length", 50)
        self.max_depth = data.get("max_depth", 10)
        self.feature_names = data.get("feature_names", [])
        self.grammar = data.get("grammar", {})

        # Evolution options with defaults
        opts = data.get("evol_options", {})
        self.elitism_percentage = opts.get("elitism_percentage", 0.2)
        self.parent_selection_size = opts.get("parent_selection_size", 0.1)
        self.mutations_per_genome = opts.get("muatations_per_genome", 1)

        logger.info("EvolutionConfig initialized with: generations=%d, population_size=%d, genome_length=%d, max_depth=%d",
                    self.generations, self.population_size, self.genome_length, self.max_depth)

    @property
    def elitism_count(self):
        return max(1, ceil(self.population_size * self.elitism_percentage))

    @property
    def top_parents_count(self):
        return max(1, ceil(self.population_size * self.parent_selection_size))

class TreeNode:
    def __init__(self, symbol, children=None):
        self.symbol = symbol
        self.children = children or []

    def __repr__(self):
        return self.to_infix()

    def __str__(self):
        return self.to_infix()

    def to_infix(self):
        """
        Render the tree as a mathematical infix expression.
        Handles unary pre-ops (sin, cos, exp, log, inv), binary ops (+,-,*,/),
        and structural wrappers (start, seq, parentheses) gracefully.
        """
        sym = self.symbol

        # Terminals
        if not self.children:
            return str(sym)

        # Structural wrappers
        if sym == 'start':
            return self.children[0].to_infix() if self.children else ''
        if sym == 'seq':
            return ', '.join(child.to_infix() for child in self.children)
        if sym in ('(', ')'):
            return self.children[0].to_infix() if self.children else ''

        # Unary pre-ops
        if sym in {'sin', 'cos', 'exp', 'log', 'inv'}:
            arg = self.children[0].to_infix() if self.children else ''
            return f"{sym}({arg})"

        # Binary arithmetic ops
        if sym in {'+', '-', '*', '/'} and len(self.children) >= 2:
            left = self.children[0].to_infix()
            right = self.children[1].to_infix()
            return f"({left} {sym} {right})"

        # Fallback: prefix-style for anything unexpected
        return f"{sym}({', '.join(child.to_infix() for child in self.children)})"

# Example usage
if __name__ == "__main__":
    cfg = EvolutionConfig("config.json")
    logger.info("Elitism count: %d", cfg.elitism_count)
    logger.info("Top parents count: %d", cfg.top_parents_count)
    logger.info("Crossover rate: %.2f", cfg.crossover_rate)
    logger.info("Mutation rate: %.2f", cfg.mutation_rate)