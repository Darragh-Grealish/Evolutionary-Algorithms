import json
from math import ceil

class EvolutionConfig:
    def __init__(self, filepath="./config.json"):
        with open(filepath, "r") as f:
            data = json.load(f)

        # Basic params
        self.generations = data.get("generations", 10)
        self.population_size = data.get("population_size", 100)
        self.genome_length = data.get("genome_length", 50)
        self.max_depth = data.get("max_depth", 10)

        # Evolution options with defaults
        opts = data.get("evol_options", {})
        self.elitism_percentage = opts.get("elitism_percentage", 0.2)
        self.parent_selection_size = opts.get("parent_selection_size", 0.1)
        self.crossover_rate = opts.get("crossover_rate", 0.7)
        self.mutation_rate = opts.get("mutation_rate", 0.1)
        self.min_mutation_rate = opts.get("min_mutation_rate", 0.01)
        self.max_mutation_rate = opts.get("max_mutation_rate", 0.5)

    @property
    def elitism_count(self):
        return max(1, ceil(self.population_size * self.elitism_percentage))

    @property
    def top_parents_count(self):
        return max(1, ceil(self.population_size * self.parent_selection_size))

# Example usage
if __name__ == "__main__":
    cfg = EvolutionConfig("config.json")
    print("Elitism count:", cfg.elitism_count)
    print("Top parents count:", cfg.top_parents_count)
    print("Crossover rate:", cfg.crossover_rate)
    print("Mutation rate:", cfg.mutation_rate)
