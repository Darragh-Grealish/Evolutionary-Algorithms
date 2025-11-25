import random

class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate, chromosome_length=50):
        self.chromosome_length = chromosome_length
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = self.initialize_population()

    def initialize_population(self):
        return [self.random_individual() for _ in range(self.population_size)]

    def random_individual(self):
        # Create a random bitstring individual
        return [random.choice([0, 1]) for _ in range(self.chromosome_length)]

    def fitness(self, individual):
        return sum(individual)

    def select_parents(self):
        # Select parents based on fitness
        weighted_population = [
            (individual, self.fitness(individual)) for individual in self.population
        ]
        total_fitness = sum(fitness for _, fitness in weighted_population)
        print(f"Total fitness: {total_fitness}")
        if total_fitness == 0:
            return random.sample(self.population, 2)
        selection_probs = [fitness / total_fitness for _, fitness in weighted_population]
        parents = random.choices(self.population, weights=selection_probs, k=2)
        return parents

    def crossover(self, parent1, parent2):
        # Perform crossover between two parents
        crossover_point = random.randint(1, self.chromosome_length - 1)
        offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
        offspring2 = parent2[:crossover_point] + parent1[crossover_point:]
        return offspring1, offspring2

    def mutate(self, individual):
        # Mutate an individual based on mutation rate
        return [ # 
            gene if random.random() > self.mutation_rate else 1 - gene
            for gene in individual
        ]

    def run_generation(self):
        # Run one generation of the genetic algorithm
        new_population = []
        # for hal
        for _ in range(self.population_size // 2):
            parent1, parent2 = self.select_parents()
            offspring1, offspring2 = self.crossover(parent1, parent2)
            new_population.append(self.mutate(offspring1))
            new_population.append(self.mutate(offspring2))
        self.population = new_population

    def run(self, generations):
        for _ in range(generations): 
            self.run_generation()

sample_ga = GeneticAlgorithm(population_size=100, mutation_rate=0.01)
sample_ga.run(generations=10)
best_individual = max(sample_ga.population, key=sample_ga.fitness)
print("Best individual:", best_individual)
