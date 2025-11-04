import random
from Tree import TreeNode
from typing import List

class GeneticProgram:
    def __init__(self, population_size, terminal_set, function_set, mutation_rate=0.1, max_tree_depth=10):
        self.population_size = population_size
        self.terminal_set = terminal_set
        self.function_set = function_set
        self.max_tree_depth = max_tree_depth
        self.mutation_rate = mutation_rate
        self.population: List[TreeNode] = self.initialize_population()

    def initialize_population(self):
        return [self.ramped_half_and_half(i) for i in range(self.population_size)]

    # Create 50 % full and 50 % grow trees
    def ramped_half_and_half(self, individual_index) -> TreeNode:
        # Increase max depth every 10% of population
        if individual_index % (self.population_size // 10) == 0:
            self.max_tree_depth += 1
            print(f"Increasing max tree depth to {self.max_tree_depth}")

        # 50 / 50 chance of full or grow
        if random.random() < 0.5:
            return self.full_tree(self.max_tree_depth)
        else:
            return self.grow_tree(self.max_tree_depth)
        
    # Create a full tree
    def full_tree(self, max_depth) -> TreeNode:
        """
        Create a full (binary) tree of exactly `depth` levels (depth is remaining depth).
        Leaf nodes are chosen from terminal_set when depth == 0.
        Internal nodes are chosen from function_set and are given two children.
        If a chosen function has a different arity, it is still used but two children are created
        to satisfy the "two subnodes" requirement (prefer functions with arity == 2).
        """
        if max_depth == 0:
            terminal = random.choice(self.terminal_set)
            return TreeNode(terminal)
        else:
            # Prefer functions with arity == 2
            binary_funcs = [f for f in self.function_set if getattr(f, "arity", None) == 2]
            function = random.choice(binary_funcs) if binary_funcs else random.choice(self.function_set)
            # Always create two children for a full binary shape
            children = [self.full_tree(max_depth - 1) for _ in range(2)]
            return TreeNode(function, children)
        
    # Create a grow tree
    def grow_tree(self, max_depth) -> TreeNode:
        if max_depth == 0 or (max_depth < self.max_tree_depth and random.random() < 0.5):
            terminal = random.choice(self.terminal_set)
            return TreeNode(terminal)
        else:
            function = random.choice(self.function_set)
            children = [self.grow_tree(max_depth - 1) for _ in range(function.arity)]
            return TreeNode(function, children)

    def fitness(self, individual):
        # Add mechanism to parse tree and evaluate fitness

        return None  # Placeholder for fitness function

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

    def sub_tree_crossover(self, parent1: TreeNode, parent2: TreeNode) -> tuple[TreeNode, TreeNode]:
        # Pick random crossover points in both parents: randomly pick a child node
        crossover_point1 = random.choice(list(parent1.children)) if parent1.children else parent1
        crossover_point2 = random.choice(list(parent2.children)) if parent2.children else parent2

        # Create offspring by swapping subtrees
        offspring1 = TreeNode(parent1.value, parent1.children.copy())
        offspring2 = TreeNode(parent2.value, parent2.children.copy())

        # Find corresponding nodes in offspring
        def find_and_replace(node, target, replacement):
            if node == target:
                return replacement
            new_children = [find_and_replace(child, target, replacement) for child in node.children]
            return TreeNode(node.value, new_children)   
        
        offspring1 = find_and_replace(offspring1, crossover_point1, crossover_point2)
        offspring2 = find_and_replace(offspring2, crossover_point2, crossover_point1)

        return offspring1, offspring2

    def mutate(self, individual):
        # Mutate an individual based on mutation rate
        if random.random() < self.mutation_rate:
            # Replace a random subtree with a new randomly generated subtree
            mutation_point = random.choice(list(individual.children)) if individual.children else individual
            new_subtree = self.ramped_half_and_half(0)  # Generate a new random subtree
            # Find and replace the mutation point
            def find_and_replace(node, target, replacement):
                if node == target:
                    return replacement
                new_children = [find_and_replace(child, target, replacement) for child in node.children]
                return TreeNode(node.value, new_children)   
            return find_and_replace(individual, mutation_point, new_subtree)
        else:
            return individual
        
    def run_generation(self):
        # Run one generation of the genetic algorithm
        new_population = []
        for _ in range(self.population_size // 2):
            parent1, parent2 = self.select_parents()
            offspring1, offspring2 = self.crossover(parent1, parent2)
            new_population.append(self.mutate(offspring1))
            new_population.append(self.mutate(offspring2))
        self.population = new_population

    def run(self, generations):
        for _ in range(generations): 
            self.run_generation()
