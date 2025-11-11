import random, math, csv, copy
from typing import Callable, List, Optional, Any, Tuple
from Tree import TreeNode
import matplotlib.pyplot as plt

class Function:
    """Lightweight wrapper for operator functions.

    Attributes:
        name: printable name (e.g. '+')
        func: callable that accepts `arity` positional args and returns a value
        arity: number of operands required
    """
    def __init__(self, name: str, func: Callable, arity: int = 2) -> None:
        self.name = name
        self.func = func
        self.arity = arity

    def __repr__(self) -> str:
        return f"Function({self.name!r}, arity={self.arity})"

class GeneticProgram:
    def __init__(
        self,
        population_size: int,
        terminal_set: List[Any],
        function_set: List[Function],
        mutation_rate: float = 0.1,
        max_tree_depth: int = 10,
        training_data: Optional[List[Tuple[dict, float]]] = None, # list of training data, which maps a dict of input variables and their values to the expected output
    ) -> None:
        self.population_size = population_size
        self.terminal_set = terminal_set
        self.function_set = function_set
        self.mutation_rate = mutation_rate
        self.max_tree_depth = max_tree_depth # max depth per individual tree
        self.cur_tree_depth = max_tree_depth // 2
        # If provided, fitness is 1/(1 + MSE) across the dataset (higher fitness is better)
        self.training_data = training_data
        self.population: List[TreeNode] = self.initialize_population()
        self.generation_total_fitness = []

    def initialize_population(self):
        return [self.ramped_half_and_half(i) for i in range(self.population_size)]

    def ramped_half_and_half(self, individual_index: int) -> TreeNode:
        """
        Create 50 % full and 50 % grow trees
        """
        
        # every 20% increase the max depth by 1
        if individual_index % (self.population_size // 5) == 0:
            self.cur_tree_depth = min(self.max_tree_depth, self.cur_tree_depth + 1)

        # 50/50 chance full or grow
        if random.random() < 0.5:
            tree = self.grow_tree(self.cur_tree_depth)
            return tree
        else:
            tree = self.full_tree(self.cur_tree_depth)
            return tree

    # Create a full tree
    def full_tree(self, max_depth: int) -> TreeNode:
        """
        Create a full (binary) tree of exactly `depth` levels (depth is remaining depth).
        Leaf nodes are chosen from terminal_set when depth == 0.
        Internal nodes are chosen from function_set and are given two children.
        """
        if max_depth == 0:
            terminal = random.choice(self.terminal_set)
            return TreeNode(terminal)
        else:
            # Prefer functions matching binary arity when possible but support n-ary
            preferred = [f for f in self.function_set if f.arity == 2]
            function = random.choice(preferred) if preferred else random.choice(self.function_set)
            children = [self.full_tree(max_depth - 1) for _ in range(function.arity)]
            return TreeNode(function, children)
        
    # Create a grow tree
    def grow_tree(self, max_depth: int) -> TreeNode:
        if max_depth == 0 or (max_depth < self.max_tree_depth and random.random() < 0.5):
            terminal = random.choice(self.terminal_set)
            return TreeNode(terminal)
        else:
            function = random.choice(self.function_set)
            children = [self.grow_tree(max_depth - 1) for _ in range(function.arity)]
            return TreeNode(function, children)

    def fitness(self, individual):
        # If training data is provided, evaluate the individual on all training samples
        if self.training_data:
            tree_errors = []
            for input_vals, target in self.training_data: # get error of each training data piece
                try:
                    pred = self.evaluate(individual, input_vals) # evaluate a set of input values on a tree
                    if not math.isfinite(pred):
                        return 0.0
                    tree_errors.append((pred - target) ** 2) # add squared error to list
                except Exception:
                    return 0.0
            mse = sum(tree_errors) / len(tree_errors) if tree_errors else float('inf')
            print(f"Average MSE of {mse} from {len(tree_errors)} samples")
            return 1.0 / (1.0 + mse) # fitness is inverse of MSE, as we want a higher MSE to be worse

        # Otherwise treat evaluate(individual) as a raw score and map to (0,1) via logistic
        try:
            val = self.evaluate(individual, {}) # empty dict means no input training data, means there should be no variables in the tree
            if not math.isfinite(val):
                return 0.0
            return 1.0 / (1.0 + math.exp(-float(val)))
        except Exception:
            return 0.0

    # roulette selection of parents
    def select_parents(self):
        # Select parents based on fitness
        weighted_population = [
            (individual, self.fitness(individual)) for individual in self.population
        ]
        # get total fitness accross all individuals
        self.generation_total_fitness.append(sum(fitness for _, fitness in weighted_population))
        if self.generation_total_fitness[-1] == 0:
            return random.sample(self.population, 2)
        selection_probs = [fitness / self.generation_total_fitness[-1] for _, fitness in weighted_population]
        parents = random.choices(self.population, weights=selection_probs, k=2)
        return parents
    
    def parse_tree(self, node: TreeNode) -> str:
        """Return a readable s-expression-like string for the subtree.
        Functions are printed as (name arg1 arg2 ...). Terminals are shown as numbers
        or variable names.
        """
        if node.is_leaf():
            return repr(node.value)
        else:
            # node.value should be a Function
            func = node.value
            children_expressions = [self.parse_tree(child) for child in node.children]
            return f"({func.name} {' '.join(children_expressions)})"

    def evaluate(self, node: TreeNode, input_training_data: Optional[dict] = None) -> float:
        """Evaluate the tree numerically"""

        input_training_data = input_training_data or {}
        if node.is_leaf(): # if node is a leaf then we return the value
            v = node.value
            if isinstance(v, (int, float)):
                return v
            elif isinstance(v, str): # indictaes node holds a variable
                return float(input_training_data[v]) # return the variables value
            else:
                raise TypeError(f"Unsupported terminal type: {type(v)}")
        else:
            func: Function = node.value
            child_vals = [self.evaluate(c, input_training_data) for c in node.children] # recursively evaluate subchildren of node
            # protect division by zero for safety
            try:
                return func.func(*child_vals)
            except ZeroDivisionError:
                return float('inf')

    def sub_tree_crossover(self, parent1: TreeNode, parent2: TreeNode) -> tuple[TreeNode, TreeNode]:
        
        # avoid shallow copy as it can effect fitness scores during single generation
        parent1 = copy.deepcopy(parent1)
        parent2 = copy.deepcopy(parent2)
        
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
        print(f"Running generation - cur pop size {self.population_size} range {self.population_size // 2}")
        for _ in range(self.population_size // 2):
            parent1, parent2 = self.select_parents()
            # use subtree crossover implementation
            offspring1, offspring2 = self.sub_tree_crossover(
                copy.deepcopy(parent1),
                copy.deepcopy(parent2)
            )
            new_population.append(self.mutate(offspring1))
            new_population.append(self.mutate(offspring2))
        self.population = new_population
        print(f"Total fitness {self.generation_total_fitness[-1]}")

    def run(self, generations):
        for _ in range(generations): 
            self.run_generation()

# test genetic program
if __name__ == "__main__":
    # Example: terminals include variable names (strings) and numeric constants (ints)
    terminal_set = ['x', 'y', 1, 2, 3]

    # Use Function wrappers with callables (binary operators here)
    def safe_div(a, b):
        try:
            return a / b
        except ZeroDivisionError:
            return float('inf')

    function_set = [
        Function('+', lambda a, b: a + b, arity=2),
        Function('-', lambda a, b: a - b, arity=2),
        Function('*', lambda a, b: a * b, arity=2),
        Function('/', safe_div, arity=2),
    ]

    training_data = []
    with open('./training_data.csv', mode ='r')as file:
        csvFile = csv.reader(file)
        for i, entry in enumerate(csvFile):
                if i == 0:
                    continue
                training_data.append(({'x': entry[0], 'y': entry[1]}, entry[2])) # hard code reading of data from csv and adding it to training data

    gp = GeneticProgram(population_size=100, terminal_set=terminal_set, function_set=function_set, max_tree_depth=10)

    gp.run(generations=100)

    plt.plot(range(len(gp.generation_total_fitness)), gp.generation_total_fitness) # TODO add a smoothing process to total generation fitness
    plt.xlabel("Generation Number")
    plt.ylabel("Total Fitness")
    plt.title("Fitness vs Generations")
    plt.savefig("fitness_vs_generations.png")
