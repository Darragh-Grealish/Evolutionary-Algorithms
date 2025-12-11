import json, logging, os, re
from math import ceil
from typing import List, Dict

logger = logging.getLogger(__name__)

Symbol = str
Production = List[Symbol]

class Grammar:
    """
    Grammar class that parses BNF grammar files and provides dictionary-like access.
    Maintains compatibility with the existing Dict[Symbol, List[Production]] interface.
    """
    
    def __init__(self, bnf_filepath: str = None):
        """
        Initialize Grammar from a BNF file.
        
        Args:
            bnf_filepath: Path to the .bnf grammar file. If None, creates empty grammar.
        """
        self._rules: Dict[Symbol, List[Production]] = {}
        
        if bnf_filepath and os.path.exists(bnf_filepath):
            self._parse_bnf(bnf_filepath)
        elif bnf_filepath:
            logger.warning(f"Grammar file not found: {bnf_filepath}")
    
    def _parse_bnf(self, filepath: str):
        """Parse a BNF grammar file and populate the rules dictionary."""
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Pattern to match BNF rules: <non-terminal> ::= production1 | production2 | ...
        # Handle multi-line rules
        lines = content.split('\n')
        current_nt = None
        current_productions = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this is a new rule definition
            if '::=' in line:
                # Save previous rule if exists
                if current_nt and current_productions:
                    self._rules[current_nt] = current_productions
                
                # Parse new rule
                parts = line.split('::=')
                current_nt = parts[0].strip().strip('<>')
                
                # Get productions on the same line
                rhs = parts[1].strip()
                current_productions = []
                
                if rhs:
                    # Split by | for alternatives
                    alternatives = [alt.strip() for alt in rhs.split('|')]
                    for alt in alternatives:
                        production = self._parse_production(alt)
                        if production is not None:
                            current_productions.append(production)
            
            elif line.startswith('|') and current_nt:
                # Continuation of alternatives
                alt = line[1:].strip()
                production = self._parse_production(alt)
                if production is not None:
                    current_productions.append(production)
        
        # Don't forget the last rule
        if current_nt and current_productions:
            self._rules[current_nt] = current_productions
    
    def _parse_production(self, production_str: str) -> List[Symbol]:
        """
        Parse a single production string into a list of symbols.
        Handles quoted terminals and non-terminals in angle brackets.
        """
        if not production_str:
            return None
        
        symbols = []
        i = 0
        
        while i < len(production_str):
            # Skip whitespace
            while i < len(production_str) and production_str[i].isspace():
                i += 1
            
            if i >= len(production_str):
                break
            
            # Non-terminal: <name>
            if production_str[i] == '<':
                end = production_str.find('>', i)
                if end != -1:
                    nt = production_str[i+1:end]
                    symbols.append(nt)
                    i = end + 1
                else:
                    i += 1
            
            # Quoted terminal: "terminal"
            elif production_str[i] == '"':
                end = production_str.find('"', i + 1)
                if end != -1:
                    terminal = production_str[i+1:end]
                    symbols.append(terminal)
                    i = end + 1
                else:
                    i += 1
            
            else:
                # Unquoted token (shouldn't happen in proper BNF, but handle it)
                j = i
                while j < len(production_str) and not production_str[j].isspace() and production_str[j] not in '<"|':
                    j += 1
                if j > i:
                    symbols.append(production_str[i:j])
                    i = j
                else:
                    i += 1
        
        return symbols if symbols else None
    
    # Dictionary-like interface for compatibility
    def __getitem__(self, key: Symbol) -> List[Production]:
        """Enable dict-like access: grammar[key]"""
        return self._rules[key]
    
    def __setitem__(self, key: Symbol, value: List[Production]):
        """Enable dict-like assignment: grammar[key] = value"""
        self._rules[key] = value
    
    def __contains__(self, key: Symbol) -> bool:
        """Enable 'in' operator: key in grammar"""
        return key in self._rules
    
    def keys(self):
        """Return dictionary keys for compatibility"""
        return self._rules.keys()
    
    def values(self):
        """Return dictionary values"""
        return self._rules.values()
    
    def items(self):
        """Return dictionary items"""
        return self._rules.items()
    
    def get(self, key: Symbol, default=None):
        """Dictionary get method"""
        return self._rules.get(key, default)
    
    def __repr__(self):
        return f"Grammar({len(self._rules)} rules)"
    
    def __str__(self):
        lines = []
        for nt, prods in self._rules.items():
            prods_str = " | ".join([" ".join(p) for p in prods])
            lines.append(f"<{nt}> ::= {prods_str}")
        return "\n".join(lines)

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
        logger.info("EvolutionConfig options: elitism_percentage=%.2f, parent_selection_size=%.2f, mutations_per_genome=%d\n",
                    self.elitism_percentage, self.parent_selection_size, self.mutations_per_genome)

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