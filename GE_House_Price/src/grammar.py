import random, re, os

pattern = re.compile(r'(<[^<>]+>)')

def tokenize(rule):
    parts = pattern.split(rule)
    return [p for p in parts if p.strip()]

def parse_bnf_file(path="grammar/houseprice.bnf"):
    grammar = {}
    if not os.path.isabs(path):
        base = os.path.dirname(os.path.dirname(__file__))
        path = os.path.join(base, path)
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): 
                continue
            if "::=" not in line: 
                continue
            lhs, rhs = line.split("::=", 1)
            lhs = lhs.strip()
            productions = [p.strip() for p in rhs.split("|")]
            grammar[lhs] = productions
    return grammar


grammar = parse_bnf_file()


def genome_to_expression(genome, max_depth, start='<expr>'):
    codon_index = 0
    def expand(symbol, depth):
        nonlocal codon_index
        if depth > max_depth:
            if symbol == '<expr>':
                return expand(random.choice(['<var>', '<st>']), depth+1)
            if symbol in grammar:
                rules = grammar[symbol]
                i = genome[codon_index % len(genome)] % len(rules)
                codon_index += 1
                return rules[i]
            return ''
        rules = grammar[symbol]
        i = genome[codon_index % len(genome)] % len(rules)
        codon_index += 1
        selected = rules[i]
        out = []
        for tok in tokenize(selected):
            out.append(expand(tok, depth+1) if tok in grammar else tok)
        return ' '.join(out).strip()
    return expand(start, 0)
