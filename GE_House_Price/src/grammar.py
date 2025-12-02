import random, re

pattern = re.compile(r'(<[^<>]+>)')

def tokenize(rule):
    parts = pattern.split(rule)
    return [p for p in parts if p.strip()]

grammar = {
    '<expr>': [
        '<expr> <op> <expr>',
        '(<expr> <op> <expr>)',
        '<var>',
        '<st>'
    ],
    '<op>': ['+', '-', '*', '/'],
    '<var>': [
        'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 
        'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement', 
        'yr_built', 'yr_renovated', 'city_num', 'statezip_num', 'country_num'
    ],
    '<st>': [f"{i}" for i in range(0, 50)]
}


def genome_to_expression(genome, max_depth, start='<expr>'):
    codon_index = 0
    current_depth = 0

    def expand(symbol, depth):
        nonlocal codon_index
        if depth > max_depth:
            # Force a terminal rule if possible: [TODO: Need to change so it's based on Max Depth as apposed to forcing a Terminal]
            if symbol == '<expr>':
                # Pick a random terminal ('<var>' or '<st>') - we ensure it does not recurse further
                term_rule = random.choice(['<var>', '<st>'])
                return expand(term_rule, depth + 1)
            if symbol in ['<var>', '<st>']:
                rules = grammar[symbol]
                rule_i = genome[codon_index % len(genome)] % len(rules)
                codon_index += 1
                return rules[rule_i]
            return ''
        
        rules = grammar[symbol]
        rule_i = genome[codon_index % len(genome)] % len(rules)
        codon_index += 1
        selected_rule = rules[rule_i]

        output = ''
        for token in tokenize(selected_rule):
            if token in grammar:
                output += expand(token, depth + 1)
            else:
                output += token
            output += ' '
        return output.strip()
    return expand(start, 0)
