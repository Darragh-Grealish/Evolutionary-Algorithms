import src.population as population
from src.population import map_genotype, TreeNode, initialise_population, GRAMMAR


def assert_tree_equal(a: TreeNode, b: TreeNode):
    assert a.symbol == b.symbol
    assert len(a.children) == len(b.children)
    for ca, cb in zip(a.children, b.children):
        assert_tree_equal(ca, cb)


def test_map_genotype_parenthesised_binary():
    grammar = {
        "start": [["expr"]],
        "expr": [["(", "expr", "op", "expr", ")"], ["var"]],
        "op": [["+"], ["-"]],
        "var": [["x"], ["y"]],
    }
    genotype = {
        "start": [0],
        "expr": [0, 1, 1],
        "op": [0],
        "var": [0, 1],
    }

    tree = map_genotype(grammar, genotype, start_nt="start", max_depth=5)
    expected = TreeNode("+", [TreeNode("x"), TreeNode("y")])
    assert_tree_equal(tree, expected)


def test_map_genotype_unary_pre_op_with_parentheses(monkeypatch):
    grammar = {
        "start": [["expr"]],
        "expr": [["pre_op", "(", "expr", ")"], ["var"]],
        "pre_op": [["sin"], ["cos"]],
        "var": [["x"]],
    }
    genotype = {
        "start": [0],
        "expr": [0, 1],
        "pre_op": [0],
        "var": [0],
    }

    tree = map_genotype(grammar, genotype, start_nt="start", max_depth=5)
    expected = TreeNode("sin", [TreeNode("x")])
    assert_tree_equal(tree, expected)


def test_map_genotype_direct_binary_expr_op_expr():
    grammar = {
        "start": [["expr"]],
        "expr": [["expr", "op", "expr"], ["var"]],
        "op": [["+"] , ["-"], ["*"], ["/"]],
        "var": [["x"], ["y"]],
    }
    genotype = {
        "start": [0],
        "expr": [0, 1, 1],
        "op": [2],
        "var": [0, 1],
    }

    tree = map_genotype(grammar, genotype, start_nt="start", max_depth=5)
    expected = TreeNode("*", [TreeNode("x"), TreeNode("y")])
    assert_tree_equal(tree, expected)


def test_map_genotype_var_single_node():
    grammar = {
        "start": [["expr"]],
        "expr": [["var"]],
        "var": [["x"]],
    }
    genotype = {"start": [0], "expr": [0], "var": [0]}

    tree = map_genotype(grammar, genotype, start_nt="start", max_depth=5)
    expected = TreeNode("x")
    assert_tree_equal(tree, expected)


def test_map_genotype_extends_genotype_when_gene_list_exhausted(monkeypatch):
    grammar = {
        "start": [["expr"]],
        "expr": [["var"], ["var"]],
        "var": [["x"], ["y"]],
    }
    genotype = {
        "start": [0],
        "expr": [],
        "var": [0],
    }

    def fake_choose_production(grammar_, nt, depth, max_depth):
        assert nt == "expr"
        return 1

    monkeypatch.setattr(population, "choose_production", fake_choose_production)

    tree = map_genotype(grammar, genotype, start_nt="start", max_depth=5)
    expected = TreeNode("x")
    assert_tree_equal(tree, expected)

    assert len(genotype["expr"]) == 1
    assert genotype["expr"][0] == 1


def test_map_genotype_uses_cache_for_same_genotype(monkeypatch):
    grammar = {
        "start": [["expr"]],
        "expr": [["var"]],
        "var": [["x"]],
    }
    genotype = {"start": [0], "expr": [0], "var": [0]}

    population.genome_to_expression_cache.clear()

    t1 = map_genotype(grammar, genotype, "start", max_depth=5)
    t2 = map_genotype(grammar, genotype, "start", max_depth=5)

    assert_tree_equal(t1, t2)

    key = tuple((nt, tuple(genotype.get(nt, []))) for nt in sorted(grammar.keys()))
    assert key in population.genome_to_expression_cache




class DummyCfg:
    def __init__(self, population_size, max_depth):
        self.population_size = population_size
        self.max_depth = max_depth


def test_initialise_population_shapes():
    cfg = DummyCfg(population_size=5, max_depth=4)
    pop = initialise_population(cfg)
    assert len(pop) == cfg.population_size
    for ind in pop:
        assert "genotype" in ind
        assert "phenotype" in ind and ind["phenotype"] is None
        assert "fitness" in ind and ind["fitness"] is None
        assert isinstance(ind["genotype"], dict)


def test_initialise_population_size_and_keys():
    cfg = DummyCfg(population_size=10, max_depth=4)

    pop = initialise_population(cfg)

    assert len(pop) == cfg.population_size
    for ind in pop:
        assert isinstance(ind, dict)
        assert set(ind.keys()) == {"genotype", "phenotype", "fitness"}
        assert ind["phenotype"] is None
        assert ind["fitness"] is None


def test_initialise_population_genotype_structure():
    cfg = DummyCfg(population_size=3, max_depth=3)

    pop = initialise_population(cfg)

    for ind in pop:
        g = ind["genotype"]
        assert isinstance(g, dict)
        # all gene lists should be lists of intgers
        for _, genes in g.items():
            assert isinstance(genes, list)
            for idx in genes:
                assert isinstance(idx, int)


def test_initialise_population_basic_randomness():
    cfg = DummyCfg(population_size=7, max_depth=5)

    pop1 = initialise_population(cfg)
    pop2 = initialise_population(cfg)

    genos1 = [ind["genotype"] for ind in pop1]
    genos2 = [ind["genotype"] for ind in pop2]

    assert genos1 != genos2