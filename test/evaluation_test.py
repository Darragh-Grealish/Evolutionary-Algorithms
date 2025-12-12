import math
import pytest

from src.evaluation import eval_tree, TreeNode, safe_div


def node(sym, *children):
    return TreeNode(symbol=sym, children=list(children))


# terminals

def test_eval_tree_literal():
    t = node("3.5")
    assert eval_tree(t, {}) == pytest.approx(3.5)


def test_eval_tree_variable():
    t = node("x")
    sample = {"x": 10.0}
    assert eval_tree(t, sample) == pytest.approx(10.0)


def test_eval_tree_parse_error_logs(caplog):
    t = node("not_a_number")
    caplog.set_level("ERROR")
    v = eval_tree(t, {})
    assert v == pytest.approx(0.0)
    assert any("Failed to parse tree" in r.message for r in caplog.records)


def test_eval_tree_plain_terminal_values():
    assert eval_tree(7.0, {}) == pytest.approx(7.0)
    assert eval_tree("8.5", {}) == pytest.approx(8.5)


# pre-ops

def test_eval_tree_sin():
    t = node("sin", node("0"))
    assert eval_tree(t, {}) == pytest.approx(math.sin(0.0))


def test_eval_tree_log_nonpositive():
    t = node("log", node("0"))
    v = eval_tree(t, {})
    assert math.isfinite(v)


# arithmetic

def test_eval_tree_add():
    t = node("+", node("2"), node("3"))
    assert eval_tree(t, {}) == pytest.approx(5.0)


def test_eval_tree_sub():
    t = node("-", node("5"), node("2"))
    assert eval_tree(t, {}) == pytest.approx(3.0)


def test_eval_tree_mul():
    t = node("*", node("4"), node("2.5"))
    assert eval_tree(t, {}) == pytest.approx(10.0)


def test_eval_tree_div():
    t = node("/", node("10"), node("2"))
    assert eval_tree(t, {}) == pytest.approx(5.0)


def test_eval_tree_div_by_zero():
    t = node("/", node("10"), node("0"))
    expect = safe_div(10.0, 0.0)
    assert eval_tree(t, {}) == pytest.approx(expect)


# structural

def test_eval_tree_left_paren():
    inner = node("+", node("2"), node("3"))
    t = node("(", inner)
    assert eval_tree(t, {}) == pytest.approx(5.0)


def test_eval_tree_right_paren():
    inner = node("*", node("2"), node("4"))
    t = node(")", inner)
    assert eval_tree(t, {}) == pytest.approx(8.0)


def test_eval_tree_empty_paren():
    assert eval_tree(node("("), {}) == pytest.approx(0.0)
    assert eval_tree(node(")"), {}) == pytest.approx(0.0)


def test_eval_tree_seq_last_child():
    a = node("1")
    b = node("2")
    t = node("seq", a, b)
    assert eval_tree(t, {}) == pytest.approx(2.0)


def test_eval_tree_seq_empty():
    t = node("seq")
    assert eval_tree(t, {}) == pytest.approx(0.0)


def test_eval_tree_start_wraps_expr():
    expr = node("*", node("2"), node("3"))
    t = node("start", expr)
    assert eval_tree(t, {}) == pytest.approx(6.0)


def test_eval_tree_start_empty():
    t = node("start")
    assert eval_tree(t, {}) == pytest.approx(0.0)
