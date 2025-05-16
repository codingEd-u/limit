import hypothesis.strategies as st
from hypothesis import given

from limit.limit_ast import ASTNode


def test_astnode_repr():
    node = ASTNode("assign", "x", [])
    assert repr(node) == "ASTNode(assign, value='x')"


def test_astnode_eq_equal():
    n1 = ASTNode("assign", "x", [])
    n2 = ASTNode("assign", "x", [])
    assert n1 == n2


def test_astnode_eq_not_equal_kind():
    n1 = ASTNode("assign", "x", [])
    n2 = ASTNode("arith", "x", [])
    assert n1 != n2


def test_astnode_eq_not_equal_children():
    n1 = ASTNode("assign", "x", [ASTNode("identifier", "x")])
    n2 = ASTNode("assign", "x", [ASTNode("identifier", "y")])
    assert n1 != n2


def test_astnode_to_dict_basic():
    node = ASTNode("assign", "x", [ASTNode("number", "1")], line=1, col=2)
    d = node.to_dict()
    assert d["kind"] == "assign"
    assert d["value"] == "x"
    assert d["line"] == 1
    assert d["col"] == 2
    assert isinstance(d["children"], list)
    assert d["children"][0]["kind"] == "number"


def test_astnode_to_dict_with_type():
    node = ASTNode("func", "f", [], type_=["x"], return_type="int")
    d = node.to_dict()
    assert d["type"] == ["x"]
    assert d["return_type"] == "int"


def test_astnode_eq_with_type_and_return_type():
    n1 = ASTNode("func", "f", [], type_=["x"], return_type="int")
    n2 = ASTNode("func", "f", [], type_=["x"], return_type="int")
    assert n1 == n2


def test_astnode_eq_different_return_type():
    n1 = ASTNode("func", "f", [], type_=["x"], return_type="int")
    n2 = ASTNode("func", "f", [], type_=["x"], return_type="float")
    assert n1 != n2


def test_astnode_eq_non_astnode():
    n = ASTNode("assign", "x")
    assert n != "not an ast"


@given(st.text(min_size=1), st.text(min_size=1))
def test_astnode_eq_same_kind_value(kind, value):
    assert ASTNode(kind, value) == ASTNode(kind, value)


@given(st.text(min_size=1), st.text(min_size=1))
def test_astnode_eq_different_kind_value(kind, value):
    assert ASTNode(kind, value) != ASTNode(kind + "x", value + "x")


@given(
    kind=st.text(min_size=1),
    value=st.text(min_size=1),
    type_=st.lists(st.text(min_size=1), max_size=3),
    return_type=st.text(min_size=1),
)
def test_astnode_repr_and_eq_extended(kind, value, type_, return_type):
    children = [ASTNode("literal", str(i)) for i in range(4)]
    else_children = [ASTNode("error", "e1"), ASTNode("error", "e2")]
    node = ASTNode(kind, value, children=children, type_=type_, return_type=return_type)
    node.else_children = else_children
    s = repr(node)
    assert kind in s
    assert f"value={repr(value)}" in s
    assert "children=[" in s
    assert "else_children=[" in s
    assert (
        ASTNode(kind, value, children[:], type_=type_, return_type=return_type) != node
    )  # else_children mismatch


@given(st.text(), st.text(), st.integers(), st.integers())
def test_astnode_to_dict_and_back(kind, value, line, col):
    node = ASTNode(kind, value, line=line, col=col)
    d = node.to_dict()
    assert d["kind"] == kind
    assert d["value"] == value
    assert d["line"] == line
    assert d["col"] == col
    assert isinstance(d["children"], list)
    assert isinstance(d["else_children"], list)
