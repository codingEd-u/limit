from typing import Any, cast

import hypothesis.strategies as st
from hypothesis import given

from limit.limit_ast import ASTNode


def test_astnode_repr() -> None:
    node = ASTNode("assign", "x", [])
    assert repr(node) == "ASTNode(assign, value='x')"


def test_astnode_eq_equal() -> None:
    n1 = ASTNode("assign", "x", [])
    n2 = ASTNode("assign", "x", [])
    assert n1 == n2


def test_astnode_eq_not_equal_kind() -> None:
    n1 = ASTNode("assign", "x", [])
    n2 = ASTNode("arith", "x", [])
    assert n1 != n2


def test_astnode_eq_not_equal_children() -> None:
    n1 = ASTNode("assign", "x", [ASTNode("identifier", "x")])
    n2 = ASTNode("assign", "x", [ASTNode("identifier", "y")])
    assert n1 != n2


def test_astnode_to_dict_basic() -> None:
    node = ASTNode("assign", "x", [ASTNode("number", "1")], line=1, col=2)
    d = node.to_dict()
    assert d["kind"] == "assign"
    assert d["value"] == "x"
    assert d["line"] == 1
    assert d["col"] == 2
    assert isinstance(d["children"], list)
    assert d["children"][0]["kind"] == "number"


def test_astnode_to_dict_with_type() -> None:
    node = ASTNode("func", "f", [], type_="x", return_type="int")
    d = node.to_dict()
    assert d["type"] == "x"
    assert d["return_type"] == "int"


def test_astnode_eq_with_type_and_return_type() -> None:
    n1 = ASTNode("func", "f", [], type_="x", return_type="int")
    n2 = ASTNode("func", "f", [], type_="x", return_type="int")
    assert n1 == n2


def test_astnode_eq_different_return_type() -> None:
    n1 = ASTNode("func", "f", [], type_="x", return_type="int")
    n2 = ASTNode("func", "f", [], type_="x", return_type="float")
    assert n1 != n2


def test_astnode_eq_non_astnode() -> None:
    n = ASTNode("assign", "x")
    assert n != "not an ast"


@given(st.text(min_size=1), st.text(min_size=1))  # type: ignore[misc]
def test_astnode_eq_same_kind_value(kind: str, value: str) -> None:
    assert ASTNode(kind, value) == ASTNode(kind, value)


@given(st.text(min_size=1), st.text(min_size=1))  # type: ignore[misc]
def test_astnode_eq_different_kind_value(kind: str, value: str) -> None:
    assert ASTNode(kind, value) != ASTNode(kind + "x", value + "x")


@given(
    st.text(min_size=1),
    st.text(min_size=1),
    st.text(min_size=1),
    st.text(min_size=1),
)  # type: ignore[misc]
def test_astnode_repr_and_eq_extended(
    kind: str, value: str, type_: str, return_type: str
) -> None:
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
    )


@given(st.text(), st.text(), st.integers(), st.integers())  # type: ignore[misc]
def test_astnode_to_dict_and_back(kind: str, value: str, line: int, col: int) -> None:
    node = ASTNode(kind, value, line=line, col=col)
    d = node.to_dict()
    assert d["kind"] == kind
    assert d["value"] == value
    assert d["line"] == line
    assert d["col"] == col
    assert isinstance(d["children"], list)
    assert isinstance(d["else_children"], list)


def test_astnode_repr_includes_value() -> None:
    node = ASTNode(kind="number", value="42")
    r = repr(node)
    assert "value='42'" in r


def test_astnode_repr_truncates_children() -> None:
    children = [ASTNode(kind=f"child{i}") for i in range(5)]
    node = ASTNode(kind="parent", children=children)
    r = repr(node)
    assert "children=[" in r
    assert "..." in r


def test_astnode_repr_truncates_else_children() -> None:
    else_children = [ASTNode(kind=f"else{i}") for i in range(5)]
    node = ASTNode(kind="if")
    node.else_children = else_children
    r = repr(node)
    assert "else_children=[" in r
    assert "..." in r


def test_astnode_repr_shows_all_children_if_three_or_fewer() -> None:
    children = [ASTNode(kind=f"child{i}") for i in range(3)]
    node = ASTNode(kind="parent", children=children)
    r = repr(node)
    assert "children=[" in r
    assert "..." not in r


def test_astnode_repr_truncates_children_if_more_than_three() -> None:
    children = [ASTNode(kind=f"child{i}") for i in range(5)]
    node = ASTNode(kind="parent", children=children)
    r = repr(node)
    assert "children=[" in r
    assert "..." in r


def test_to_dict_with_astnode_value() -> None:
    inner = ASTNode(kind="number", value="42")
    outer = ASTNode(kind="wrapper", value=inner)
    d = outer.to_dict()
    assert isinstance(d["value"], dict)
    assert d["value"]["kind"] == "number"


def test_to_dict_with_list_value() -> None:
    outer = ASTNode(kind="container", value="temp")
    outer.value = cast(Any, [ASTNode(kind="val", value="x"), "raw", 123])
    d = outer.to_dict()
    assert isinstance(d["value"][0], dict)
    assert d["value"][1] == "raw"
    assert d["value"][2] == 123
