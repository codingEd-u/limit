# tests/test_py_emitter.py

from typing import Literal, cast

import pytest

from limit.emitters.py_emitter import PythonEmitter
from limit.limit_ast import ASTNode


def test_emit_identifier() -> None:
    emitter = PythonEmitter()
    assert emitter.emit_identifier(ASTNode("identifier", "x")) == "x"


def test_emit_number() -> None:
    emitter = PythonEmitter()
    assert emitter.emit_number(ASTNode("number", "42")) == "42"


def test_emit_string() -> None:
    emitter = PythonEmitter()
    assert emitter.emit_string(ASTNode("string", "hi")) == "'hi'"


def test_emit_expr_arith() -> None:
    emitter = PythonEmitter()
    node = ASTNode("arith", "PLUS", [ASTNode("number", "1"), ASTNode("number", "2")])
    assert emitter.emit_expr(node) == "(1 + 2)"


def test_emit_expr_bool_not() -> None:
    emitter = PythonEmitter()
    node = ASTNode("bool", "NOT", [ASTNode("identifier", "flag")])
    assert emitter.emit_expr(node) == "(not flag)"


def test_emit_expr_bool_and() -> None:
    emitter = PythonEmitter()
    node = ASTNode(
        "bool", "AND", [ASTNode("identifier", "a"), ASTNode("identifier", "b")]
    )
    assert emitter.emit_expr(node) == "(a and b)"


def test_emit_expr_call() -> None:
    emitter = PythonEmitter()
    callee = ASTNode("identifier", "f")
    node = ASTNode("call", callee, [ASTNode("identifier", "x"), ASTNode("number", "5")])
    assert emitter.emit_expr(node) == "f(x, 5)"


def test_emit_func_definition() -> None:
    emitter = PythonEmitter()
    func = ASTNode(
        "func",
        "add",
        [
            ASTNode(
                "return",
                children=[
                    ASTNode(
                        "arith",
                        "PLUS",
                        [ASTNode("identifier", "a"), ASTNode("identifier", "b")],
                    )
                ],
            )
        ],
        type_=["a", "b"],  # type: ignore
    )
    emitter._visit(func)
    output = emitter.get_output()
    assert "def add(a, b):" in output
    assert "return (a + b)" in output


def test_emit_export_func() -> None:
    emitter = PythonEmitter()
    func = ASTNode(
        "func", "hello", [ASTNode("print", children=[ASTNode("string", "hi")])]
    )
    export = ASTNode("export", "hello", [func])
    emitter._visit(export)
    out = emitter.get_output()
    assert "__all__ = ['hello']" in out
    assert "def hello()" in out or "def hello() -> None:" in out


def test_emit_loop_for() -> None:
    emitter = PythonEmitter()
    node = ASTNode(
        "loop",
        "FOR",
        [
            ASTNode(
                "range",
                None,
                [
                    ASTNode("identifier", "i"),
                    ASTNode("number", "5"),  # FIXED: no double wrap
                ],
            ),
            ASTNode("print", children=[ASTNode("identifier", "i")]),
        ],
    )
    emitter._visit(node)
    out = emitter.get_output()
    assert "for i in range(0, 5):" in out
    assert "print(i)" in out


def test_emit_loop_while() -> None:
    emitter = PythonEmitter()
    node = ASTNode(
        "loop",
        "WHILE",
        [
            ASTNode("identifier", "x"),
            ASTNode("print", children=[ASTNode("identifier", "x")]),
        ],
    )
    emitter._visit(node)
    out = emitter.get_output()
    assert "while x:" in out
    assert "print(x)" in out


def test_emit_if_else() -> None:
    emitter = PythonEmitter()
    node = ASTNode(
        "if",
        ASTNode("identifier", "x"),  # wrapped condition as ASTNode
        [
            ASTNode("print", children=[ASTNode("string", "positive")]),
        ],
    )
    node.else_children = [
        ASTNode("print", children=[ASTNode("string", "non-positive")])
    ]
    emitter._visit(node)
    out = emitter.get_output()
    assert "if x:" in out
    assert "else:" in out
    assert "print('positive')" in out
    assert "print('non-positive')" in out


def test_emit_try_catch_finally() -> None:
    emitter = PythonEmitter()
    try_node = ASTNode(
        "try",
        None,
        [
            ASTNode("print", children=[ASTNode("string", "try")]),
            ASTNode(
                "catch", "e", [ASTNode("print", children=[ASTNode("string", "catch")])]
            ),
            ASTNode(
                "finally",
                None,
                [ASTNode("print", children=[ASTNode("string", "finally")])],
            ),
        ],
    )
    emitter._visit(try_node)
    out = emitter.get_output()
    assert "try:" in out
    assert "except Exception as e:" in out
    assert "finally:" in out
    assert "print('try')" in out
    assert "print('catch')" in out
    assert "print('finally')" in out


def test_emit_class() -> None:
    emitter = PythonEmitter()
    class_node = ASTNode("class", "MyClass", [])
    emitter._visit(class_node)
    out = emitter.get_output()
    assert "class MyClass(object):" in out
    assert "pass" in out


def test_emit_func_with_return_type() -> None:
    emitter = PythonEmitter()
    func = ASTNode(
        "func",
        "double",
        [
            ASTNode(
                "return",
                children=[
                    ASTNode(
                        "arith",
                        "MULT",
                        [ASTNode("identifier", "x"), ASTNode("number", "2")],
                    )
                ],
            )
        ],
        type_=["x"],  # type: ignore
        return_type="int",
    )
    emitter._visit(func)
    out = emitter.get_output()
    assert "def double(x) -> int:" in out
    assert "return (x * 2)" in out


def test_emit_propagate() -> None:
    emitter = PythonEmitter()
    node = ASTNode("propagate", None, [ASTNode("identifier", "err")])
    emitter._visit(node)
    out = emitter.get_output()
    assert "__tmp = err" in out
    assert "if __tmp: return __tmp" in out


def test_emit_input() -> None:
    emitter = PythonEmitter()
    node = ASTNode("input_from_file", "val", [ASTNode("file", "val")])
    emitter._visit(node)
    out = emitter.get_output()
    assert "val = open('val').read()" in out


def test_emit_break_and_continue() -> None:
    emitter = PythonEmitter()
    loop = ASTNode(
        "loop",
        "WHILE",
        [ASTNode("identifier", "x"), ASTNode("break"), ASTNode("continue")],
    )
    emitter._visit(loop)
    out = emitter.get_output()
    assert "break" in out
    assert "continue" in out


def test_emit_export_with_func() -> None:
    emitter = PythonEmitter()
    fn = ASTNode(
        "func", "greet", [ASTNode("print", children=[ASTNode("string", "hi")])]
    )
    export = ASTNode("export", "greet", [fn])
    emitter._visit(export)
    out = emitter.get_output()
    assert "__all__ = ['greet']" in out
    assert "def greet()" in out or "def greet() -> None:" in out


def test_emit_expr_member_chain() -> None:
    emitter = PythonEmitter()
    node = ASTNode(
        "member", "b", [ASTNode("member", "a", [ASTNode("identifier", "x")])]
    )
    result = emitter.emit_expr(node)
    assert result == "x.a.b"


def test_emit_expr_new() -> None:
    emitter = PythonEmitter()
    node = ASTNode("new", "MyClass", [ASTNode("number", "1"), ASTNode("string", "x")])
    assert emitter.emit_expr(node) == "MyClass(1, 'x')"


def test_emit_expr_call_no_args() -> None:
    emitter = PythonEmitter()
    callee = ASTNode("identifier", "f")
    node = ASTNode("call", callee, [])
    assert emitter.emit_expr(node) == "f()"


def test_emit_literal_values() -> None:
    emitter = PythonEmitter()
    assert emitter.emit_literal(ASTNode("literal", "TRUE")) == "True"
    assert emitter.emit_literal(ASTNode("literal", "FALSE")) == "False"
    assert emitter.emit_literal(ASTNode("literal", "NULL")) == "None"


def test_emit_literal_invalid() -> None:
    emitter = PythonEmitter()
    with pytest.raises(ValueError, match="Unknown literal: MAYBE"):
        emitter.emit_literal(ASTNode("literal", "MAYBE"))


def test_emit_expr_empty() -> None:
    emitter = PythonEmitter()
    node = ASTNode("empty", None)
    assert emitter.emit_expr_empty(node) == "None"


def test_emit_float() -> None:
    emitter = PythonEmitter()
    node = ASTNode("float", "3.14")
    assert emitter.emit_float(node) == "3.14"


def test_emit_expr_call_type_error() -> None:
    emitter = PythonEmitter()
    with pytest.raises(TypeError, match="Expected ASTNode for call callee"):
        node = ASTNode("call", "not_astnode", [])
        emitter.emit_expr_call(node)


def test_emit_assign() -> None:
    emitter = PythonEmitter()
    assign_node = ASTNode(
        "assign", None, [ASTNode("identifier", "x"), ASTNode("number", "10")]
    )
    emitter._visit(assign_node)
    out = emitter.get_output()
    assert "x = 10" in out


def test_emit_print_empty() -> None:
    emitter = PythonEmitter()
    node = ASTNode("print", None, [])
    emitter._visit(node)
    out = emitter.get_output()
    assert "print()" in out


def test_emit_return_with_value() -> None:
    emitter = PythonEmitter()
    node = ASTNode("return", None, [ASTNode("number", "5")])
    emitter._visit(node)
    out = emitter.get_output()
    assert "return 5" in out


def test_emit_return_without_value() -> None:
    emitter = PythonEmitter()
    node = ASTNode("return", None, [])
    emitter._visit(node)
    out = emitter.get_output()
    assert "return" in out
    assert "return " not in out.strip()  # ensure it's just `return`, no value


def test_emit_propagate_type_error() -> None:
    emitter = PythonEmitter()
    node = ASTNode("propagate", None, ["not_astnode"])  # type: ignore
    with pytest.raises(TypeError, match="Expected ASTNode in propagate"):
        emitter._visit(node)


def test_emit_call_statement() -> None:
    emitter = PythonEmitter()
    node = ASTNode("call", ASTNode("identifier", "greet"), [ASTNode("string", "world")])
    emitter.emit_call(node)  # use emit_call directly, not _visit
    out = emitter.get_output()
    assert "greet('world')" in out


def test_emit_expr_stmt() -> None:
    emitter = PythonEmitter()
    node = ASTNode(
        "expr_stmt", None, [ASTNode("call", ASTNode("identifier", "hello"), [])]
    )
    emitter._visit(node)
    out = emitter.get_output()
    assert "hello()" in out


def test_emit_import() -> None:
    emitter = PythonEmitter()
    node = ASTNode("import", "math")
    emitter._visit(node)
    out = emitter.get_output()
    assert "import math" in out


@pytest.mark.parametrize(
    "token,op",
    [
        ("LT", "<"),
        ("GT", ">"),
        ("EQ", "=="),
        ("NE", "!="),
        ("LE", "<="),
        ("GE", ">="),
    ],
)  # type: ignore[misc]
def test_emit_expr_compare_standard(
    token: Literal["LT", "GT", "EQ", "NE", "LE", "GE"], op: str
) -> None:
    emitter = PythonEmitter()
    node = ASTNode(
        "compare", token, [ASTNode("identifier", "a"), ASTNode("identifier", "b")]
    )
    assert emitter.emit_expr_compare(node) == f"(a {op} b)"


def test_emit_module_noop() -> None:
    emitter = PythonEmitter()
    node = ASTNode("module", "whatever")
    emitter._visit(node)
    out = emitter.get_output()
    assert out.strip() == ""  # should produce no output


def test_emit_call_method() -> None:
    emitter = PythonEmitter()
    node = ASTNode(
        "call",
        ASTNode("identifier", "do_something"),
        [ASTNode("number", "1"), ASTNode("number", "2")],
    )
    emitter.emit_call(node)
    out = emitter.get_output()
    assert "do_something(1, 2)" in out


def test_emit_input_int() -> None:
    emitter = PythonEmitter()
    node = ASTNode("input", "x")
    node.type = "int"
    emitter._visit(node)
    out = emitter.get_output()
    assert "x = int(input())" in out


def test_emit_input_float() -> None:
    emitter = PythonEmitter()
    node = ASTNode("input", "y")
    node.type = "float"
    emitter._visit(node)
    out = emitter.get_output()
    assert "y = float(input())" in out


def test_emit_input_default() -> None:
    emitter = PythonEmitter()
    node = ASTNode("input", "z")
    emitter._visit(node)
    out = emitter.get_output()
    assert "z = input()" in out


def test_emit_input_from_file_type_error() -> None:
    emitter = PythonEmitter()
    node = ASTNode("input_from_file", "val", children=["not_astnode"])  # type: ignore
    with pytest.raises(TypeError, match=r"Expected ASTNode in children\[0\]"):
        emitter._visit(node)


def test_emit_skip() -> None:
    emitter = PythonEmitter()
    node = ASTNode("skip")
    emitter._visit(node)
    out = emitter.get_output()
    assert out.strip() == "pass"


def test_emit_class_extends_type_error() -> None:
    emitter = PythonEmitter()
    bad_node = ASTNode("extends", value=cast(str, 123))
    node = ASTNode("class", "MyClass", [bad_node])
    with pytest.raises(TypeError, match="Expected string in class extends base"):
        emitter._visit(node)


def test_emit_class_with_extends_and_init() -> None:
    emitter = PythonEmitter()
    node = ASTNode(
        "class",
        "Child",
        [
            ASTNode("extends", "Parent"),
            ASTNode("func", "init", [ASTNode("return")], type_=[]),  # type: ignore
        ],
    )
    emitter._visit(node)
    out = emitter.get_output()
    assert "class Child(Parent):" in out
    assert "def __init__(self):" in out
    assert "super().__init__()" in out


def test_emit_class_with_no_body() -> None:
    emitter = PythonEmitter()
    node = ASTNode("class", "EmptyClass", [])
    emitter._visit(node)
    out = emitter.get_output()
    assert "class EmptyClass(object):" in out
    assert "pass" in out


def test_visit_skip() -> None:
    emitter = PythonEmitter()
    node = ASTNode("skip")
    result = emitter._visit(node)
    assert result is None
    assert emitter.get_output().strip() == "pass"


def test_visit_expr_kind() -> None:
    emitter = PythonEmitter()
    node = ASTNode("bool", "NOT", [ASTNode("identifier", "flag")])
    result = emitter._visit(node)
    assert result == "(not flag)"


def test_visit_unsupported_kind() -> None:
    emitter = PythonEmitter()
    node = ASTNode("nonsense")
    with pytest.raises(
        NotImplementedError, match="PythonEmitter: no emitter for nonsense"
    ):
        emitter._visit(node)


def test_emit_func_with_str_type() -> None:
    emitter = PythonEmitter()
    node = ASTNode("func", "sum", [], type_="a, b")
    emitter._visit(node)
    out = emitter.get_output()
    assert "def sum(a, b):" in out


def test_emit_func_with_no_body() -> None:
    emitter = PythonEmitter()
    node = ASTNode("func", "noop", [])  # no children
    emitter._visit(node)
    out = emitter.get_output()
    assert "def noop():" in out
    assert "pass" in out


def test_emit_expr_fallback_method() -> None:
    emitter = PythonEmitter()
    node = ASTNode("print", children=[ASTNode("string", "hello")])
    result = emitter.emit_expr(node)
    assert result == "None"  # fallback emits None, but gets str()'d


def test_emit_expr_not_implemented() -> None:
    emitter = PythonEmitter()
    node = ASTNode("unknown_expr_kind")
    with pytest.raises(
        NotImplementedError, match="No expression emitter for kind 'unknown_expr_kind'"
    ):
        emitter.emit_expr(node)


def test_emit_loop_for_to_n() -> None:
    emitter = PythonEmitter()
    node = ASTNode(
        "loop",
        "FOR",
        [
            ASTNode(
                "range", None, [ASTNode("identifier", "i"), ASTNode("number", "5")]
            ),
            ASTNode("print", children=[ASTNode("identifier", "i")]),
        ],
    )
    emitter._visit(node)
    out = emitter.get_output()
    assert "for i in range(0, 5):" in out


def test_emit_loop_for_at_to() -> None:
    emitter = PythonEmitter()
    node = ASTNode(
        "loop",
        "FOR",
        [
            ASTNode(
                "range",
                None,
                [
                    ASTNode("identifier", "i"),
                    ASTNode("number", "2"),
                    ASTNode("number", "8"),
                ],
            ),
            ASTNode("print", children=[ASTNode("identifier", "i")]),
        ],
    )
    emitter._visit(node)
    out = emitter.get_output()
    assert "for i in range(2, 8):" in out


def test_emit_loop_with_else_block_error() -> None:
    emitter = PythonEmitter()
    node = ASTNode("loop", "WHILE", [ASTNode("identifier", "x")])
    node.else_children = [ASTNode("print", children=[ASTNode("string", "nope")])]
    with pytest.raises(
        NotImplementedError, match="ELSE blocks are only valid after IF, not loops."
    ):
        emitter._visit(node)


def test_emit_if_invalid_condition_type() -> None:
    emitter = PythonEmitter()
    node = ASTNode(
        "if",
        "not_an_astnode",  # invalid condition
        [ASTNode("print", children=[ASTNode("string", "oops")])],
    )
    with pytest.raises(TypeError, match="Expected ASTNode in value of IF condition"):
        emitter._visit(node)


def test_py_emitter_class_variants() -> None:
    from limit.emitters.py_emitter import PythonEmitter
    from limit.limit_ast import ASTNode

    # Case 1: Class with no children (triggers "pass")
    emitter = PythonEmitter()
    empty_class = ASTNode(kind="class", value="Empty", children=[])
    emitter._visit(empty_class)
    assert "class Empty(object):" in emitter.get_output()
    assert "pass" in emitter.get_output()

    # Case 2: Class with __init__ and no extends
    emitter = PythonEmitter()
    init_func = ASTNode(kind="func", value="init", children=[])
    class_with_init = ASTNode(kind="class", value="Simple", children=[init_func])
    emitter._visit(class_with_init)
    code = emitter.get_output()
    assert "class Simple(object):" in code
    assert "def __init__(" in code

    # Case 3: Class with __init__ and extends triggers super().__init__()
    emitter = PythonEmitter()
    init_func2 = ASTNode(kind="func", value="init", children=[])
    extended = ASTNode(kind="extends", value="Base")
    class_extending = ASTNode(
        kind="class", value="Child", children=[extended, init_func2]
    )
    emitter._visit(class_extending)
    code = emitter.get_output()
    assert "class Child(Base):" in code
    assert "super().__init__()" in code

    # Case 4: Class with regular method
    emitter = PythonEmitter()
    method = ASTNode(kind="func", value="run", children=[])
    class_with_method = ASTNode(kind="class", value="Runner", children=[method])
    emitter._visit(class_with_method)
    code = emitter.get_output()
    assert "def run(" in code


def test_emit_if_without_else() -> None:
    emitter = PythonEmitter()
    node = ASTNode(
        "if",
        ASTNode("identifier", "x"),
        [ASTNode("print", children=[ASTNode("string", "yes")])],
    )
    emitter._visit(node)
    out = emitter.get_output()
    assert "if x:" in out
    assert "else" not in out


def test_emit_try_no_catch_finally() -> None:
    emitter = PythonEmitter()
    node = ASTNode(
        "try", None, [ASTNode("print", children=[ASTNode("string", "only try")])]
    )
    emitter._visit(node)
    out = emitter.get_output()
    assert "try:" in out
    assert "only try" in out


def test_emit_export_without_func() -> None:
    emitter = PythonEmitter()
    node = ASTNode("export", "val", [])
    emitter._visit(node)
    out = emitter.get_output()
    assert "__all__ = ['val']" in out


def test_emit_loop_range_3_children() -> None:
    emitter = PythonEmitter()
    node = ASTNode(
        "loop",
        "FOR",
        [
            ASTNode(
                "range",
                None,
                [
                    ASTNode("identifier", "i"),
                    ASTNode("number", "1"),
                    ASTNode("number", "5"),
                ],
            ),
            ASTNode("print", children=[ASTNode("identifier", "i")]),
        ],
    )
    emitter._visit(node)
    out = emitter.get_output()
    assert "range(1, 5)" in out


def test_emit_class_with_regular_func() -> None:
    emitter = PythonEmitter()
    method = ASTNode(
        "func", "run", [ASTNode("return", children=[ASTNode("number", "1")])]
    )
    node = ASTNode("class", "Task", [method])
    emitter._visit(node)
    out = emitter.get_output()
    assert "def run" in out


def test_emit_loop_for_at_to_by_full_branch() -> None:
    emitter = PythonEmitter()
    node = ASTNode(
        "loop",
        "FOR",
        [
            ASTNode(
                "range",
                None,
                [
                    ASTNode("identifier", "i"),
                    ASTNode("number", "1"),
                    ASTNode("number", "10"),
                    ASTNode("number", "2"),
                ],
            ),
            ASTNode("print", children=[ASTNode("identifier", "i")]),
        ],
    )
    emitter._visit(node)
    out = emitter.get_output()
    assert "range(1, 10, 2)" in out


def test_emit_class_super_insert_logic() -> None:
    emitter = PythonEmitter()
    init = ASTNode("func", "init", [])
    node = ASTNode("class", "Child", [ASTNode("extends", "Base"), init])
    emitter._visit(node)
    out = emitter.get_output()
    assert "super().__init__()" in out


def test_emit_expr_preferred_but_missing() -> None:
    emitter = PythonEmitter()
    node = ASTNode("file", "path/to/file")

    # Temporarily remove class method to force fallback
    old = PythonEmitter.emit_expr_file
    delattr(PythonEmitter, "emit_expr_file")
    try:
        result = emitter.emit_expr(node)
        assert "path/to/file" in result
    finally:
        PythonEmitter.emit_expr_file = old


def test_emit_loop_for_at_to_by() -> None:
    emitter = PythonEmitter()
    node = ASTNode(
        "loop",
        "FOR",  # ✅ Must be "FOR" exactly
        [
            ASTNode(
                "range",
                None,
                [
                    ASTNode("identifier", "i"),
                    ASTNode("number", "1"),
                    ASTNode("number", "10"),
                    ASTNode("number", "2"),
                ],
            ),
            ASTNode("print", children=[ASTNode("identifier", "i")]),
        ],
    )
    emitter._visit(node)
    out = emitter.get_output()
    assert "for i in range(1, 10, 2):" in out


def test_emit_expr_bool_truthy() -> None:
    emitter = PythonEmitter()
    node = ASTNode("bool", "TRUTHY", [ASTNode("identifier", "val")])
    result = emitter.emit_expr(node)
    assert result == "bool(val)"
