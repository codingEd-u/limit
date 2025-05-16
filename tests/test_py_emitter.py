# tests/test_py_emitter.py

from limit.emitters.py_emitter import PythonEmitter
from limit.limit_ast import ASTNode


def test_emit_identifier():
    emitter = PythonEmitter()
    assert emitter.emit_identifier(ASTNode("identifier", "x")) == "x"


def test_emit_number():
    emitter = PythonEmitter()
    assert emitter.emit_number(ASTNode("number", "42")) == "42"


def test_emit_string():
    emitter = PythonEmitter()
    assert emitter.emit_string(ASTNode("string", "hi")) == "'hi'"


def test_emit_expr_arith():
    emitter = PythonEmitter()
    node = ASTNode("arith", "PLUS", [ASTNode("number", "1"), ASTNode("number", "2")])
    assert emitter.emit_expr(node) == "(1 + 2)"


def test_emit_expr_bool_not():
    emitter = PythonEmitter()
    node = ASTNode("bool", "NOT", [ASTNode("identifier", "flag")])
    assert emitter.emit_expr(node) == "(not flag)"


def test_emit_expr_bool_and():
    emitter = PythonEmitter()
    node = ASTNode(
        "bool", "AND", [ASTNode("identifier", "a"), ASTNode("identifier", "b")]
    )
    assert emitter.emit_expr(node) == "(a and b)"


def test_emit_expr_call():
    emitter = PythonEmitter()
    callee = ASTNode("identifier", "f")
    node = ASTNode("call", callee, [ASTNode("identifier", "x"), ASTNode("number", "5")])
    assert emitter.emit_expr(node) == "f(x, 5)"


def test_emit_func_definition():
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
        type_=["a", "b"],
    )
    emitter._visit(func)
    output = emitter.get_output()
    assert "def add(a, b):" in output
    assert "return (a + b)" in output


def test_emit_export_func():
    emitter = PythonEmitter()
    func = ASTNode(
        "func", "hello", [ASTNode("print", children=[ASTNode("string", "hi")])]
    )
    export = ASTNode("export", "hello", [func])
    emitter._visit(export)
    out = emitter.get_output()
    assert "__all__ = ['hello']" in out
    assert "def hello()" in out or "def hello():" in out


def test_emit_loop_for():
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


def test_emit_loop_while():
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


def test_emit_if_else():
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


def test_emit_try_catch_finally():
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


def test_emit_class():
    emitter = PythonEmitter()
    class_node = ASTNode("class", "MyClass", [])
    emitter._visit(class_node)
    out = emitter.get_output()
    assert "class MyClass(object):" in out
    assert "pass" in out


def test_emit_func_with_return_type():
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
        type_=["x"],
        return_type="int",
    )
    emitter._visit(func)
    out = emitter.get_output()
    assert "def double(x) -> int:" in out
    assert "return (x * 2)" in out


def test_emit_propagate():
    emitter = PythonEmitter()
    node = ASTNode("propagate", None, [ASTNode("identifier", "err")])
    emitter._visit(node)
    out = emitter.get_output()
    assert "__tmp = err" in out
    assert "if __tmp: return __tmp" in out


def test_emit_input():
    emitter = PythonEmitter()
    node = ASTNode("input_from_file", "val", [ASTNode("file", "val")])
    emitter._visit(node)
    out = emitter.get_output()
    assert "val = open('val').read()" in out


def test_emit_break_and_continue():
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


def test_emit_export_with_func():
    emitter = PythonEmitter()
    fn = ASTNode(
        "func", "greet", [ASTNode("print", children=[ASTNode("string", "hi")])]
    )
    export = ASTNode("export", "greet", [fn])
    emitter._visit(export)
    out = emitter.get_output()
    assert "__all__ = ['greet']" in out
    assert "def greet()" in out or "def greet():" in out


def test_emit_expr_member_chain():
    emitter = PythonEmitter()
    node = ASTNode(
        "member", "b", [ASTNode("member", "a", [ASTNode("identifier", "x")])]
    )
    result = emitter.emit_expr(node)
    assert result == "x.a.b"


def test_emit_expr_new():
    emitter = PythonEmitter()
    node = ASTNode("new", "MyClass", [ASTNode("number", "1"), ASTNode("string", "x")])
    assert emitter.emit_expr(node) == "MyClass(1, 'x')"


def test_emit_expr_call_no_args():
    emitter = PythonEmitter()
    callee = ASTNode("identifier", "f")
    node = ASTNode("call", callee, [])
    assert emitter.emit_expr(node) == "f()"
