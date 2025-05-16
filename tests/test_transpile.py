import pytest

from limit.limit_ast import ASTNode
from limit.limit_transpile import Transpiler


class DummyEmitter:
    def __init__(self):
        self.calls = []

    def emit_assign(self, node):
        self.calls.append(("assign", node))

    def get_output(self):
        return "result"


def test_transpiler_selects_python(monkeypatch):
    monkeypatch.setattr("limit.limit_transpile.PythonEmitter", DummyEmitter)
    transpiler = Transpiler("py")
    assert isinstance(transpiler.emitter, DummyEmitter)


def test_transpiler_selects_python_case_insensitive(monkeypatch):
    monkeypatch.setattr("limit.limit_transpile.PythonEmitter", DummyEmitter)
    transpiler = Transpiler("PYTHON")
    assert isinstance(transpiler.emitter, DummyEmitter)


def test_transpiler_invalid_target_raises():
    with pytest.raises(ValueError, match="Unknown transpilation target"):
        Transpiler("brainfuck")


def test_transpiler_rejects_non_ast(monkeypatch):
    monkeypatch.setattr("limit.limit_transpile.PythonEmitter", DummyEmitter)
    transpiler = Transpiler("py")
    with pytest.raises(TypeError, match="ASTNode"):
        transpiler.transpile(["not-an-ast"])


def test_transpiler_calls_emit_method(monkeypatch):
    dummy = DummyEmitter()
    monkeypatch.setattr("limit.limit_transpile.PythonEmitter", lambda: dummy)
    transpiler = Transpiler("py")
    node = ASTNode("assign", "x", [], line=1, col=2)
    result = transpiler.transpile([node])
    assert result == "result"
    assert dummy.calls == [("assign", node)]


def test_transpiler_missing_emit_method_raises(monkeypatch):
    class IncompleteEmitter:
        def get_output(self):
            return "incomplete"

    monkeypatch.setattr(
        "limit.limit_transpile.PythonEmitter", lambda: IncompleteEmitter()
    )
    transpiler = Transpiler("py")
    node = ASTNode("nonexistent", None, [], line=3, col=4)
    with pytest.raises(
        NotImplementedError, match="No emitter method for node kind 'nonexistent'"
    ):
        transpiler.transpile([node])
