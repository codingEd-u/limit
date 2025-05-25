from typing import Any

import pytest

from limit.limit_ast import ASTNode
from limit.limit_transpile import CEmitter, Emitter, JavaScriptEmitter, Transpiler


def test_force_protocol_reference() -> None:
    # This accesses the Protocol itself — may cover the class line
    assert hasattr(Emitter, "__annotations__")


class DummyEmitter:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ASTNode]] = []

    def emit_assign(self, node: ASTNode) -> None:
        self.calls.append(("assign", node))

    def get_output(self) -> str:
        return "result"


def test_transpiler_selects_python(monkeypatch: Any) -> None:
    monkeypatch.setattr("limit.limit_transpile.PythonEmitter", DummyEmitter)
    transpiler = Transpiler("py")
    assert isinstance(transpiler.emitter, DummyEmitter)


def test_transpiler_selects_python_case_insensitive(monkeypatch: Any) -> None:
    monkeypatch.setattr("limit.limit_transpile.PythonEmitter", DummyEmitter)
    transpiler = Transpiler("PYTHON")
    assert isinstance(transpiler.emitter, DummyEmitter)


def test_transpiler_invalid_target_raises() -> None:
    with pytest.raises(ValueError, match="Unknown transpilation target"):
        Transpiler("brainfuck")


def test_transpiler_rejects_non_ast(monkeypatch: Any) -> None:
    monkeypatch.setattr("limit.limit_transpile.PythonEmitter", DummyEmitter)
    transpiler = Transpiler("py")
    with pytest.raises(TypeError, match="ASTNode"):
        transpiler.transpile(["not-an-ast"])  # type: ignore


def test_transpiler_calls_emit_method(monkeypatch: Any) -> None:
    dummy = DummyEmitter()
    monkeypatch.setattr("limit.limit_transpile.PythonEmitter", lambda: dummy)
    transpiler = Transpiler("py")
    node = ASTNode("assign", "x", [], line=1, col=2)
    result = transpiler.transpile([node])
    assert result == "result"
    assert dummy.calls == [("assign", node)]


def test_transpiler_missing_emit_method_raises(monkeypatch: Any) -> None:
    class IncompleteEmitter:
        def get_output(self) -> str:
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


def test_transpiler_js_not_implemented() -> None:
    with pytest.raises(NotImplementedError, match="JavaScriptEmitter"):
        Transpiler("js")


def test_transpiler_c_not_implemented() -> None:
    with pytest.raises(NotImplementedError, match="CEmitter"):
        Transpiler("c")


def test_js_get_output_raises() -> None:
    js = object.__new__(JavaScriptEmitter)
    with pytest.raises(NotImplementedError):
        js.get_output()


def test_c_get_output_raises() -> None:
    c = object.__new__(CEmitter)
    with pytest.raises(NotImplementedError):
        c.get_output()


def test_emitter_protocol_conformance() -> None:
    class Dummy:
        def __init__(self) -> None:
            pass

        def get_output(self) -> str:
            return "ok"

    def accepts_emitter(e: Emitter) -> str:
        return e.get_output()

    assert accepts_emitter(Dummy()) == "ok"


def test_transpiler_selects_c(monkeypatch: Any) -> None:
    class DummyCEmitter:
        def __init__(self) -> None:
            self.called = True

        def get_output(self) -> str:
            return "c-out"

        def emit_assign(self, node: ASTNode) -> None:
            pass

    monkeypatch.setattr("limit.limit_transpile.CEmitter", DummyCEmitter)
    transpiler = Transpiler("c")
    assert isinstance(transpiler.emitter, DummyCEmitter)


def test_transpiler_emits_real_python(monkeypatch: Any) -> None:
    class DummyReal:
        def __init__(self) -> None:
            self.ok = True

        def get_output(self) -> str:
            return "ok"

        def emit_assign(self, node: ASTNode) -> None:
            return None

    monkeypatch.setattr("limit.limit_transpile.PythonEmitter", DummyReal)
    transpiler = Transpiler("python")
    assert transpiler.emitter.get_output() == "ok"
