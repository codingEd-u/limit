import builtins
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

import limit.limit_repl
from limit.limit_ast import ASTNode
from limit.limit_lexer import CharacterStream, Token
from limit.limit_parser import Parser
from limit.limit_repl import handle_sugar_command, print_traceback, start_repl, uimap


@pytest.fixture  # type: ignore[misc]
def mock_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    class MockLexer:
        def __init__(self, cs: CharacterStream) -> None:
            self.tokens = iter(
                [
                    Token("PRINT", "!", 1, 1),
                    Token("STRING", "hello", 1, 3),
                    Token("EOF", "EOF", 1, 10),
                ]
            )

        def next_token(self) -> Token:
            return next(self.tokens)

    class MockParser:
        def __init__(self, tokens: list[Token]) -> None:
            pass

        def parse(self) -> list[Any]:
            return ["FAKE_AST"]

    class MockTranspiler:
        def __init__(self, target: str) -> None:
            self.target = target

        def transpile(self, ast: list[ASTNode]) -> str:
            if self.target == "py":
                return 'print("hello")'
            return "noop();"

    monkeypatch.setattr("limit.limit_lexer.Lexer", MockLexer)
    monkeypatch.setattr("limit.limit_parser.Parser", MockParser)
    monkeypatch.setattr("limit.limit_transpile.Transpiler", MockTranspiler)


def test_repl_quit(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(builtins, "input", lambda _: "quit")
    start_repl()
    out = capsys.readouterr().out
    assert "Exiting Limit REPL" in out


def test_repl_exit(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(builtins, "input", lambda _: "exit")
    start_repl()
    out = capsys.readouterr().out
    assert "Exiting Limit REPL" in out


def test_repl_empty_input_skipped(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = iter(["   ", "quit"])
    monkeypatch.setattr(builtins, "input", lambda _: next(calls))
    start_repl()
    out = capsys.readouterr().out
    assert "Exiting Limit REPL" in out


def test_repl_valid_input_exec(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mock_pipeline: Any,
) -> None:
    calls = iter(['! "hello"', "quit"])
    monkeypatch.setattr(builtins, "input", lambda _: next(calls))
    start_repl(verbose=True)
    out = capsys.readouterr().out.lower()
    assert "hello" in out


@pytest.mark.usefixtures("mock_pipeline")  # type: ignore[misc]
def test_repl_valid_input_unsupported_target(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mock_pipeline: Any,
) -> None:
    calls = iter(['! "hello"', "quit"])
    monkeypatch.setattr("builtins.input", lambda _: next(calls))

    with patch("builtins.print") as mock_print:
        start_repl(target="c")

    printed = [
        "".join(str(arg) for call in mock_print.call_args_list for arg in call.args)
    ]
    output = "\n".join(printed).lower()
    assert "transpilation error: cemitter is not yet implemented." in output
    assert "exiting limit repl" in output


def test_repl_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        builtins, "input", lambda _: (_ for _ in ()).throw(KeyboardInterrupt())
    )
    start_repl()
    out = capsys.readouterr().out
    assert "Exiting Limit REPL" in out


def test_repl_verbose_mode(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = iter(["verbose-mode", "quit"])
    monkeypatch.setattr("builtins.input", lambda _: next(calls))
    start_repl()
    out = capsys.readouterr().out
    assert "verbose mode" in out.lower()


def test_repl_verbose_mode_toggle_and_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyLexer:
        def __init__(self, _: Any) -> None:
            self.tokens = iter(
                [Token("DUMMY", "dummy", 1, 1), Token("EOF", "EOF", 1, 2)]
            )

        def next_token(self) -> Token:
            return next(self.tokens)

    class DummyAST:
        def __init__(self) -> None:
            self.kind = "arith"
            self.value = "+"
            self.children: list[Any] = []

    class DummyEmitter:
        def emit_expr(self, _: Any) -> str:
            return "1 + 1"

        def get_output(self) -> str:
            return ""

        def _visit(self, _: Any) -> None:
            pass

    class DummyTranspiler:
        def __init__(self, _: Any) -> None:
            self.emitter = DummyEmitter()

    class DummyParser:
        def __init__(self, _: Any) -> None:
            pass

        def parse(self) -> list[Any]:
            return [DummyAST()]

    # First input toggles verbose mode OFF, second runs code, third exits
    inputs = iter(["verbose-mode", "1 + 1", "quit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    monkeypatch.setattr("limit.limit_lexer.Lexer", lambda _: DummyLexer(None))
    monkeypatch.setattr("limit.limit_parser.Parser", lambda _: DummyParser(None))
    monkeypatch.setattr(
        "limit.limit_transpile.Transpiler", lambda _: DummyTranspiler(None)
    )

    with patch("builtins.print") as mock_print:
        start_repl(verbose=True)

    output = "\n".join(
        "".join(str(arg) for arg in call.args) for call in mock_print.call_args_list
    ).lower()

    assert "[mode] >>> verbose mode off" in output
    assert "exiting limit repl" in output


def test_is_expression_node_covers_all_kinds() -> None:
    from limit.limit_repl import is_expression_node

    assert is_expression_node(ASTNode("arith"))
    assert is_expression_node(ASTNode("bool"))
    assert not is_expression_node(ASTNode("assign"))


def test_repl_entry_banner(monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = iter(["quit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    with patch("builtins.print") as mock_print:
        start_repl(verbose=False)
    output = "\n".join(
        "".join(str(arg) for arg in call.args) for call in mock_print.call_args_list
    ).lower()
    assert "limit repl" in output


def test_print_traceback_outputs_error() -> None:
    with patch("builtins.print") as mock_print:
        try:
            raise ValueError("intentional test error")
        except Exception:
            print_traceback()

    printed = [
        "".join(str(arg) for arg in call.args) for call in mock_print.call_args_list
    ]
    joined = "\n".join(printed).lower()

    assert "[error] >>>" in joined
    assert "valueerror" in joined
    assert "intentional test error" in joined


def test_repl_lexer_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    class ExplodingLexer:
        def __init__(self, stream: CharacterStream) -> None:
            pass

        def next_token(self) -> Token:
            print(">>> NEXT_TOKEN CALLED <<<")
            raise RuntimeError("boom")

    inputs = iter(["INVALID", "quit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    monkeypatch.setattr("limit.limit_repl.Lexer", ExplodingLexer)

    with patch("builtins.print") as mock_print:
        start_repl(verbose=True)

    output = "\n".join(
        "".join(str(arg) for arg in call.args) for call in mock_print.call_args_list
    ).lower()

    assert "[error] >>>" in output
    assert "runtimeerror" in output
    assert "boom" in output
    assert "exiting limit repl" in output


def test_repl_runtime_emit_expr_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeAST:
        def __init__(self) -> None:
            self.kind = "arith"
            self.value = "+"
            self.children: list[Any] = []

    class DummyLexer:
        def __init__(self, _: Any) -> None:
            self.tokens = iter(
                [Token("DUMMY", "dummy", 1, 1), Token("EOF", "EOF", 1, 2)]
            )

        def next_token(self) -> Token:
            return next(self.tokens)

    class BadEmitter:
        def __init__(self) -> None:
            self.lines: list[Any] = []

        def emit_expr(self, _: Any) -> str:
            return "1 / 0"  # runtime error

        def get_output(self) -> str:
            return ""

        def _visit(self, _: Any) -> None:
            pass

    class BadTranspiler:
        def __init__(self, _: Any) -> None:
            self.emitter = BadEmitter()

    class MockParser:
        def __init__(self, _: Any) -> None:
            pass

        def parse(self) -> list[Any]:
            return [FakeAST()]

    inputs = iter(["noop", "quit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    monkeypatch.setattr("limit.limit_repl.Lexer", lambda _: DummyLexer(None))
    monkeypatch.setattr("limit.limit_repl.Parser", lambda _: MockParser(None))
    monkeypatch.setattr("limit.limit_repl.Transpiler", lambda _: BadTranspiler(None))

    with patch("builtins.print") as mock_print:
        start_repl(verbose=True)

    output = "\n".join(
        "".join(str(arg) for arg in call.args) for call in mock_print.call_args_list
    ).lower()
    assert "[error] >>>" in output
    assert "division" in output or "zero" in output


def test_repl_fallback_expr_parse(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyLexer:
        def __init__(self, _: Any) -> None:
            self.tokens = iter(
                [Token("DUMMY", "dummy", 1, 1), Token("EOF", "EOF", 1, 2)]
            )

        def next_token(self) -> Token:
            return next(self.tokens)

    class MockParser:
        def __init__(self, _: Any) -> None:
            self.position = 0

        def parse(self) -> list[Any]:
            raise SyntaxError("Invalid statement")

        def parse_expr_entrypoint(self) -> list[ASTNode]:
            return [ASTNode("arith", "+", [])]

    class DummyEmitter:
        def get_output(self) -> str:
            return ""

        def _visit(self, _: Any) -> None:
            self.lines: list[Any] = []

    class DummyTranspiler:
        def __init__(self, _: Any) -> None:
            self.emitter = DummyEmitter()

    inputs = iter(["noop", "quit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    monkeypatch.setattr("limit.limit_repl.Lexer", lambda _: DummyLexer(None))
    monkeypatch.setattr("limit.limit_repl.Parser", lambda _: MockParser(None))
    monkeypatch.setattr("limit.limit_repl.Transpiler", lambda _: DummyTranspiler(None))

    with patch("builtins.print") as mock_print:
        start_repl(verbose=True)

    output = "\n".join(
        "".join(str(arg) for arg in call.args) for call in mock_print.call_args_list
    ).lower()
    assert "[error] >>>" not in output


def test_repl_invalid_token(monkeypatch: pytest.MonkeyPatch) -> None:
    class LexerWithError:
        def __init__(self, _: Any) -> None:
            self.tokens = iter([Token("ERROR", "$", 1, 1), Token("EOF", "EOF", 1, 2)])

        def next_token(self) -> Token:
            return next(self.tokens)

    inputs = iter(["trigger", "quit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    monkeypatch.setattr("limit.limit_repl.Lexer", lambda _: LexerWithError(None))
    monkeypatch.setattr("limit.limit_repl.Parser", lambda _: None)
    monkeypatch.setattr("limit.limit_repl.Transpiler", lambda _: None)

    with patch("builtins.print") as mock_print:
        start_repl(verbose=True)

    output = "\n".join(
        " ".join(str(arg) for arg in call.args) for call in mock_print.call_args_list
    ).lower()
    assert "invalid token" in output


def test_repl_transpiler_not_implemented(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyLexer:
        def __init__(self, _: Any) -> None:
            self.tokens = iter(
                [Token("DUMMY", "dummy", 1, 1), Token("EOF", "EOF", 1, 2)]
            )

        def next_token(self) -> Token:
            return next(self.tokens)

    class DummyParser:
        def __init__(self, _: Any) -> None:
            pass

        def parse(self) -> list[Any]:
            return ["dummy"]

    class FailingTranspiler:
        def __init__(self, _: Any) -> None:
            raise NotImplementedError("nope")

    inputs = iter(["noop", "quit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    monkeypatch.setattr("limit.limit_repl.Lexer", lambda _: DummyLexer(None))
    monkeypatch.setattr("limit.limit_repl.Parser", lambda _: DummyParser(None))
    monkeypatch.setattr("limit.limit_repl.Transpiler", FailingTranspiler)

    with patch("builtins.print") as mock_print:
        start_repl(verbose=True)

    output = "\n".join(
        "".join(str(arg) for arg in call.args) for call in mock_print.call_args_list
    ).lower()
    assert "transpilation error" in output


def test_repl_skips_empty_input(monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = iter(["   ", "quit"])  # First input is whitespace only
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    with patch("builtins.print") as mock_print:
        start_repl(verbose=True)

    output = "\n".join(
        "".join(str(arg) for arg in call.args) for call in mock_print.call_args_list
    ).lower()
    assert "exiting limit repl" in output


def test_parse_expr_entrypoint_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyLexer:
        def __init__(self, _: Any) -> None:
            self.tokens = iter(
                [Token("DUMMY", "dummy", 1, 1), Token("EOF", "EOF", 1, 2)]
            )

        def next_token(self) -> Token:
            return next(self.tokens)

    class ParserWithDoubleFailure:
        def __init__(self, _: Any) -> None:
            self.position = 0

        def parse(self) -> list[Any]:
            raise SyntaxError("Statement failed")

        def parse_expr_entrypoint(self) -> None:
            raise ValueError("Expr fallback also failed")

    class DummyTranspiler:
        def __init__(self, _: Any) -> None:
            self.emitter = type(
                "E", (), {"get_output": lambda self: "", "_visit": lambda self, _: None}
            )()

    inputs = iter(["noop", "quit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    monkeypatch.setattr("limit.limit_repl.Lexer", lambda _: DummyLexer(None))
    monkeypatch.setattr(
        "limit.limit_repl.Parser", lambda _: ParserWithDoubleFailure(None)
    )
    monkeypatch.setattr("limit.limit_repl.Transpiler", DummyTranspiler)

    with patch("builtins.print") as mock_print:
        start_repl(verbose=True)

    output = "\n".join(
        " ".join(map(str, call.args)) for call in mock_print.call_args_list
    ).lower()
    assert "[error] >>>" in output
    assert "statement failed" in output


def test_repl_skips_empty_token_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    class EmptyLexer:
        def __init__(self, _: Any) -> None:
            self.called = False

        def next_token(self) -> Token:
            if not self.called:
                self.called = True
                return Token("EOF", "EOF", 1, 1)
            raise StopIteration()

    inputs = iter(["x", "quit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    monkeypatch.setattr("limit.limit_repl.Lexer", lambda _: EmptyLexer(None))
    monkeypatch.setattr("limit.limit_repl.Parser", lambda _: None)
    monkeypatch.setattr("limit.limit_repl.Transpiler", lambda _: None)

    with patch("builtins.print") as mock_print:
        start_repl(verbose=True)

    output = "\n".join(
        " ".join(map(str, call.args)) for call in mock_print.call_args_list
    ).lower()
    assert "exiting limit repl" in output


def test_repl_empty_expr_list(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyLexer:
        def __init__(self, _: Any) -> None:
            self.tokens = iter(
                [
                    Token("PRINT", "!", 1, 1),
                    Token("EOF", "EOF", 1, 2),
                ]
            )

        def next_token(self) -> Token:
            return next(self.tokens)

    class DummyParser:
        def __init__(self, _: Any) -> None:
            self.position = 0

        def parse(self) -> list[Any]:
            raise SyntaxError("fail")

        def parse_expr_entrypoint(self) -> list[Any]:
            return []

    class DummyTranspiler:
        def __init__(self, _: Any) -> None:
            self.emitter = type(
                "E", (), {"get_output": lambda self: "", "_visit": lambda self, _: None}
            )()

    inputs = iter(["noop", "quit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    monkeypatch.setattr("limit.limit_repl.Lexer", lambda _: DummyLexer(None))
    monkeypatch.setattr("limit.limit_repl.Parser", lambda _: DummyParser(None))
    monkeypatch.setattr("limit.limit_repl.Transpiler", DummyTranspiler)

    with patch("builtins.print") as mock_print:
        start_repl(verbose=True)
    output = "\n".join(
        " ".join(str(arg) for arg in call.args) for call in mock_print.call_args_list
    ).lower()
    assert "[error] >>>" not in output  # ensures fallback executed cleanly
    assert "exiting limit repl" in output


def test_repl_expr_node_exec_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class ASTWithExpr:
        def __init__(self) -> None:
            self.kind = "arith"
            self.value = "+"
            self.children: list[Any] = []

    class DummyLexer:
        def __init__(self, _: Any) -> None:
            self.tokens = iter(
                [
                    Token("FAKE", "noop", 1, 1),
                    Token("EOF", "EOF", 1, 2),
                ]
            )

        def next_token(self) -> Token:
            return next(self.tokens)

    class DummyParser:
        def __init__(self, _: Any) -> None:
            pass

        def parse(self) -> list[Any]:
            return [ASTWithExpr()]

    class BadEmitter:
        def emit_expr(self, _: Any) -> str:
            return "1 + 'x'"  # will cause TypeError at runtime

        def get_output(self) -> str:
            return ""

        def _visit(self, _: Any) -> None:
            pass

    class DummyTranspiler:
        def __init__(self, _: Any) -> None:
            self.emitter = BadEmitter()

    inputs = iter(["noop", "quit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    monkeypatch.setattr("limit.limit_repl.Lexer", lambda _: DummyLexer(None))
    monkeypatch.setattr("limit.limit_repl.Parser", lambda _: DummyParser(None))
    monkeypatch.setattr("limit.limit_repl.Transpiler", lambda _: DummyTranspiler(None))

    with patch("builtins.print") as mock_print:
        start_repl(verbose=True)

    output = "\n".join(
        " ".join(str(arg) for arg in call.args) for call in mock_print.call_args_list
    ).lower()
    assert "[error] >>>" in output
    assert "typeerror" in output or "unsupported operand" in output


def test_repl_skips_tokenless_input(monkeypatch: pytest.MonkeyPatch) -> None:
    class EmptyLexer:
        def __init__(self, _: Any) -> None:
            self.done = False

        def next_token(self) -> Token:
            if not self.done:
                self.done = True
                return Token("EOF", "EOF", 1, 1)
            raise StopIteration()

    inputs = iter(["noop", "quit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    monkeypatch.setattr("limit.limit_repl.Lexer", lambda _: EmptyLexer(None))
    monkeypatch.setattr("limit.limit_repl.Parser", lambda _: None)
    monkeypatch.setattr("limit.limit_repl.Transpiler", lambda _: None)

    with patch("builtins.print") as mock_print:
        start_repl(verbose=True)

    output = "\n".join(
        "".join(str(arg) for arg in call.args) for call in mock_print.call_args_list
    ).lower()
    assert "exiting" in output


def test_repl_exec_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeAST:
        def __init__(self) -> None:
            self.kind = "arith"
            self.value = ("+",)
            self.children: list[Any] = []

    class BadEmitter:
        def emit_expr(self, _: Any) -> str:
            return "raise Exception('fail')"

        def get_output(self) -> str:
            return ""

        def _visit(self, _: Any) -> None:
            pass

    class BadTranspiler:
        def __init__(self, _: Any) -> None:
            self.emitter = BadEmitter()

    class DummyLexer:
        def __init__(self, _: Any) -> None:
            self.tokens = iter([Token("DUMMY", "x", 1, 1), Token("EOF", "EOF", 1, 2)])

        def next_token(self) -> Token:
            return next(self.tokens)

    class DummyParser:
        def __init__(self, _: Any) -> None:
            pass

        def parse(self) -> list[Any]:
            return [FakeAST()]

    inputs = iter(["noop", "quit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    monkeypatch.setattr("limit.limit_repl.Lexer", lambda _: DummyLexer(None))
    monkeypatch.setattr("limit.limit_repl.Parser", lambda _: DummyParser(None))
    monkeypatch.setattr("limit.limit_repl.Transpiler", lambda _: BadTranspiler(None))

    with patch("builtins.print") as mock_print:
        start_repl(verbose=True)

    out = "\n".join(
        "".join(str(a) for a in call.args) for call in mock_print.call_args_list
    ).lower()
    assert "[error]" in out and "fail" in out


def test_emit_expr_printed(monkeypatch: pytest.MonkeyPatch) -> None:
    class ASTWithExpr:
        def __init__(self) -> None:
            self.kind = "arith"
            self.value = "+"
            self.children: list[Any] = []

    class DummyLexer:
        def __init__(self, _: Any) -> None:
            self.tokens = iter([Token("FAKE", "noop", 1, 1), Token("EOF", "EOF", 1, 2)])

        def next_token(self) -> Token:
            return next(self.tokens)

    class DummyParser:
        def __init__(self, _: Any) -> None:
            pass

        def parse(self) -> list[Any]:
            return [ASTWithExpr()]

    class DummyEmitter:
        def emit_expr(self, _: Any) -> str:
            return "1 + 2"

        def get_output(self) -> str:
            return ""

        def _visit(self, _: Any) -> None:
            pass

    class DummyTranspiler:
        def __init__(self, _: Any) -> None:
            self.emitter = DummyEmitter()

    inputs = iter(["noop", "quit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    monkeypatch.setattr("limit.limit_repl.Lexer", lambda _: DummyLexer(None))
    monkeypatch.setattr("limit.limit_repl.Parser", lambda _: DummyParser(None))
    monkeypatch.setattr("limit.limit_repl.Transpiler", lambda _: DummyTranspiler(None))

    with patch("builtins.print") as mock_print:
        start_repl(verbose=True)

    printed = "\n".join(
        "".join(str(arg) for arg in call.args).lower()
        for call in mock_print.call_args_list
    )
    assert "[py-expr] >>>" in printed
    assert "1 + 2" in printed


def test_repl_as_script_runs() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"  # <- add the path where `limit` lives
    result = subprocess.run(
        ["python", "src/limit/limit_repl.py"],
        input="quit\n",
        text=True,
        capture_output=True,
        env=env,
    )
    assert "Limit REPL" in result.stdout
    assert "Exiting Limit REPL" in result.stdout


def test_repl_top_level_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    class BrokenLexer:
        def __init__(self, _: Any) -> None:
            pass

        def next_token(self) -> Token:
            raise Exception("top-level failure")

    inputs = iter(["trigger", "quit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    monkeypatch.setattr("limit.limit_repl.Lexer", lambda _: BrokenLexer(None))

    with patch("builtins.print") as mock_print:
        start_repl(verbose=True)

    output = "\n".join(
        "".join(str(arg) for arg in call.args) for call in mock_print.call_args_list
    ).lower()
    assert "[error] >>>" in output
    assert "top-level failure" in output
    assert "exiting limit repl" in output


def test_repl_valid_transpile_non_python(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyLexer:
        def __init__(self, _: Any) -> None:
            self.tokens = iter(
                [Token("DUMMY", "noop", 1, 1), Token("EOF", "EOF", 1, 2)]
            )

        def next_token(self) -> Token:
            return next(self.tokens)

    class DummyAST:
        def __init__(self) -> None:
            self.kind = "arith"
            self.value = "+"
            self.children: list[Any] = []

    class DummyEmitter:
        def __init__(self) -> None:
            self.lines: list[Any] = []

        def emit_expr(self, _: Any) -> str:
            return "x + y"

        def get_output(self) -> str:
            return "x + y"

        def _visit(self, _: Any) -> None:
            self.lines.append("x + y")

    class DummyTranspiler:
        def __init__(self, _: Any) -> None:
            self.emitter = DummyEmitter()

    class DummyParser:
        def __init__(self, _: Any) -> None:
            pass

        def parse(self) -> list[Any]:
            return [DummyAST()]

    inputs = iter(["noop", "quit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    monkeypatch.setattr("limit.limit_repl.Lexer", lambda _: DummyLexer(None))
    monkeypatch.setattr("limit.limit_repl.Parser", lambda _: DummyParser(None))
    monkeypatch.setattr("limit.limit_repl.Transpiler", lambda _: DummyTranspiler(None))

    with patch("builtins.print") as mock_print:
        start_repl(verbose=True, target="js")

    output = "\n".join(
        "".join(str(arg) for arg in call.args) for call in mock_print.call_args_list
    ).lower()
    assert "[info] transpilation complete" in output
    assert "not supported for: js" in output


def test_repl_transpile_execution_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyLexer:
        def __init__(self, _: Any) -> None:
            self.tokens = iter(
                [
                    Token("PRINT", "!", 1, 1),
                    Token("STRING", "hi", 1, 3),
                    Token("EOF", "EOF", 1, 10),
                ]
            )

        def next_token(self) -> Token:
            return next(self.tokens)

    class DummyAST:
        def __init__(self) -> None:
            self.kind = "assign"
            self.value = "x"
            self.children: list[Any] = []

    class DummyParser:
        def __init__(self, _: Any) -> None:
            pass

        def parse(self) -> list[Any]:
            return [DummyAST()]

    class BrokenEmitter:
        def __init__(self) -> None:
            self.lines: list[Any] = []

        def emit_expr(self, _: Any) -> str:
            return "noop()"  # Won't be used

        def get_output(self) -> str:
            return "noop()"

        def _visit(self, _: Any) -> None:
            raise RuntimeError("boom!")  # This is what we want to trigger

    class DummyTranspiler:
        def __init__(self, _: Any) -> None:
            self.emitter = BrokenEmitter()

    inputs = iter(['! "hi"', "quit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    monkeypatch.setattr("limit.limit_repl.Lexer", lambda _: DummyLexer(None))
    monkeypatch.setattr("limit.limit_repl.Parser", lambda _: DummyParser(None))
    monkeypatch.setattr("limit.limit_repl.Transpiler", lambda _: DummyTranspiler(None))

    with patch("builtins.print") as mock_print:
        start_repl(target="js", verbose=True)

    output = "\n".join(
        "".join(str(arg) for arg in call.args) for call in mock_print.call_args_list
    ).lower()
    assert "[error] >>>" in output
    assert "boom" in output
    assert "exiting limit repl" in output


def test_non_sugar_command_returns_false() -> None:
    assert handle_sugar_command("not a sugar command") is False


def test_empty_sugar_command_shows_report(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: None)
    assert handle_sugar_command("SUGAR") is True


def test_valid_sugar_command(monkeypatch: pytest.MonkeyPatch) -> None:
    from limit.limit_repl import uimap as real_uimap  # never shadow

    output: list[Any] = []
    monkeypatch.setattr("builtins.print", output.append)
    assert handle_sugar_command("SUGAR {'eq': 'ASSIGN'}") is True
    token = real_uimap.get_token("eq", 0, 0)
    assert token is not None and token.type == "ASSIGN"


def test_tuple_key_sugar(monkeypatch: pytest.MonkeyPatch) -> None:
    output: list[Any] = []
    monkeypatch.setattr("builtins.print", output.append)
    assert handle_sugar_command("SUGAR {['plus', 'add']: 'PLUS'}") is True
    assert uimap.get_token("plus", 0, 0).type == "PLUS"  # type: ignore
    assert uimap.get_token("add", 0, 0).type == "PLUS"  # type: ignore


def test_invalid_sugar_command(monkeypatch: pytest.MonkeyPatch) -> None:
    output: list[Any] = []
    monkeypatch.setattr("builtins.print", output.append)
    assert handle_sugar_command("SUGAR {invalid_syntax") is True
    assert any("Failed to configure sugar aliases" in str(line) for line in output)


def test_handle_sugar_command_intercepts(monkeypatch: pytest.MonkeyPatch) -> None:
    called = []

    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: called.append(args))
    result = handle_sugar_command("SUGAR {'foo': 'PLUS'}")
    assert result is True
    assert uimap.get_token("foo", 0, 0).type == "PLUS"  # type: ignore


def test_module_command_sets_module(monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = iter(["MODULE core\n", "exit"])
    printed = []

    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    monkeypatch.setattr(
        "builtins.print",
        lambda *args, **kwargs: printed.append(" ".join(map(str, args))),
    )
    start_repl()

    assert any("[module] >>> Current module: core" in line for line in printed)


def test_import_command(monkeypatch: pytest.MonkeyPatch) -> None:
    # Create a fake .limit file
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".limit") as f:
        f.write("= x 1")
        f.flush()
        path = f.name

    inputs = iter([f'IMPORT "{path}"\n', "exit"])
    printed = []

    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    monkeypatch.setattr(
        "builtins.print",
        lambda *args, **kwargs: printed.append(" ".join(map(str, args))),
    )
    start_repl()

    os.unlink(path)  # Cleanup
    assert any("[imported] >>>" in line for line in printed)


def test_repl_expr_fallback_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    # This input causes:
    # - parse() to fail with SyntaxError
    # - parse_expr_entrypoint() to raise something else (like ValueError)
    inputs = iter(["+ 'x", "exit"])
    printed = []

    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    monkeypatch.setattr(
        "builtins.print",
        lambda *args, **kwargs: printed.append(" ".join(map(str, args))),
    )
    start_repl()

    assert any("[error]" in line or "Traceback" in line for line in printed)


def test_repl_skips_comment_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = iter(["# this is a comment", "exit"])
    printed = []

    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    monkeypatch.setattr(
        "builtins.print",
        lambda *args, **kwargs: printed.append(" ".join(map(str, args))),
    )

    start_repl()

    # Nothing to assert — just confirm it runs without error and exits cleanly
    assert any("Exiting Limit REPL." in line for line in printed)


def test_repl_sugar_command_triggers_continue(monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = iter(["SUGAR {'sum': 'PLUS'}", "exit"])
    printed = []

    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    monkeypatch.setattr(
        "builtins.print",
        lambda *args, **kwargs: printed.append(" ".join(map(str, args))),
    )

    start_repl()

    # Should show sugar confirmation
    assert any("Sugar aliases updated" in line for line in printed)


def test_repl_import_appends_dot_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    limit_file = tmp_path / "test_file.limit"
    limit_file.write_text("= x 1")

    # Give import command without ".limit"
    inputs = iter([f'IMPORT "{str(limit_file)[:-6]}"', "exit"])
    printed = []

    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    monkeypatch.setattr(
        "builtins.print",
        lambda *args, **kwargs: printed.append(" ".join(map(str, args))),
    )

    start_repl()

    assert any("[imported]" in line for line in printed)


def test_repl_import_skips_duplicate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    limit_file = tmp_path / "dupe_file.limit"
    limit_file.write_text("= x 1")

    path_str = str(limit_file)

    # Import same file twice
    inputs = iter([f'IMPORT "{path_str}"', f'IMPORT "{path_str}"', "exit"])
    printed = []

    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    monkeypatch.setattr(
        "builtins.print",
        lambda *args, **kwargs: printed.append(" ".join(map(str, args))),
    )

    start_repl()

    assert any("[imported]" in line for line in printed)
    assert any("[import skipped]" in line for line in printed)


def test_repl_import_invalid_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bad_file = tmp_path / "bad.limit"
    bad_file.write_text("~~~")  # Invalid symbol that lexer will not recognize

    inputs = iter([f'IMPORT "{str(bad_file)}"', "exit"])
    printed = []

    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    monkeypatch.setattr(
        "builtins.print",
        lambda *args, **kwargs: printed.append(" ".join(map(str, args))),
    )

    start_repl()

    assert any("[import error] >>> Invalid tokens in file." in line for line in printed)


def test_repl_import_emitter_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import limit.limit_repl  # actual usage site

    test_file = tmp_path / "broken.limit"
    test_file.write_text("= x 1")

    class BrokenEmitter:
        def _visit(self, node: ASTNode) -> str | None:
            raise RuntimeError("boom")

        def get_output(self) -> str:
            return "x = 1"

    class BrokenTranspiler:
        def __init__(self, target: str) -> None:
            self.emitter = BrokenEmitter()

    # Patch where it's imported (used), not where it's defined
    monkeypatch.setattr(limit.limit_repl, "Transpiler", BrokenTranspiler)

    inputs = iter([f'IMPORT "{test_file}"', "exit"])
    printed = []

    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    monkeypatch.setattr(
        "builtins.print",
        lambda *args, **kwargs: printed.append(" ".join(map(str, args))),
    )

    limit.limit_repl.start_repl()

    assert any("import error" in line for line in printed)
    return None


def test_repl_skips_empty_parsed_ast(monkeypatch: pytest.MonkeyPatch) -> None:
    from limit import limit_repl
    from limit.limit_parser import Parser

    inputs = iter(["= x 1", "exit"])
    printed = []

    # Patch Parser.parse to return empty list
    monkeypatch.setattr(Parser, "parse", lambda self: [])

    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    monkeypatch.setattr(
        "builtins.print",
        lambda *args, **kwargs: printed.append(" ".join(map(str, args))),
    )

    limit_repl.start_repl()

    # No crash = success; we’re only testing branch coverage
    assert any("Exiting Limit REPL." in line for line in printed)
    return None


def test_repl_non_py_target_exec_block(monkeypatch: pytest.MonkeyPatch) -> None:
    import limit.limit_repl

    # Fake .limit file with valid assign
    inputs = iter(["= x 1", "exit"])
    printed = []

    # Force the Transpiler to return some fake code
    class DummyEmitter:
        def _visit(self, node: ASTNode) -> str | None:
            pass

        def get_output(self) -> str:
            return "x = 1"  # <- non-empty code

    class DummyTranspiler:
        def __init__(self, target: str) -> None:
            self.emitter = DummyEmitter()

    monkeypatch.setattr(limit.limit_repl, "Transpiler", DummyTranspiler)
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    monkeypatch.setattr(
        "builtins.print",
        lambda *args, **kwargs: printed.append(" ".join(map(str, args))),
    )

    limit.limit_repl.start_repl(target="c")

    assert any("[info] transpilation complete" in line for line in printed)
    assert any("Execution not supported for: c" in line for line in printed)


def test_repl_flattens_nested_ast_list(monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = iter(["= x 1", "exit"])
    printed = []

    nested_ast = [[ASTNode("assign", "x", [])], [ASTNode("assign", "y", [])]]

    # Patch parser to return nested AST
    monkeypatch.setattr(Parser, "parse", lambda self: nested_ast)

    # Transpiler that collects visited nodes
    class DummyEmitter:
        def __init__(self) -> None:
            self.lines: list[Any] = []

        def _visit(self, node: ASTNode) -> str | None:
            self.lines.append(f"visited {node.value}")
            return None

        def get_output(self) -> str:
            return "\n".join(self.lines)

    class DummyTranspiler:
        def __init__(self, target: str) -> None:
            self.emitter = DummyEmitter()

    monkeypatch.setattr(limit.limit_repl, "Transpiler", DummyTranspiler)
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    monkeypatch.setattr(
        "builtins.print",
        lambda *args, **kwargs: printed.append(" ".join(map(str, args))),
    )

    start_repl()

    # Check if both nested nodes were visited
    assert any("visited x" in line for line in printed)
    assert any("visited y" in line for line in printed)


def test_repl_parsed_ast_is_single_node(monkeypatch: pytest.MonkeyPatch) -> None:
    import limit.limit_repl
    from limit.limit_ast import ASTNode
    from limit.limit_parser import Parser

    inputs = iter(["= x 1", "exit"])
    printed = []

    # Return a single ASTNode (not a list)
    monkeypatch.setattr(Parser, "parse", lambda self: ASTNode("assign", "x", []))

    class DummyEmitter:
        def __init__(self) -> None:
            self.lines: list[Any] = []

        def _visit(self, node: ASTNode) -> str | None:
            self.lines.append(f"got {node.value}")
            return None

        def get_output(self) -> str:
            return "\n".join(self.lines)

    class DummyTranspiler:
        def __init__(self, target: str) -> None:
            self.emitter = DummyEmitter()

    monkeypatch.setattr(limit.limit_repl, "Transpiler", DummyTranspiler)
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    monkeypatch.setattr(
        "builtins.print",
        lambda *args, **kwargs: printed.append(" ".join(map(str, args))),
    )

    limit.limit_repl.start_repl()

    assert any("got x" in line for line in printed)


def test_repl_parse_expr_entrypoint_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    import limit.limit_repl
    from limit.limit_parser import Parser

    inputs = iter(["+ 5 6", "exit"])  # Forces expr_entrypoint path
    printed = []

    # Patch to raise during first dispatch
    monkeypatch.setattr(
        Parser,
        "parse_expr_entrypoint",
        lambda self: (_ for _ in ()).throw(RuntimeError("fail")),
    )

    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    monkeypatch.setattr(
        "builtins.print",
        lambda *args, **kwargs: printed.append(" ".join(map(str, args))),
    )

    limit.limit_repl.start_repl()

    # This ensures the top-level `except Exception:` block is hit
    assert any("fail" in line for line in printed)
    assert any("[error]" in line for line in printed)


def test_repl_direct_expr_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    uimap.configure({"+": "PLUS"})  # Ensure '+' is recognized as PLUS
    inputs = iter(["+ 5 6", "exit"])
    printed = []

    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    monkeypatch.setattr(
        "builtins.print",
        lambda *args, **kwargs: printed.append(" ".join(map(str, args))),
    )
    start_repl()

    assert any("11" in line for line in printed)


def test_repl_partial_branches(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    inputs = iter(
        [
            "FUNC noop() { }",  # hits code.strip() == "" branch
            "noop()",  # runs a call that returns None
            "FUNC blank() { RETURN }",  # valid return with None
            "blank()",  # triggers result_val is None
            "exit",  # exit REPL
        ]
    )

    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    from limit import limit_repl

    limit_repl.start_repl()
    captured = capsys.readouterr()

    assert "Limit REPL" in captured.out


def test_repl_empty_code(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    inputs = iter(
        [
            "@ empty() { }",
            "empty()",
            "exit",
        ]
    )
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    from limit import limit_repl

    limit_repl.start_repl()
    captured = capsys.readouterr()
    assert "Limit REPL" in captured.out


def test_repl_flatten_ast_invalid_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    import limit.limit_repl
    from limit.limit_ast import ASTNode
    from limit.limit_lexer import Token

    class DummyParser:
        def __init__(self, _: Any) -> None:
            pass

        def parse(self) -> list[Any]:
            return [[ASTNode("assign", "x", [])], "not_a_list"]

    class DummyLexer:
        def __init__(self, _: Any) -> None:
            self.called = False

        def next_token(self) -> Token:
            if not self.called:
                self.called = True
                return Token("IDENT", "x", 1, 1)
            return Token("EOF", "EOF", 1, 2)

    monkeypatch.setattr("limit.limit_repl.Lexer", DummyLexer)
    monkeypatch.setattr("limit.limit_repl.Parser", DummyParser)

    inputs = iter(["= x 1", "exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    printed: list[str] = []
    monkeypatch.setattr(
        "builtins.print",
        lambda *args, **kwargs: printed.append(" ".join(str(arg) for arg in args)),
    )

    limit.limit_repl.start_repl()

    assert any("Expected list of lists of ASTNode" in line for line in printed)
