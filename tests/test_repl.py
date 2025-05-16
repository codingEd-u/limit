import builtins
import os
import subprocess
from unittest.mock import patch

import pytest

from limit.limit_ast import ASTNode
from limit.limit_lexer import Token
from limit.limit_repl import print_traceback, start_repl


@pytest.fixture
def mock_pipeline(monkeypatch):
    class MockLexer:
        def __init__(self, cs):
            self.tokens = iter(
                [
                    Token("PRINT", "!", 1, 1),
                    Token("STRING", "hello", 1, 3),
                    Token("EOF", "EOF", 1, 10),
                ]
            )

        def next_token(self):
            return next(self.tokens)

    class MockParser:
        def __init__(self, tokens):
            pass

        def parse(self):
            return ["FAKE_AST"]

    class MockTranspiler:
        def __init__(self, target):
            self.target = target

        def transpile(self, ast):
            if self.target == "py":
                return 'print("hello")'
            return "noop();"

    monkeypatch.setattr("limit.limit_lexer.Lexer", MockLexer)
    monkeypatch.setattr("limit.limit_parser.Parser", MockParser)
    monkeypatch.setattr("limit.limit_transpile.Transpiler", MockTranspiler)


def test_repl_quit(monkeypatch, capsys):
    monkeypatch.setattr(builtins, "input", lambda _: "quit")
    start_repl()
    out = capsys.readouterr().out
    assert "Exiting Limit REPL" in out


def test_repl_exit(monkeypatch, capsys):
    monkeypatch.setattr(builtins, "input", lambda _: "exit")
    start_repl()
    out = capsys.readouterr().out
    assert "Exiting Limit REPL" in out


def test_repl_empty_input_skipped(monkeypatch, capsys):
    calls = iter(["   ", "quit"])
    monkeypatch.setattr(builtins, "input", lambda _: next(calls))
    start_repl()
    out = capsys.readouterr().out
    assert "Exiting Limit REPL" in out


def test_repl_valid_input_exec(monkeypatch, capsys, mock_pipeline):
    calls = iter(['! "hello"', "quit"])
    monkeypatch.setattr(builtins, "input", lambda _: next(calls))
    start_repl(verbose=True)
    out = capsys.readouterr().out.lower()
    assert "hello" in out


@pytest.mark.usefixtures("mock_pipeline")
def test_repl_valid_input_unsupported_target(monkeypatch, capsys, mock_pipeline):
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


def test_repl_keyboard_interrupt(monkeypatch, capsys):
    monkeypatch.setattr(
        builtins, "input", lambda _: (_ for _ in ()).throw(KeyboardInterrupt())
    )
    start_repl()
    out = capsys.readouterr().out
    assert "Exiting Limit REPL" in out


def test_repl_verbose_mode(monkeypatch, capsys):
    calls = iter(["verbose-mode", "quit"])
    monkeypatch.setattr("builtins.input", lambda _: next(calls))
    start_repl()
    out = capsys.readouterr().out
    assert "verbose mode" in out.lower()


def test_repl_verbose_mode_toggle_and_exit(monkeypatch):
    class DummyLexer:
        def __init__(self, _):
            self.tokens = iter(
                [Token("DUMMY", "dummy", 1, 1), Token("EOF", "EOF", 1, 2)]
            )

        def next_token(self):
            return next(self.tokens)

    class DummyAST:
        def __init__(self):
            self.kind = "arith"
            self.value = "+"
            self.children = []

    class DummyEmitter:
        def emit_expr(self, _):
            return "1 + 1"

        def get_output(self):
            return ""

        def _visit(self, _):
            pass

    class DummyTranspiler:
        def __init__(self, _):
            self.emitter = DummyEmitter()

    class DummyParser:
        def __init__(self, _):
            pass

        def parse(self):
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


def test_is_expression_node_covers_all_kinds():
    from limit.limit_repl import is_expression_node

    assert is_expression_node(ASTNode("arith"))
    assert is_expression_node(ASTNode("bool"))
    assert not is_expression_node(ASTNode("assign"))


def test_repl_entry_banner(monkeypatch):
    inputs = iter(["quit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    with patch("builtins.print") as mock_print:
        start_repl(verbose=False)
    output = "\n".join(
        "".join(str(arg) for arg in call.args) for call in mock_print.call_args_list
    ).lower()
    assert "limit repl" in output


def test_print_traceback_outputs_error():
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


def test_repl_lexer_exception(monkeypatch):
    class ExplodingLexer:
        def __init__(self, stream):
            pass

        def next_token(self):
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


def test_repl_runtime_emit_expr_error(monkeypatch):
    class FakeAST:
        def __init__(self):
            self.kind = "arith"
            self.value = "+"
            self.children = []

    class DummyLexer:
        def __init__(self, _):
            self.tokens = iter(
                [Token("DUMMY", "dummy", 1, 1), Token("EOF", "EOF", 1, 2)]
            )

        def next_token(self):
            return next(self.tokens)

    class BadEmitter:
        def __init__(self):
            self.lines = []

        def emit_expr(self, _):
            return "1 / 0"  # runtime error

        def get_output(self):
            return ""

        def _visit(self, _):
            pass

    class BadTranspiler:
        def __init__(self, _):
            self.emitter = BadEmitter()

    class MockParser:
        def __init__(self, _):
            pass

        def parse(self):
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


def test_repl_fallback_expr_parse(monkeypatch):
    class DummyLexer:
        def __init__(self, _):
            self.tokens = iter(
                [Token("DUMMY", "dummy", 1, 1), Token("EOF", "EOF", 1, 2)]
            )

        def next_token(self):
            return next(self.tokens)

    class MockParser:
        def __init__(self, _):
            self.position = 0

        def parse(self):
            raise SyntaxError("Invalid statement")

        def parse_expr_entrypoint(self):
            return [ASTNode("arith", "+", [])]

    class DummyEmitter:
        def get_output(self):
            return ""

        def _visit(self, _):
            self.lines = []

    class DummyTranspiler:
        def __init__(self, _):
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


def test_repl_invalid_token(monkeypatch):
    class LexerWithError:
        def __init__(self, _):
            self.tokens = iter([Token("ERROR", "$", 1, 1), Token("EOF", "EOF", 1, 2)])

        def next_token(self):
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


def test_repl_transpiler_not_implemented(monkeypatch):
    class DummyLexer:
        def __init__(self, _):
            self.tokens = iter(
                [Token("DUMMY", "dummy", 1, 1), Token("EOF", "EOF", 1, 2)]
            )

        def next_token(self):
            return next(self.tokens)

    class DummyParser:
        def __init__(self, _):
            pass

        def parse(self):
            return ["dummy"]

    class FailingTranspiler:
        def __init__(self, _):
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


def test_repl_skips_empty_input(monkeypatch):
    inputs = iter(["   ", "quit"])  # First input is whitespace only
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    with patch("builtins.print") as mock_print:
        start_repl(verbose=True)

    output = "\n".join(
        "".join(str(arg) for arg in call.args) for call in mock_print.call_args_list
    ).lower()
    assert "exiting limit repl" in output


def test_parse_expr_entrypoint_failure(monkeypatch):
    class DummyLexer:
        def __init__(self, _):
            self.tokens = iter(
                [Token("DUMMY", "dummy", 1, 1), Token("EOF", "EOF", 1, 2)]
            )

        def next_token(self):
            return next(self.tokens)

    class ParserWithDoubleFailure:
        def __init__(self, _):
            self.position = 0

        def parse(self):
            raise SyntaxError("Statement failed")

        def parse_expr_entrypoint(self):
            raise ValueError("Expr fallback also failed")

    class DummyTranspiler:
        def __init__(self, _):
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


def test_repl_skips_empty_token_stream(monkeypatch):
    class EmptyLexer:
        def __init__(self, _):
            self.called = False

        def next_token(self):
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


def test_repl_empty_expr_list(monkeypatch):
    class DummyLexer:
        def __init__(self, _):
            self.tokens = iter(
                [
                    Token("PRINT", "!", 1, 1),
                    Token("EOF", "EOF", 1, 2),
                ]
            )

        def next_token(self):
            return next(self.tokens)

    class DummyParser:
        def __init__(self, _):
            self.position = 0

        def parse(self):
            raise SyntaxError("fail")

        def parse_expr_entrypoint(self):
            return []

    class DummyTranspiler:
        def __init__(self, _):
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


def test_repl_expr_node_exec_failure(monkeypatch):
    class ASTWithExpr:
        def __init__(self):
            self.kind = "arith"
            self.value = "+"
            self.children = []

    class DummyLexer:
        def __init__(self, _):
            self.tokens = iter(
                [
                    Token("FAKE", "noop", 1, 1),
                    Token("EOF", "EOF", 1, 2),
                ]
            )

        def next_token(self):
            return next(self.tokens)

    class DummyParser:
        def __init__(self, _):
            pass

        def parse(self):
            return [ASTWithExpr()]

    class BadEmitter:
        def emit_expr(self, _):
            return "1 + 'x'"  # will cause TypeError at runtime

        def get_output(self):
            return ""

        def _visit(self, _):
            pass

    class DummyTranspiler:
        def __init__(self, _):
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


def test_repl_skips_tokenless_input(monkeypatch):
    class EmptyLexer:
        def __init__(self, _):
            self.done = False

        def next_token(self):
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


def test_repl_exec_failure(monkeypatch):
    class FakeAST:
        def __init__(self):
            self.kind, self.value, self.children = "arith", "+", []

    class BadEmitter:
        def emit_expr(self, _):
            return "raise Exception('fail')"

        def get_output(self):
            return ""

        def _visit(self, _):
            pass

    class BadTranspiler:
        def __init__(self, _):
            self.emitter = BadEmitter()

    class DummyLexer:
        def __init__(self, _):
            self.tokens = iter([Token("DUMMY", "x", 1, 1), Token("EOF", "EOF", 1, 2)])

        def next_token(self):
            return next(self.tokens)

    class DummyParser:
        def __init__(self, _):
            pass

        def parse(self):
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


def test_emit_expr_printed(monkeypatch):
    class ASTWithExpr:
        def __init__(self):
            self.kind = "arith"
            self.value = "+"
            self.children = []

    class DummyLexer:
        def __init__(self, _):
            self.tokens = iter([Token("FAKE", "noop", 1, 1), Token("EOF", "EOF", 1, 2)])

        def next_token(self):
            return next(self.tokens)

    class DummyParser:
        def __init__(self, _):
            pass

        def parse(self):
            return [ASTWithExpr()]

    class DummyEmitter:
        def emit_expr(self, _):
            return "1 + 2"

        def get_output(self):
            return ""

        def _visit(self, _):
            pass

    class DummyTranspiler:
        def __init__(self, _):
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


def test_repl_as_script_runs():
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


def test_repl_top_level_exception(monkeypatch):
    class BrokenLexer:
        def __init__(self, _):
            pass

        def next_token(self):
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


def test_repl_valid_transpile_non_python(monkeypatch):
    class DummyLexer:
        def __init__(self, _):
            self.tokens = iter(
                [Token("DUMMY", "noop", 1, 1), Token("EOF", "EOF", 1, 2)]
            )

        def next_token(self):
            return next(self.tokens)

    class DummyAST:
        def __init__(self):
            self.kind = "arith"
            self.value = "+"
            self.children = []

    class DummyEmitter:
        def __init__(self):
            self.lines = []

        def emit_expr(self, _):
            return "x + y"

        def get_output(self):
            return "x + y"

        def _visit(self, _):
            self.lines.append("x + y")

    class DummyTranspiler:
        def __init__(self, _):
            self.emitter = DummyEmitter()

    class DummyParser:
        def __init__(self, _):
            pass

        def parse(self):
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


def test_repl_transpile_execution_exception(monkeypatch):
    class DummyLexer:
        def __init__(self, _):
            self.tokens = iter(
                [
                    Token("PRINT", "!", 1, 1),
                    Token("STRING", "hi", 1, 3),
                    Token("EOF", "EOF", 1, 10),
                ]
            )

        def next_token(self):
            return next(self.tokens)

    class DummyAST:
        def __init__(self):
            self.kind = "assign"
            self.value = "x"
            self.children = []

    class DummyParser:
        def __init__(self, _):
            pass

        def parse(self):
            return [DummyAST()]

    class BrokenEmitter:
        def __init__(self):
            self.lines = []

        def emit_expr(self, _):
            return "noop()"  # Won't be used

        def get_output(self):
            return "noop()"

        def _visit(self, _):
            raise RuntimeError("boom!")  # This is what we want to trigger

    class DummyTranspiler:
        def __init__(self, _):
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
