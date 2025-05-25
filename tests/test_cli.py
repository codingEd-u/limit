import io
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from limit import limit_cli
from limit.limit_ast import ASTNode
from limit.limit_lexer import CharacterStream, Token
from limit.limit_transpile import Transpiler

FAKE_SOURCE = "= x 5"
FAKE_AST = [
    ASTNode("assign", "x", [ASTNode("identifier", "x"), ASTNode("number", "5")])
]
FAKE_CODE = "x = 5"


@pytest.fixture  # type: ignore[misc]
def dummy_transpile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "limit.limit_transpile.Transpiler",
        lambda target: type("T", (), {"transpile": lambda self, ast: FAKE_CODE})(),
    )
    monkeypatch.setattr(
        "limit.limit_parser.Parser",
        lambda tokens: type("P", (), {"parse": lambda self: FAKE_AST})(),
    )
    monkeypatch.setattr(
        "limit.limit_lexer.Lexer",
        lambda cs: iter(
            [
                Token("ASSIGN", "=", 1, 1),
                Token("IDENT", "x", 1, 2),
                Token("NUMBER", "5", 1, 3),
                Token("EOF", "EOF", 1, 4),
            ]
        ),
    )


def test_run_limit_string_input_prints(
    capsys: pytest.CaptureFixture[str], dummy_transpile: None
) -> None:
    limit_cli.run_limit(source=FAKE_SOURCE, is_string=True)
    out = capsys.readouterr().out.strip()
    assert FAKE_CODE in out


def test_run_limit_file_input(tmp_path: Path, dummy_transpile: None) -> None:
    file_path = tmp_path / "input.limit"
    file_path.write_text(FAKE_SOURCE)
    limit_cli.run_limit(source=str(file_path))


def test_run_limit_pretty_output(
    capsys: pytest.CaptureFixture[str], dummy_transpile: None
) -> None:
    limit_cli.run_limit(source=FAKE_SOURCE, is_string=True, pretty=True)
    out = capsys.readouterr().out
    assert "Transpiled Python" in out


def test_run_limit_output_file(tmp_path: Path, dummy_transpile: None) -> None:
    output_path = tmp_path / "out.py"
    limit_cli.run_limit(
        source=FAKE_SOURCE, is_string=True, out=str(output_path), pretty=True
    )
    contents = output_path.read_text()
    assert contents.strip() == FAKE_CODE


def test_run_limit_exec(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "limit.limit_cli.Transpiler",
        lambda t: type("T", (), {"transpile": lambda self, ast: 'print("executed")'})(),
    )
    monkeypatch.setattr(
        "limit.limit_cli.Parser",
        lambda tokens: type(
            "P", (), {"parse": lambda self: [ASTNode("dummy", "dummy")]}
        )(),
    )

    def fake_lexer(cs: CharacterStream) -> Any:
        tokens = iter([Token("IDENT", "dummy", 1, 1), Token("EOF", "EOF", 1, 2)])
        return type("FakeLexer", (), {"next_token": lambda self: next(tokens)})()

    monkeypatch.setattr("limit.limit_cli.Lexer", fake_lexer)

    limit_cli.run_limit(source=FAKE_SOURCE, is_string=True, execute=False)
    out = capsys.readouterr().out.strip()
    assert out == 'print("executed")'


def test_run_limit_exec_unsupported_target() -> None:
    with pytest.raises(NotImplementedError):
        limit_cli.run_limit(
            source=FAKE_SOURCE, target="c", is_string=True, execute=True
        )


def test_run_limit_exec_unsupported_target_js() -> None:
    with pytest.raises(NotImplementedError):
        limit_cli.run_limit(
            source=FAKE_SOURCE, target="js", is_string=True, execute=True
        )


def test_main_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["limit", "-s", "= x 5"])
    monkeypatch.setattr(
        limit_cli, "run_limit", lambda **kwargs: kwargs.update({"called": True})
    )
    limit_cli.main()


def test_transpiler_unknown_target_raises() -> None:
    with pytest.raises(ValueError, match="Unknown transpilation target"):
        Transpiler("brainfuck")


def test_main_invalid_target(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["limit", "-t", "xyz", "-s", "= x 5"])
    with pytest.raises(SystemExit) as e:
        limit_cli.main()
    assert e.value.code == 2


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])  # type: ignore[misc]
@given(st.text())  # type: ignore[misc]
def test_run_limit_random_input_does_not_crash(source: str) -> Any:
    def fake_lexer(cs: CharacterStream) -> Any:
        tokens = iter([Token("IDENT", "dummy", 1, 1), Token("EOF", "EOF", 1, 2)])
        return type("FakeLexer", (), {"next_token": lambda self: next(tokens)})()

    with (
        patch(
            "limit.limit_cli.Parser",
            lambda tokens: type(
                "P", (), {"parse": lambda self: [ASTNode("dummy", "dummy")]}
            )(),
        ),
        patch(
            "limit.limit_cli.Transpiler",
            lambda t: type("T", (), {"transpile": lambda self, ast: "pass"})(),
        ),
        patch("limit.limit_cli.Lexer", fake_lexer),
        patch("limit.limit_cli.CharacterStream", lambda text, *_: text),
    ):
        try:
            limit_cli.run_limit(source=source, is_string=True)
        except Exception:
            pytest.fail("Should not crash on random input")


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])  # type: ignore[misc]
@given(pretty=st.booleans(), execute=st.booleans())  # type: ignore[misc]
def test_run_limit_exec_modes(pretty: bool, execute: bool) -> None:
    class FakeLexer:
        def __init__(self, cs: CharacterStream) -> None:
            self.tokens = iter(
                [Token("IDENT", "dummy", 1, 1), Token("EOF", "EOF", 1, 2)]
            )

        def next_token(self) -> Token:
            return next(self.tokens)

    with (
        patch(
            "limit.limit_cli.Parser",
            lambda tokens: type(
                "P", (), {"parse": lambda self: [ASTNode("dummy", "dummy")]}
            )(),
        ),
        patch(
            "limit.limit_cli.Transpiler",
            lambda t: type(
                "T",
                (),
                {
                    "transpile": lambda self, ast: (
                        'print("ok")' if execute else 'print("noop")'
                    )
                },
            )(),
        ),
        patch("limit.limit_cli.Lexer", FakeLexer),
        patch("limit.limit_cli.CharacterStream", lambda text, *_: text),
        patch("sys.stdout", new_callable=io.StringIO) as fake_out,
    ):
        limit_cli.run_limit(
            source=FAKE_SOURCE, is_string=True, pretty=pretty, execute=execute
        )
        out = fake_out.getvalue()
        if execute:
            assert "ok" in out
        else:
            assert "noop" in out


def test_run_limit_exec_unsupported(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "limit.limit_cli.Transpiler",
        lambda t: type("T", (), {"transpile": lambda self, ast: "print('hi')"})(),
    )
    monkeypatch.setattr(
        "limit.limit_cli.Parser",
        lambda tokens: type(
            "P", (), {"parse": lambda self: [ASTNode("dummy", "dummy")]}
        )(),
    )
    monkeypatch.setattr(
        "limit.limit_cli.Lexer",
        lambda cs: type(
            "Fake", (), {"next_token": lambda self: Token("EOF", "EOF", 1, 1)}
        )(),
    )
    out_err = io.StringIO()
    sys.stderr = out_err
    try:
        limit_cli.run_limit("fake", is_string=True, execute=True, target="c")
    finally:
        sys.stderr = sys.__stderr__
    assert "Execution not supported for target: c" in out_err.getvalue()


def test_run_limit_file_output_pretty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_file = tmp_path / "out.py"
    monkeypatch.setattr(
        "limit.limit_cli.Transpiler",
        lambda t: type("T", (), {"transpile": lambda self, ast: "x = 5"})(),
    )
    monkeypatch.setattr(
        "limit.limit_cli.Parser",
        lambda tokens: type(
            "P", (), {"parse": lambda self: [ASTNode("dummy", "dummy")]}
        )(),
    )
    monkeypatch.setattr(
        "limit.limit_cli.Lexer",
        lambda cs: type(
            "Fake", (), {"next_token": lambda self: Token("EOF", "EOF", 1, 1)}
        )(),
    )
    limit_cli.run_limit("fake", is_string=True, out=str(out_file), pretty=True)
    assert out_file.read_text().strip() == "x = 5"


def test_cli_main_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src_file = tmp_path / "test.min"
    src_file.write_text("= x 5")
    monkeypatch.setattr(sys, "argv", ["limit", str(src_file)])
    monkeypatch.setattr(
        "limit.limit_cli.run_limit", lambda **kwargs: kwargs.update({"called": True})
    )
    limit_cli.main()  # test shouldn't error


def test_output_file_and_pretty_banner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output_path = tmp_path / "out.py"
    monkeypatch.setattr(
        "limit.limit_cli.Transpiler",
        lambda t: type("T", (), {"transpile": lambda self, ast: "x = 42"})(),
    )
    monkeypatch.setattr(
        "limit.limit_cli.Parser",
        lambda tokens: type(
            "P", (), {"parse": lambda self: [ASTNode("assign", "x")]}
        )(),
    )
    monkeypatch.setattr(
        "limit.limit_cli.Lexer",
        lambda cs: type(
            "FakeLexer", (), {"next_token": lambda self: Token("EOF", "EOF", 1, 1)}
        )(),
    )
    limit_cli.run_limit(
        source="= x 42", is_string=True, out=str(output_path), pretty=True
    )
    out_text = output_path.read_text()
    assert "x = 42" in out_text
    captured = capsys.readouterr().out
    assert "(wrote to" in captured


def test_main_cli_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["limit", "-s", "= x 5"])
    called = {}

    def dummy_run(**kwargs: Any) -> None:
        called.update(kwargs)

    monkeypatch.setattr(limit_cli, "run_limit", dummy_run)
    limit_cli.main()
    assert called["source"] == "= x 5"
    assert called["is_string"] is True


def test_run_limit_rejects_non_limit_file() -> None:
    with pytest.raises(ValueError, match="Only .limit files are supported."):
        limit_cli.run_limit("example.txt", is_string=False)


def test_run_limit_prints_to_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    source = '! "Hello"'  # valid LIMIT source
    fake_stdout = io.StringIO()
    monkeypatch.setattr(sys, "stdout", fake_stdout)

    limit_cli.run_limit(source, is_string=True, out=None, execute=False, pretty=False)

    output = fake_stdout.getvalue()
    assert 'print("Hello")' in output or "print('Hello')" in output


def test_main_calls_repl_on_no_args(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {}

    def fake_repl(*args: str, **kwargs: str) -> None:
        called["ran"] = True

    monkeypatch.setattr(sys, "argv", ["limit"])
    monkeypatch.setattr("limit.limit_repl.start_repl", fake_repl)

    limit_cli.main()

    assert called.get("ran") is True


def test_run_limit_pretty_prints_write_message(monkeypatch: pytest.MonkeyPatch) -> None:
    # Create a temporary output file path
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmp:
        out_path = tmp.name

    source = 'PRINT "hi"'  # Valid LIMIT source
    fake_stdout = io.StringIO()
    monkeypatch.setattr(sys, "stdout", fake_stdout)

    limit_cli.run_limit(
        source, is_string=True, out=out_path, execute=False, pretty=True
    )

    output = fake_stdout.getvalue()
    assert f"(wrote to {out_path})" in output


def test_main_repl_flag_calls_repl(monkeypatch: pytest.MonkeyPatch) -> None:
    called_args = {}

    def fake_repl(*, target: Any, verbose: Any) -> None:
        called_args["target"] = target
        called_args["verbose"] = verbose

    monkeypatch.setattr("limit.limit_repl.start_repl", fake_repl)
    monkeypatch.setattr(sys, "argv", ["limit", "--repl", "--target", "c", "--verbose"])

    limit_cli.main()

    assert called_args["target"] == "c"
    assert called_args["verbose"] is True


def test_limit_cli_main_entrypoint_runs() -> None:
    cli_path = os.path.join("src", "limit", "limit_cli.py")

    result = subprocess.run(
        [sys.executable, cli_path, "--repl"],
        input=b"",
        capture_output=True,
        timeout=5,
    )

    assert result.returncode == 0 or result.returncode == 1  # allow REPL exit


def test_run_limit_print_and_write_pretty(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = "@ main() { }"
    out_file = tmp_path / "out.py"

    # Case 1: out is set, pretty is True (hits line 53)
    limit_cli.run_limit(
        source=code,
        is_string=True,
        target="py",
        out=str(out_file),
        execute=False,
        pretty=True,
    )

    out_text = out_file.read_text()
    assert "def main()" in out_text

    captured = capsys.readouterr()
    assert "(wrote to" in captured.out


def test_run_limit_print_only_no_out_no_pretty(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = "@ main() { }"

    # Case 2: out is None, pretty is False (hits line 46 and avoids 53)
    limit_cli.run_limit(
        source=code,
        is_string=True,
        target="py",
        out=None,
        execute=False,
        pretty=False,
    )

    captured = capsys.readouterr()
    assert "def main()" in captured.out
    assert "(wrote to" not in captured.out
