import argparse
import io
import sys

from limit.limit_lexer import CharacterStream, Lexer
from limit.limit_parser import Parser
from limit.limit_transpile import Transpiler


def run_limit(
    source: str,
    is_string: bool = False,
    target: str = "py",
    out: str | None = None,
    execute: bool = False,
    pretty: bool = False,
) -> None:
    if not is_string and not source.endswith(".limit"):
        raise ValueError("Only .limit files are supported.")
    # 1. Read source
    if not is_string:
        with open(source, encoding="utf-8") as f:
            source = f.read()

    # 2. Lexing
    cs = CharacterStream(source, 0, 1, 1)
    lexer = Lexer(cs)
    tokens = []
    while True:
        tok = lexer.next_token()
        if tok.type == "EOF":
            break
        tokens.append(tok)

    # 3. Parsing
    ast = Parser(tokens).parse()

    # 4. Transpiling
    transpiler = Transpiler(target)
    code = transpiler.transpile(ast)

    # 5. Output result
    if pretty and target == "py":
        banner = "=" * 20
        print(f"{banner}\nTranspiled Python\n{banner}\n{code}\n{banner}\n")
    elif not out:
        print(code)
    else:
        pass  # pragma: no cover

    # 6. Optional write to file
    if out:
        with open(out, "w", encoding="utf-8") as f:
            f.write(code)
        if pretty:
            print(f"(wrote to {out})")
        else:
            pass  # pragma: no cover

    # 7. Optional execution
    if execute and target == "py":
        buf = io.StringIO()
        old_stdout = sys.stdout
        try:
            sys.stdout = buf
            exec(code, globals(), locals())  # nosec B102
        finally:
            sys.stdout = old_stdout
        if pretty:
            print("<<< OUTPUT >>>")
        print(buf.getvalue().rstrip())
    elif execute:
        print(f"Execution not supported for target: {target}", file=sys.stderr)


def main() -> None:
    if len(sys.argv) == 1:
        # No args passed: open REPL instead
        from limit.limit_repl import start_repl

        start_repl()
        return
    parser = argparse.ArgumentParser(prog="limit")
    parser.add_argument("source", nargs="?", help="Filename or raw source (with -s)")
    parser.add_argument(
        "-s", "--string", action="store_true", help="Interpret source as literal string"
    )
    parser.add_argument(
        "-t",
        "--target",
        choices=("py", "c"),
        default="py",
        help="Transpile target (default: py)",
    )
    parser.add_argument("-o", "--out", metavar="OUTFILE", help="Output to file")
    parser.add_argument(
        "-e",
        "--exec",
        dest="execute",
        action="store_true",
        help="Exec transpiled Python code",
    )
    parser.add_argument(
        "-p", "--pretty", action="store_true", help="Show code/output with banners"
    )
    parser.add_argument(
        "--repl",
        action="store_true",
        help="Launch interactive REPL instead of transpiling",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Verbose REPL mode (if --repl)"
    )

    args = parser.parse_args()

    if args.repl or args.source is None:
        from limit.limit_repl import start_repl

        start_repl(target=args.target, verbose=args.verbose)
    else:
        run_limit(
            source=args.source,
            is_string=args.string,
            target=args.target,
            out=args.out,
            execute=args.execute,
            pretty=args.pretty,
        )


if __name__ == "__main__" and not any("pytest" in arg for arg in sys.argv):
    main()
