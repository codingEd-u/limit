import io

# import os
# import json
import re
import traceback

# from collections import Counter, defaultdict
from typing import Any

from limit.limit_ast import ASTNode
from limit.limit_lexer import CharacterStream, Lexer, Token
from limit.limit_parser import Parser
from limit.limit_transpile import Transpiler
from limit.limit_uimap import UserInterfaceMapper

uimap = UserInterfaceMapper.from_canonical()
# sugar_history: list[str] = []
# locked_aliases: set[str] = set()
# undo_stack: list[dict[str, str]] = []
# redo_stack: list[dict[str, str]] = []
# snapshots: dict[str, tuple[dict[str, str], dict[str, str]]] = {}


def is_expression_node(node: ASTNode) -> bool:
    is_base_expr = node.kind in {
        "arith",
        "bool",
        "identifier",
        "call",
        "member",
        "float",
        "number",
        "string",
        "new",
    }

    is_expr_stmt = (
        node.kind == "expr_stmt"
        and bool(node.children)
        and node.children[0].kind
        in {"call", "member", "identifier", "arith", "bool", "compare", "new"}
    )

    return (is_base_expr or is_expr_stmt) and node.kind != "print"


def print_traceback() -> None:
    buf = io.StringIO()
    traceback.print_exc(file=buf)
    print("[error] >>>")
    print(buf.getvalue())


def handle_sugar_command(src: str) -> bool:
    src = src.strip()
    if not src.upper().startswith("SUGAR"):
        return False
    command = src[5:].strip()
    if command == "":
        print(uimap.report(verbose=True))
        return True
    try:
        command = re.sub(r"\[\s*(.*?)\s*\]", lambda m: "(" + m.group(1) + ",)", command)
        raw_map = eval(
            command, {}, {}
        )  # Unsafe in prod, safe here due to controlled context
        final_mapping: dict[str, str] = {}
        for key, sym in raw_map.items():
            if isinstance(key, (list, tuple, set)):
                for alias in key:
                    final_mapping[str(alias)] = sym
            else:
                final_mapping[str(key)] = sym
        uimap.configure(final_mapping)
        print("[ok] >>> Sugar aliases updated.")
        print(
            "\n".join(
                f"{alias:>12} → {sym}" for alias, sym in sorted(final_mapping.items())
            )
        )
    except Exception as e:
        print("[error] >>> Failed to configure sugar aliases:")
        print(e)
    return True


# V2
# def handle_sugar_command(src: str) -> bool:
#     src = src.strip()

#     if not src.upper().startswith("SUGAR"):
#         return False

#     command = src[5:].strip()

#     if command.upper().startswith("LOAD "):
#         path = command[5:].strip().strip('"').strip("'")
#         if not os.path.exists(path):
#             print(f"[error] >>> File not found: {path}")
#             return True
#         try:
#             uimap.load_from_json(path)
#             print("[ok] >>> Sugar aliases loaded from file.")
#             print(uimap.report(verbose=True))
#         except MappingError as e:
#             print(f"[error] >>> Failed to load sugar aliases: {e}")
#             if e.conflicts:
#                 for conflict in e.conflicts:
#                     print(" -", conflict)
#         return True

#     if command.upper().startswith("FROM ENV"):
#         path = os.getenv("SUGAR_PATH")
#         if not path:
#             print("[env] >>> SUGAR_PATH not set.")
#             return True
#         if not os.path.exists(path):
#             print(f"[env] >>> No file found at: {path}")
#             return True
#         try:
#             uimap.load_from_json(path)
#             print(f"[ok] >>> Loaded sugar from env: {path}")
#         except Exception as e:
#             print(f"[error] >>> Failed to load from SUGAR_PATH: {e}")
#         return True

#     if command.upper().startswith("SAVE "):
#         path = command[5:].strip().strip('"').strip("'")
#         try:
#             with open(path, "w", encoding="utf-8") as f:
#                 json.dump(uimap.summary(), f, indent=2)
#             print(f"[ok] >>> Sugar aliases saved to: {path}")
#         except Exception as e:
#             print(f"[error] >>> Failed to save sugar aliases: {e}")
#         return True

#     if command.upper().startswith("EXPORT "):
#         parts = command[7:].strip().split(maxsplit=1)
#         if not parts:
#             print("[error] >>> Usage: SUGAR EXPORT \"TOKEN\" TO \"file.json\"")
#             return True
#         token = parts[0].strip('"').strip("'")
#         path = "exported_aliases.json"
#         if len(parts) == 2 and parts[1].upper().startswith("TO "):
#             path = parts[1][3:].strip().strip('"').strip("'")
#         try:
#             filtered = {k: v for k, v in uimap.summary().items() if v == token}
#             with open(path, "w", encoding="utf-8") as f:
#                 json.dump(filtered, f, indent=2)
#             print(f"[ok] >>> Exported {len(filtered)} aliases for '{token}' to {path}")
#         except Exception as e:
#             print(f"[error] >>> Failed to export aliases: {e}")
#         return True

#     if command.upper() == "LEN":
#         print(f"[count] >>> Total sugar aliases: {len(uimap.summary())}")
#         return True

#     if command.upper() == "RESET":
#         uimap.configure({})
#         locked_aliases.clear()
#         sugar_history.clear()
#         print("[ok] >>> Sugar aliases reset.")
#         return True

#     if command.upper().startswith("PAGE"):
#         try:
#             _, page_str = command.split(maxsplit=1)
#             page = int(page_str.strip())
#         except Exception:
#             page = 1
#         all_aliases = uimap.summary()
#         aliases = sorted(all_aliases.items())
#         per_page = 20
#         start = (page - 1) * per_page
#         end = start + per_page
#         slice = aliases[start:end]
#         if not slice:
#             print(f"[page] >>> No aliases found on page {page}")
#         else:
#             print(f"[page] >>> Sugar aliases (Page {page}):")
#             for alias, sym in slice:
#                 print(f"{alias:>12} → {sym}")
#         return True

#     if command.upper().startswith("SEARCH "):
#         term = command[7:].strip().strip('"').strip("'").lower()
#         matches = [(a, s) for a, s in uimap.summary().items() if term in a.lower() or term in s.lower()]
#         if not matches:
#             print(f"[search] >>> No matches found for: {term}")
#         else:
#             print(f"[search] >>> Matches for '{term}':")
#             for alias, sym in sorted(matches):
#                 print(f"{alias:>12} → {sym}")
#         return True

#     if command.upper().startswith("SHOW "):
#         token = command[5:].strip().strip('"').strip("'")
#         matches = [(a, s) for a, s in uimap.summary().items() if s == token]
#         if not matches:
#             print(f"[show] >>> No aliases found for token: {token}")
#         else:
#             print(f"[show] >>> Aliases for token '{token}':")
#             for alias, sym in sorted(matches):
#                 desc = uimap.descriptions.get(alias)
#                 if desc:
#                     print(f"{alias:>12} → {sym}     # {desc}")
#                 else:
#                     print(f"{alias:>12} → {sym}")
#         return True

#     if command.upper().startswith("CLEAR "):
#         token = command[6:].strip().strip('"').strip("'")
#         current = uimap.summary()
#         new_map = {k: v for k, v in current.items() if v != token or k in locked_aliases}
#         removed = len(current) - len(new_map)
#         uimap.configure(new_map)
#         print(f"[clear] >>> Removed {removed} aliases mapped to token '{token}'")
#         return True

#     if command.upper().startswith("LOCK "):
#         alias = command[5:].strip().strip('"').strip("'")
#         if alias not in uimap.summary():
#             print(f"[lock] >>> Alias not found: {alias}")
#         else:
#             locked_aliases.add(alias)
#             print(f"[lock] >>> Alias locked: {alias}")
#         return True

#     if command.upper().startswith("RENAME "):
#         try:
#             parts = re.split(r'\s+TO\s+', command[7:], maxsplit=1)
#             if len(parts) != 2:
#                 raise ValueError
#             old, new = parts[0].strip().strip('"').strip("'"), parts[1].strip().strip('"').strip("'")
#             if old in locked_aliases:
#                 print(f"[rename] >>> Cannot rename locked alias: {old}")
#                 return True
#             mapping = uimap.summary()
#             if old not in mapping:
#                 print(f"[rename] >>> Alias not found: {old}")
#                 return True
#             if new in mapping:
#                 print(f"[rename] >>> Target alias already exists: {new}")
#                 return True
#             mapping[new] = mapping.pop(old)
#             uimap.configure(mapping)
#             print(f"[rename] >>> {old} → {new}")
#         except Exception:
#             print(f"[error] >>> Usage: SUGAR RENAME \"old\" TO \"new\"")
#         return True

#     if command.upper() == "HISTORY":
#         if not sugar_history:
#             print("[history] >>> No sugar entries this session.")
#         else:
#             print("[history] >>> Inline sugar mappings this session:")
#             for entry in sugar_history[-10:]:
#                 print(" ", entry)
#         return True

#     if command.upper() == "DIFF":
#         recent = uimap.session_diff()
#         if not recent:
#             print("[diff] >>> No new sugar aliases added this session.")
#         else:
#             print("[diff] >>> Session-added sugar aliases:")
#             for alias, sym in sorted(recent.items()):
#                 print(f"{alias:>12} → {sym}")
#         return True

#     if command.upper() == "LOCK ALL":
#         locked_aliases.update(uimap.summary().keys())
#         print(f"[lock] >>> Locked all {len(locked_aliases)} aliases.")
#         return True

#     if command.upper().startswith("LOCK TOKEN "):
#         token = command[11:].strip().strip('"').strip("'")
#         added = 0
#         for alias, sym in uimap.summary().items():
#             if sym == token:
#                 locked_aliases.add(alias)
#                 added += 1
#         print(f"[lock] >>> Locked {added} aliases for token '{token}'")
#         return True

#     if command.upper() == "UNDO":
#         current = uimap.summary()
#         if not undo_stack:
#             print("[undo] >>> Nothing to undo.")
#             return True
#         redo_stack.append(current.copy())
#         last_state = undo_stack.pop()
#         uimap.configure(last_state)
#         print("[undo] >>> Reverted to previous alias state.")
#         return True

#     if command.upper() == "REDO":
#         if not redo_stack:
#             print("[redo] >>> Nothing to redo.")
#             return True
#         undo_stack.append(uimap.summary().copy())
#         next_state = redo_stack.pop()
#         uimap.configure(next_state)
#         print("[redo] >>> Redid last alias change.")
#         return True

#     if command.upper().startswith("DESCRIBE "):
#         try:
#             parts = re.split(r'\s+AS\s+', command[9:], maxsplit=1)
#             if len(parts) != 2:
#                 raise ValueError
#             alias = parts[0].strip().strip('"').strip("'")
#             note = parts[1].strip().strip('"').strip("'")
#             if alias not in uimap.summary():
#                 print(f"[describe] >>> Alias not found: {alias}")
#             else:
#                 uimap.descriptions[alias] = note
#                 print(f"[describe] >>> {alias} → {note}")
#         except Exception:
#             print(f"[error] >>> Usage: SUGAR DESCRIBE \"alias\" AS \"note\"")
#         return True

#     if command.upper().startswith("SNAPSHOT "):
#         label = command[9:].strip().strip('"').strip("'")
#         if not label:
#             print("[snapshot] >>> Missing label.")
#             return True
#         snapshots[label] = (
#             uimap.summary().copy(),
#             uimap.descriptions.copy() if hasattr(uimap, "descriptions") else {}
#         )
#         print(f"[snapshot] >>> Snapshot saved as '{label}'")
#         return True

#     if command.upper().startswith("RESTORE "):
#         label = command[8:].strip().strip('"').strip("'")
#         if label not in snapshots:
#             print(f"[restore] >>> No snapshot found with label '{label}'")
#             return True
#         alias_map, desc_map = snapshots[label]
#         uimap.configure(alias_map)
#         if hasattr(uimap, "descriptions"):
#             uimap.descriptions = desc_map.copy()
#         print(f"[restore] >>> Snapshot '{label}' restored.")
#         return True

#     if command.upper().startswith("LOCK GROUP"):
#         try:
#             raw = command[10:].strip()
#             raw = re.sub(r'\[\s*(.*?)\s*\]', lambda m: '(' + m.group(1) + ',)', raw)
#             group = ast.literal_eval(raw)
#             added = 0
#             for alias in group:
#                 alias = str(alias)
#                 if alias in uimap.summary():
#                     locked_aliases.add(alias)
#                     added += 1
#             print(f"[lock] >>> Locked {added} aliases from group.")
#         except Exception as e:
#             print(f"[error] >>> Failed to lock group: {e}")
#         return True

#     if command.upper() == "VALIDATE":
#         from limit.limit_constants import CONTROL_SYMBOLS  # or wherever your valid tokens list comes from
#         valid_tokens = set(CONTROL_SYMBOLS)
#         invalid = [(alias, token) for alias, token in uimap.summary().items() if token not in valid_tokens]

#         if not invalid:
#             print("[validate] >>> All aliases map to valid tokens.")
#         else:
#             print(f"[validate] >>> Found {len(invalid)} invalid aliases:")
#             for alias, token in invalid:
#                 print(f"{alias:>12} → {token}  [INVALID]")
#         return True

#     if command.upper().startswith("DESCRIBE TOKEN "):
#         try:
#             parts = re.split(r'\s+AS\s+', command[15:], maxsplit=1)
#             if len(parts) != 2:
#                 raise ValueError
#             token = parts[0].strip().strip('"').strip("'")
#             note = parts[1].strip().strip('"').strip("'")
#             from limit.limit_constants import CONTROL_SYMBOLS
#             if token not in CONTROL_SYMBOLS:
#                 print(f"[describe] >>> Unknown token: {token}")
#             else:
#                 uimap.token_descriptions[token] = note
#                 print(f"[describe] >>> TOKEN {token} → {note}")
#         except Exception:
#             print(f"[error] >>> Usage: SUGAR DESCRIBE TOKEN \"X\" AS \"note\"")
#         return True

#     if command.upper() == "ANALYZE":

#         mapping = uimap.summary()
#         token_counts = Counter(mapping.values())
#         alias_lengths = {alias: len(alias) for alias in mapping}
#         inverse = defaultdict(list)
#         for alias, token in mapping.items():
#             inverse[token].append(alias)

#         print("[analyze] >>> Sugar Alias Analysis")
#         print(f"- Total aliases: {len(mapping)}")
#         print(f"- Unique token types: {len(token_counts)}")
#         print(f"- Most aliased tokens:")

#         for token, count in token_counts.most_common(5):
#             print(f"    {token:<12} → {count} aliases")

#         longest = sorted(alias_lengths.items(), key=lambda x: x[1], reverse=True)[:3]
#         if longest:
#             print("- Longest alias names:")
#             for alias, length in longest:
#                 print(f"    {alias:<12} ({length} chars)")

#         max_group = max(inverse.items(), key=lambda x: len(x[1]), default=None)
#         if max_group:
#             token, aliases = max_group
#             print(f"- Token with largest alias group: {token} ({len(aliases)} aliases)")

#         return True

#     if command == "":
#         print(uimap.report(verbose=True))
#         return True

#     # Inline sugar alias map
#     try:
#         sugar_history.append(command)
#         command = re.sub(r'\[\s*(.*?)\s*\]', lambda m: '(' + m.group(1) + ',)', command)
#         raw_map = ast.literal_eval(command)

#         final_mapping = {}
#         for key, sym in raw_map.items():
#             if isinstance(key, (list, tuple, set)):
#                 for alias in key:
#                     if alias in locked_aliases:
#                         print(f"[skip] >>> Alias locked: {alias}")
#                         continue
#                     final_mapping[str(alias)] = sym
#             else:
#                 alias = str(key)
#                 if alias in locked_aliases:
#                     print(f"[skip] >>> Alias locked: {alias}")
#                     continue
#                 final_mapping[alias] = sym

#         uimap.configure(final_mapping)
#         print("[ok] >>> Sugar aliases updated.")
#         print("\n".join(f"{alias:>12} → {sym}" for alias, sym in sorted(final_mapping.items())))

#     except Exception as e:
#         print("[error] >>> Failed to configure sugar aliases:")
#         print(e)

#     return True


def start_repl(target: str = "py", verbose: bool = False) -> None:
    print(f"Limit REPL [target={target}]. Type 'exit' or 'quit' to leave.")
    env_globals: dict[str, Any] = {}
    current_module: str | None = None
    imported_files: set[str] = set()

    while True:
        try:
            src_lines: list[str] = []
            brace_count = 0
            while True:
                prompt = ">>> " if not src_lines else "... "
                line = input(prompt)
                if line.strip() in ("exit", "quit") and not src_lines:
                    print("Exiting Limit REPL.")
                    return
                src_lines.append(line)
                brace_count += line.count("{") - line.count("}")
                if brace_count <= 0 and (
                    line.strip().endswith("}")
                    or not any("{" in line_ for line_ in src_lines)
                ):  # pragma: no branch
                    break
            src = "\n".join(src_lines).strip()
            if not src:
                continue
            if src.strip().startswith("#"):
                continue
            if src.lower() == "verbose-mode":
                verbose = not verbose
                print(f"[mode] >>> Verbose mode {'ON' if verbose else 'OFF'}")
                continue
            if handle_sugar_command(src):
                continue
            if src.upper().startswith("MODULE "):
                current_module = src.split("MODULE", 1)[1].strip()
                print(f"[module] >>> Current module: {current_module}")
                continue
            if src.upper().startswith("IMPORT "):
                import_path = src.split("IMPORT", 1)[1].strip().strip('"')
                if not import_path.endswith(".limit"):
                    import_path += ".limit"
                if import_path in imported_files:
                    print(f"[import skipped] >>> Already imported: {import_path}")
                    continue
                try:
                    with open(import_path, encoding="utf-8") as f:
                        file_src = f.read()
                    lexer = Lexer(CharacterStream(file_src))
                    file_tokens: list[Token] = []
                    while True:
                        tok = lexer.next_token()
                        if tok.type == "EOF":
                            break
                        file_tokens.append(tok)
                    if not file_tokens or any(
                        tok.type == "ERROR" for tok in file_tokens
                    ):
                        print("[import error] >>> Invalid tokens in file.")
                        continue
                    parsed_ast: list[ASTNode] = Parser(file_tokens).parse()
                    transpiler = Transpiler(target)
                    emitter = transpiler.emitter
                    for node in parsed_ast:
                        emitter._visit(node)  # type: ignore[attr-defined]
                    code = emitter.get_output()
                    if code.strip() and target == "py":
                        exec(code, env_globals)
                    else:  # pragma: no cover
                        print("[info] transpilation complete")
                        print(f"[warn] Execution not supported for: {target}")
                    imported_files.add(import_path)
                    print(f"[imported] >>> {import_path}")
                except Exception as e:
                    print(f"[import error] >>> {e}")
                continue

            lexer = Lexer(CharacterStream(src, 0, 1, 1))
            tokens: list[Token] = []
            try:
                while True:
                    tok = lexer.next_token()
                    if tok.type == "EOF":
                        break
                    mapped = (
                        uimap.get_token(tok.value, tok.line, tok.col)
                        if tok.type == "IDENT"
                        else None
                    )
                    tokens.append(mapped if mapped else tok)
            except Exception:
                print_traceback()
                continue

            if (
                not tokens
                or all(t.type == "EOF" for t in tokens)
                or any(t.type == "ERROR" for t in tokens)
            ):
                print("[error] >>>")
                print(f"Invalid token stream: {tokens}")
                continue

            parser = Parser(tokens)
            try:
                first_type = tokens[0].type
                if first_type in ("PLUS", "SUB", "MULT", "DIV", "MOD", "AND", "NOT"):
                    parsed_ast = parser.parse_expr_entrypoint()
                else:
                    try:
                        parsed_ast = parser.parse()
                    except SyntaxError:
                        try:
                            parsed_ast = parser.parse_expr_entrypoint()
                            if parsed_ast == []:
                                continue
                        except Exception:
                            print_traceback()
                            continue
                        else:
                            continue  # fallback succeeded, skip error
            except Exception:
                print_traceback()
                continue

            if not parsed_ast or parsed_ast == []:
                continue

            try:
                transpiler = Transpiler(target)
                emitter = transpiler.emitter
            except NotImplementedError as e:
                print("[error] >>>")
                print(f"Transpilation error: {e}")
                continue

            try:
                nodes: list[ASTNode]
                if isinstance(parsed_ast, list):
                    if parsed_ast and isinstance(parsed_ast[0], list):
                        # Flatten List[List[ASTNode]] → List[ASTNode]
                        flattened: list[Any] = []
                        for sub in parsed_ast:
                            if isinstance(sub, list):
                                flattened.extend(sub)
                            else:
                                raise TypeError("Expected list of lists of ASTNode")

                        nodes = flattened
                    else:
                        nodes = parsed_ast
                else:
                    nodes = [parsed_ast]
                for node in nodes:
                    if hasattr(emitter, "line" "s"):
                        emitter.lines.clear()
                    if node.kind in (
                        "expr_stmt",
                        "arith",
                        "bool",
                        "compare",
                        "call",
                        "member",
                        "identifier",
                        "float",
                        "number",
                        "string",
                        "new",
                    ):
                        inner = (
                            node.children[0]
                            if node.kind == "expr_stmt" and node.children
                            else node
                        )
                        result = emitter.emit_expr(inner)  # type: ignore[attr-defined]
                        if target == "py":
                            try:
                                if verbose:
                                    print(f"[py-expr] >>> {result}")
                                exec(f"_ = {result}", env_globals)
                                result_val = env_globals.get("_")
                                if result_val is not None:
                                    print(result_val)
                            except Exception:
                                print_traceback()
                        else:  # pragma: no cover
                            print("[info] transpilation complete")
                            print(f"[warn] Execution not supported for: {target}")
                        continue
                    emitter._visit(node)  # type: ignore[attr-defined]
                    code = emitter.get_output()
                    if code.strip():  # pragma: no branch
                        if target == "py":
                            try:
                                exec(code, env_globals)
                            except Exception:
                                print_traceback()
                        else:
                            print("[info] transpilation complete")
                            print(f"[warn] Execution not supported for: {target}")
            except Exception:
                print_traceback()

        except (KeyboardInterrupt, EOFError):
            print("\nExiting Limit REPL.")
            break
        except Exception as e:  # pragma: no cover
            raise RuntimeError("Unreachable unless stdout is broken") from e


def main() -> None:
    start_repl()


if __name__ == "__main__":
    main()
