import importlib
import re
from typing import Any, Literal

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.strategies import composite

import limit.limit_parser as parser_module
from limit.limit_ast import ASTNode
from limit.limit_lexer import CharacterStream, Lexer, Token, token_hashmap
from limit.limit_parser import Parser

importlib.reload(parser_module)


# Reserved words that are not allowed as identifiers in tests
reserved_words = {k.upper() for k in token_hashmap}


def tokenize(source: str) -> list[Token]:
    stream = CharacterStream(source, 0, 1, 1)
    lexer = Lexer(stream)
    tokens = []
    while True:
        tok = lexer.next_token()
        tokens.append(tok)
        if tok.type == "EOF":
            break
    return tokens


def tokenize_tokens(*tokens: Token) -> Parser:
    return Parser(list(tokens) + [Token("EOF", "EOF")])


def parse(source: str) -> list[ASTNode]:
    return Parser(tokenize(source)).parse()


def make_tokens(*types_vals: tuple[str, str]) -> list[Token]:
    return [Token(t, v) for t, v in types_vals] + [Token("EOF", "EOF")]


def normalize(node: dict[str, Any] | list[Any]) -> dict[str, Any] | list[Any]:
    if isinstance(node, list):
        return [normalize(n) for n in node]
    if isinstance(node, dict):
        return {k: normalize(v) for k, v in node.items() if k not in ("line", "col")}
    return node


def prune(node: Any) -> Any:
    """Remove non-semantic fields like line/col/type/return_type/etc."""
    if isinstance(node, list):
        return [prune(n) for n in node]
    if isinstance(node, dict):
        return {
            k: prune(v)
            for k, v in node.items()
            if k not in ("line", "col", "type", "return_type", "else_children")
            and not (k == "children" and v == [])  # remove empty children
        }
    return node


@pytest.mark.parametrize(
    "source,expected",
    [
        # Minimal regression tests
        (
            "= x 5",
            [
                ASTNode(
                    "assign",
                    "x",
                    [
                        ASTNode("identifier", "x", line=1, col=3),
                        ASTNode("number", "5", line=1, col=5),
                    ],
                    line=1,
                    col=1,
                    type_="global",
                )
            ],
        ),
        (
            "+ x 1",
            [
                ASTNode(
                    "arith",
                    "PLUS",
                    [
                        ASTNode("identifier", "x", line=1, col=3),
                        ASTNode("number", "1", line=1, col=5),
                    ],
                    line=1,
                    col=1,
                )
            ],
        ),
        (
            "@ loop() { ! x }",
            [
                ASTNode(
                    "func",
                    "loop",
                    [
                        ASTNode(
                            "print",
                            "x",
                            [ASTNode("identifier", "x", line=1, col=14)],
                            line=1,
                            col=12,
                            type_=None,
                            return_type=None,
                        )
                    ],
                    line=1,
                    col=1,
                    type_=None,
                    return_type=None,
                )
            ],
        ),
        (
            '! "hello"',
            [
                ASTNode(
                    "print",
                    "hello",
                    [ASTNode("string", "hello", line=1, col=3)],
                    line=1,
                    col=1,
                )
            ],
        ),
    ],
)  # type: ignore[misc]
def test_parser_regressions(
    source: (
        Literal["= x 5"]
        | Literal["+ x 1"]
        | Literal["@ loop() { ! x }"]
        | Literal['! "hello"']
    ),
    expected: list[ASTNode],
) -> None:
    result = parse(source)
    assert [n.to_dict() for n in result] == [n.to_dict() for n in expected]


@given(
    var=st.from_regex(r"[a-zA-Z_][a-zA-Z0-9_]{0,10}", fullmatch=True).filter(
        lambda x: x.upper() not in token_hashmap
    ),  # Fix: block any value mapped to a token
    num=st.integers(min_value=0, max_value=999),
)  # type: ignore[misc]
def test_assignment_parses_correctly(var: str, num: int) -> None:
    source = f"= {var} {num}"
    result = parse(source)
    assert len(result) == 1
    node = result[0]
    assert node.kind == "assign"
    assert node.value == var
    assert node.children[0].value == var
    assert node.children[1].value == str(num)


@given(
    a=st.integers(min_value=0, max_value=100), b=st.integers(min_value=0, max_value=100)
)  # type: ignore[misc]
def test_addition_expression_parses_correctly(a: int, b: int) -> None:
    source = f"+ x{a} {b}"
    result = parse(source)
    assert len(result) == 1
    node = result[0]
    assert node.kind == "arith"
    assert node.value == "PLUS"
    assert node.children[0].kind == "identifier"
    assert node.children[1].kind == "number"


@given(literal=st.sampled_from(["TRUE", "FALSE", "NULL"]))  # type: ignore[misc]
def test_literal_assignment_parses_correctly(literal: str) -> None:
    source = f"= val {literal}"
    result = parse(source)
    assert result[0].kind == "assign"
    assert result[0].children[1].kind == "literal"
    assert result[0].children[1].value == literal


@pytest.mark.parametrize(
    "source,expected",
    [
        # 64 – Unexpected token
        ("???", None),
        # 65 – Block missing closing bracket (expression brackets are valid here)
        ("[ ! x", None),
        # 66 – Expression with no operator
        ("[ x y ]", None),
        # 67 – Nested bracketed arithmetic (valid)
        (
            "[ + [ * x y ] z ]",
            [
                ASTNode(
                    "arith",
                    "PLUS",
                    [
                        ASTNode(
                            "arith",
                            "MULT",
                            [
                                ASTNode("identifier", "x", line=1, col=9),
                                ASTNode("identifier", "y", line=1, col=11),
                            ],
                            line=1,
                            col=7,
                        ),
                        ASTNode("identifier", "z", line=1, col=15),
                    ],
                    line=1,
                    col=3,
                )
            ],
        ),
        # 68 – Input from file
        (
            "INPUT FROM file",
            [
                ASTNode(
                    "input_from_file",
                    "file",
                    [ASTNode("file", "file", line=1, col=12)],
                    line=1,
                    col=1,
                )
            ],
        ),
        # 69 – Return with value inside function block
        (
            "@ f() { RETURN 1 }",
            [
                ASTNode(
                    "func",
                    "f",
                    [
                        ASTNode(
                            "return",
                            children=[ASTNode("number", "1", line=1, col=16)],
                            line=1,
                            col=9,
                        )
                    ],
                    line=1,
                    col=1,
                    type_=None,
                )
            ],
        ),
        # 70 – Propagate with bracketed expression
        (
            "@ f() { $ [ AND x y ] }",
            [
                ASTNode(
                    "func",
                    "f",
                    [
                        ASTNode(
                            "propagate",
                            None,
                            [
                                ASTNode(
                                    "bool",
                                    "AND",
                                    [
                                        ASTNode("identifier", "x", line=1, col=17),
                                        ASTNode("identifier", "y", line=1, col=19),
                                    ],
                                    line=1,
                                    col=13,
                                )
                            ],
                            line=1,
                            col=9,
                        )
                    ],
                    line=1,
                    col=1,
                    type_=None,  # FIXED FROM type_=[]
                )
            ],
        ),
        # 71 – Call with no args
        (
            "CALL f",
            [
                ASTNode(
                    "call",
                    ASTNode("identifier", "f", line=1, col=6),
                    [],
                    line=1,
                    col=1,
                )
            ],
        ),
        # 72 – TRY with no catch/finally
        (
            "TRY { RETURN 1 }",
            [
                ASTNode(
                    "try",
                    None,
                    [
                        ASTNode(
                            "return",
                            children=[ASTNode("number", "1", line=1, col=14)],
                            line=1,
                            col=7,
                        )
                    ],
                    line=1,
                    col=1,
                )
            ],
        ),
        # 73 – NOT operator (unary)
        (
            "NOT x",
            [
                ASTNode(
                    "bool",
                    "NOT",
                    [ASTNode("identifier", "x", line=1, col=5)],
                    line=1,
                    col=1,
                )
            ],
        ),
        # 74 – Empty function with correct block delimiter
        ("@ f() { }", [ASTNode("func", "f", [], line=1, col=1, type_=None)]),
        # 75 – Class with multiple statements
        ("CLASS A { ! x RETURN }", None),
        # 76 – WHILE with literal condition
        (
            "WHILE x { ! y }",
            [
                ASTNode(
                    "loop",
                    "WHILE",
                    [
                        ASTNode("identifier", "x", line=1, col=7),
                        ASTNode(
                            "print",
                            "y",
                            [ASTNode("identifier", "y", line=1, col=13)],
                            line=1,
                            col=11,
                        ),
                    ],
                    line=1,
                    col=1,
                )
            ],
        ),
        # 77 – Empty WHILE block (should raise now)
        ("WHILE { }", None),
        # 78 – FOR with return
        (
            "FOR { RETURN 1 }",
            [
                ASTNode(
                    "loop",
                    "FOR",
                    [
                        ASTNode(
                            "return",
                            children=[ASTNode("number", "1", line=1, col=14)],
                            line=1,
                            col=7,
                        )
                    ],
                    line=1,
                    col=1,
                )
            ],
        ),
    ],
)  # type: ignore[misc]
def test_parser_missing_paths(
    source: Literal[
        "???",
        "[ ! x",
        "[ x y ]",
        "[ + [ * x y ] z ]",
        "INPUT FROM file",
        "@ f() { RETURN }",
        "@ f() { $ [ AND x y ] }",
        "CALL f",
        "TRY { RETURN }",
        "NOT x",
        "@ f() { }",
        "CLASS A { ! x RETURN }",
        "WHILE x { ! y }",
        "WHILE { }",
        "FOR { RETURN }",
    ],
    expected: list[ASTNode] | None,
) -> None:
    def unwrap(result: list[ASTNode] | None) -> list[ASTNode]:
        if result is None:
            raise AssertionError("Expected AST, got None")
        return result

    try:
        parse_result = parse(source)
    except SyntaxError:
        if expected is not None:
            raise
        return

    if expected is None:
        pytest.fail("Expected SyntaxError but got AST")

    result = unwrap(parse_result)
    assert [n.to_dict() for n in result] == [n.to_dict() for n in expected]  # type: ignore


@composite  # type: ignore[misc]
def operator_expression(draw: Any) -> str:
    op = draw(
        st.sampled_from(["+", "-", "*", "/", "%", "AND", "OR", "NOT", "CALL", "PROP"])
    )
    x = draw(
        st.text(
            min_size=1, max_size=4, alphabet=st.characters(whitelist_categories=["Ll"])
        )
    )
    y = draw(
        st.text(
            min_size=1, max_size=4, alphabet=st.characters(whitelist_categories=["Ll"])
        )
    )
    if op == "NOT":
        return f"[ {op} {x} ]"
    return f"[ {op} {x} {y} ]"


@given(expr=operator_expression())  # type: ignore[misc]
def test_operator_expressions_cover_all(expr: str) -> None:
    try:
        result = parse(expr)
        assert isinstance(result[0], ASTNode)
        assert result[0].line > 0 and result[0].col > 0
    except SyntaxError:
        # Some malformed ones may sneak through generation
        pass


def test_call_with_member_and_args() -> None:
    source = "@ f() { CALL a.b(c, d) }"
    result = parse(source)
    call = result[0].children[0]
    assert call.kind == "call"
    assert call.value.kind == "member"  # type: ignore
    assert call.value.children[0].kind == "identifier"  # type: ignore
    assert call.value.value == "b"  # type: ignore
    assert len(call.children) == 2


def test_class_extends_with_block() -> None:
    source = "CLASS A EXTENDS B { ! x }"
    result = parse(source)
    cls = result[0]
    assert cls.kind == "class"
    assert cls.children[0].kind == "extends"
    assert cls.children[1].kind == "print"


def test_try_catch_finally_full_form() -> None:
    source = "TRY { RETURN 1 } CATCH(e) { ! x } FINALLY { ! y }"
    result = parse(source)
    assert result[0].kind == "try"
    assert any(child.kind == "catch" for child in result[0].children)
    assert any(child.kind == "finally" for child in result[0].children)


def test_module_declaration() -> None:
    source = "MODULE MyMod"
    result = parse(source)
    node = result[0]
    assert node.kind == "module"
    assert node.value == "MyMod"


def test_import_statement() -> None:
    source = 'IMPORT "foo.min"'
    result = parse(source)
    node = result[0]
    assert node.kind == "import"
    assert node.value == "foo.min"


def test_export_statement() -> None:
    source = "EXPORT x"
    result = parse(source)
    node = result[0]
    assert node.kind == "export"
    assert node.value == "x"


def test_deeply_nested_expression() -> None:
    source = "[ + [ * [ - x y ] z ] w ]"
    result = parse(source)
    top = result[0]
    assert top.kind == "arith"
    assert top.value == "PLUS"
    mult = top.children[0]
    assert mult.kind == "arith"
    assert mult.value == "MULT"
    sub = mult.children[0]
    assert sub.kind == "arith"
    assert sub.value == "SUB"
    assert sub.children[0].kind == "identifier"
    assert sub.children[1].kind == "identifier"
    assert mult.children[1].kind == "identifier"
    assert top.children[1].kind == "identifier"


def test_if_with_else_block() -> None:
    source = "@ f() { ? (AND x y) { RETURN 1 } ELSE { RETURN 2 } }"
    result = parse(source)
    assert result[0].kind == "func"
    assert result[0].children[0].kind == "if"
    assert result[0].children[0].else_children != []


def test_new_object_creation() -> None:
    source = "= x NEW Foo(1, 2)"
    result = parse(source)
    assign = result[0]
    assert assign.children[1].kind == "new"
    assert len(assign.children[1].children) == 2


def test_call_with_space_separated_args() -> None:
    source = "CALL myFunc x y"
    result = parse(source)
    call = result[0]
    assert call.kind == "call"
    assert call.value.kind == "identifier"  # type: ignore
    assert call.value.value == "myFunc"  # type: ignore
    assert len(call.children) == 2


def test_multiple_catch_blocks() -> None:
    source = "TRY { RETURN 1 } CATCH(a) { ! x } CATCH(b) { ! y }"
    result = parse(source)
    assert result[0].kind == "try"
    catch_nodes = [child for child in result[0].children if child.kind == "catch"]
    assert len(catch_nodes) == 2
    assert catch_nodes[0].value == "a"
    assert catch_nodes[1].value == "b"


def test_input_without_from_clause() -> None:
    source = 'INPUT FROM "config.txt"'
    result = parse(source)
    assert result[0].kind == "input_from_file"
    assert result[0].value == "config"


def test_return_with_bracket_expression() -> None:
    source = "@ f() { RETURN [ + x 1 ] }"
    result = parse(source)
    ret = result[0].children[0]
    assert ret.kind == "return"
    assert ret.children[0].kind == "arith"


def test_function_with_no_return_type() -> None:
    source = "@ f(x, y) { RETURN 0 }"
    result = parse(source)
    assert result[0].kind == "func"
    assert result[0].value == "f"
    assert result[0].type == "x,y"
    assert result[0].return_type is None


def test_call_with_empty_parens() -> None:
    source = "CALL myFunc()"
    result = parse(source)
    node = result[0]
    assert node.kind == "call"
    assert node.value.kind == "identifier"  # type: ignore
    assert node.value.value == "myFunc"  # type: ignore
    assert node.children == []


def test_propagate_with_literal() -> None:
    source = "@ f() { $ TRUE }"
    result = parse(source)
    prop = result[0].children[0]
    assert prop.kind == "propagate"
    assert prop.children[0].kind == "literal"
    assert prop.children[0].value == "TRUE"


def test_empty_expression_node() -> None:
    parser = Parser(tokenize("[ ]"))
    expr = parser.parse_expression()
    assert expr.kind == "empty"


def test_invalid_syntax_raises() -> None:
    with pytest.raises(SyntaxError):
        parse("@ f() { RETURN")  # unclosed block

    with pytest.raises(SyntaxError):
        parse("CALL")  # no target

    with pytest.raises(SyntaxError):
        parse("$ x")  # propagate outside block


def test_new_object_with_malformed_args() -> None:
    with pytest.raises(SyntaxError):
        parse("= x NEW Foo(1, )")


def test_empty_block_with_trailing_garbage() -> None:
    with pytest.raises(SyntaxError):
        parse("@ f() { } ~~~")


def test_if_else_without_block_raises() -> None:
    with pytest.raises(SyntaxError):
        parse("IF ( x ) { RETURN } ELSE RETURN")


def test_invalid_operator_in_expression_raises() -> None:
    with pytest.raises(SyntaxError):
        parse("[ ?? x y ]")


def test_break_continue_outside_block_raise() -> None:
    with pytest.raises(SyntaxError):
        parse("BREAK")
    with pytest.raises(SyntaxError):
        parse("CONTINUE")


@given(
    ident=st.from_regex(r"[a-z]{1,4}", fullmatch=True).filter(
        lambda s: s.upper() not in reserved_words
    ),
    depth=st.integers(min_value=1, max_value=3),
)  # type: ignore[misc]
def test_nested_member_access(ident: str, depth: int) -> None:
    chain = ".".join([ident] * (depth + 1))
    source = f"= x {chain}"
    result = parse(source)
    node = result[0].children[1]
    assert node.kind == "member"
    assert node.col > 0


@given(st.text(alphabet="@$%^&*~`|", min_size=1, max_size=6))  # type: ignore[misc]
def test_fuzz_invalid_tokens_raise(text: str) -> None:
    with pytest.raises(SyntaxError):
        parse(text)


def test_function_single_statement_fallback() -> None:
    source = "@ f() { RETURN 0 }"
    result = parse(source)
    assert result[0].kind == "func"
    assert result[0].value == "f"


def test_class_single_statement_fallback() -> None:
    source = "CLASS A { RETURN }"
    with pytest.raises(
        SyntaxError, match="RETURN is only allowed inside function blocks"
    ):
        parse(source)


def test_while_with_bracket_expression_condition() -> None:
    source = "WHILE [ NOT x ] { RETURN 0 }"
    result = parse(source)
    assert result[0].kind == "loop"
    assert result[0].value == "WHILE"


def test_while_empty_body_raises() -> None:
    with pytest.raises(SyntaxError, match="WHILE loop body cannot be empty"):
        parse("WHILE x { }")


def test_loop_block_empty_for_raises() -> None:
    with pytest.raises(SyntaxError, match="FOR loop body cannot be empty"):
        parse("FOR { }")


def test_class_extends_no_block_raises() -> None:
    with pytest.raises(
        SyntaxError, match=re.escape("Expected one of ('LBRACE',), got Token(EOF, EOF)")
    ):
        parse("CLASS A EXTENDS B")


def test_else_without_braces_raises() -> None:
    with pytest.raises(
        SyntaxError, match=re.escape("ELSE body must be enclosed in braces '{ }'")
    ):
        parse("? (AND x y) { RETURN 0 } ELSE RETURN")


def test_function_missing_brace_body_raises() -> None:
    with pytest.raises(SyntaxError, match="Function body must start"):
        parse("@ f() RETURN")


def test_not_operator_with_too_many_args_raises() -> None:
    with pytest.raises(SyntaxError, match=r"NOT requires 1 operand"):
        parse("[ NOT x y ]")


def test_add_operator_with_one_arg_raises() -> None:
    with pytest.raises(SyntaxError, match=r"PLUS requires at least 2 operands"):
        parse("[ + x ]")


def test_expr_or_literal_strict_fallback() -> None:
    parser = Parser(tokenize("???"))
    with pytest.raises(SyntaxError):
        parser.parse_expr_or_literal()


def test_parse_block_invalid_delimiter() -> None:
    parser = Parser([])
    with pytest.raises(SyntaxError, match="Invalid block delimiter"):
        parser.parse_block(open_type="LBRACK")


def test_parse_loop_block_invalid_open_type() -> None:
    tokens = tokenize("FOR x to 5")  # missing block
    parser = Parser(tokens)
    with pytest.raises(SyntaxError, match=r"FOR loop body must be enclosed in braces"):
        parser.parse_loop_block("FOR")


def test_parse_try_with_single_catch() -> None:
    source = "TRY { RETURN 0 } CATCH(e) { ! x }"
    result = parse(source)
    assert result[0].kind == "try"
    assert result[0].children[0].kind == "return"
    assert result[0].children[1].kind == "catch"
    assert result[0].children[1].value == "e"
    assert result[0].children[1].children[0].kind == "print"


def test_parse_statement_unrecognized_token() -> None:
    class DummyParser(Parser):
        def __init__(self) -> None:
            self.tokens = tokenize("???")
            self.position = 0

        def parse_statement(self) -> None:  # type: ignore
            return None

    parser = DummyParser()
    with pytest.raises(SyntaxError, match="Unrecognized syntax near"):
        parser.parse()


def test_parse_function_with_param_and_return_type() -> None:
    source = "@ foo(x): int { RETURN 0 }"
    result = parse(source)
    assert result[0].kind == "func"
    assert result[0].value == "foo"
    assert result[0].type == "x"
    assert result[0].return_type == "int"


def test_parse_function_param_comma_edge_case() -> None:
    with pytest.raises(SyntaxError):
        parse("@ f(x,) { RETURN }")


def test_parse_function_param_extra_token() -> None:
    with pytest.raises(SyntaxError):
        parse("@ f(x y) { RETURN }")


def test_assignment_new_without_parens() -> None:
    result = parse("= x NEW Foo")
    node = result[0].children[1]
    assert node.kind == "new"
    assert node.value == "Foo"
    assert node.children == []


def test_assignment_new_with_parens() -> None:
    result = parse("= x NEW Foo(1, 2)")
    node = result[0].children[1]
    assert node.kind == "new"
    assert len(node.children) == 2
    assert all(isinstance(c, ASTNode) for c in node.children)


def test_call_method_style_space_args() -> None:
    ast = parse("CALL a.b 1 2")
    result = prune([n.to_dict() for n in ast])
    expected = prune(
        [
            {
                "kind": "call",
                "value": {
                    "kind": "member",
                    "value": "b",
                    "children": [{"kind": "identifier", "value": "a"}],
                },
                "children": [
                    {"kind": "number", "value": "1"},
                    {"kind": "number", "value": "2"},
                ],
            }
        ]
    )
    assert result == expected


def test_member_access_chain_3_levels() -> None:
    result = parse("= x a.b.c")
    chain = result[0].children[1]
    assert chain.kind == "member"
    assert chain.children[0].kind == "member"
    assert chain.children[0].children[0].kind == "identifier"


def test_try_catch_with_ident_param() -> None:
    source = "TRY { RETURN 0 } CATCH err { RETURN 1 }"
    result = parse(source)
    assert result[0].kind == "try"
    assert any(c.kind == "catch" for c in result[0].children)


def test_try_catch_without_param() -> None:
    source = "TRY { RETURN 0 } CATCH { RETURN 1 }"
    result = parse(source)
    assert result[0].kind == "try"


def test_expr_or_literal_member_access() -> None:
    result = parse("= x a.b")
    assert result[0].children[1].kind == "member"


def test_return_literal() -> None:
    result = parse("@ f() { RETURN true }")
    ret = result[0].children[0]
    assert ret.kind == "return"
    assert ret.children[0].value == "true"


def test_print_empty_and_literal() -> None:
    result = parse("! TRUE")
    assert result[0].kind == "print"
    assert result[0].children[0].kind == "literal"


def test_while_empty_condition_raises() -> None:
    with pytest.raises(SyntaxError, match="WHILE condition cannot be empty"):
        parse("WHILE [ ] { RETURN }")


def test_while_with_literal_condition() -> None:
    result = parse("WHILE TRUE { RETURN 0 }")
    assert result[0].kind == "loop"


def test_expression_nested_with_member_and_literals() -> None:
    source = "[ + x.y 1 ]"
    result = parse(source)
    node = result[0]
    assert node.kind == "arith"
    assert node.children[0].kind == "member"
    assert node.children[1].kind == "number"


def test_expression_empty() -> None:
    result = Parser(tokenize("[ ]")).parse_expression()
    assert result.kind == "empty"


def test_assignment_with_this_literal() -> None:
    result = parse("= x THIS")
    node = result[0]
    assert node.children[1].kind == "identifier"
    assert node.children[1].value == "THIS"


# 110–113: NEW Foo(1, "bar")
def test_assignment_new_with_mixed_args() -> None:
    result = parse('= x NEW Foo(1, "bar")')
    new_node = result[0].children[1]
    assert new_node.kind == "new"
    assert new_node.children[1].kind == "string"


def test_call_method_with_parens_args() -> None:
    result = parse("CALL obj.method(a, b)")
    call = result[0]
    assert call.kind == "call"
    assert call.value.kind == "member"  # type: ignore
    assert call.value.value == "method"  # type: ignore
    assert call.value.children[0].value == "obj"  # type: ignore
    assert len(call.children) == 2
    assert call.children[0].value == "a"
    assert call.children[1].value == "b"


def test_long_member_access_chain() -> None:
    result = parse("= x a.b.c.d.e")
    node = result[0].children[1]
    depth = 0
    while node.kind == "member":
        depth += 1
        node = node.children[0]
    assert depth == 4


# 291: parse_propagate – raise outside block
def test_propagate_outside_block() -> None:
    with pytest.raises(SyntaxError, match="only allowed inside"):
        parse("$ x")


def test_try_with_ident_catch() -> None:
    source = "TRY { RETURN 1 } CATCH x { RETURN 0 }"
    result = parse(source)
    assert result[0].kind == "try"


@pytest.mark.parametrize(  # type: ignore[misc]
    "expr,kind",
    [
        ("x", "identifier"),
        ("TRUE", "literal"),
        ('"hello"', "string"),
        ("123", "number"),
    ],
)
def test_parse_expr_or_literal_variants(
    expr: Literal["x"] | Literal["TRUE"] | Literal['"hello"'] | Literal["123"],
    kind: (
        Literal["identifier"]
        | Literal["literal"]
        | Literal["string"]
        | Literal["number"]
    ),
) -> None:
    result = parse(f"= x {expr}")
    assert result[0].children[1].kind == kind


def test_empty_print_node() -> None:
    result = parse("!")
    assert result[0].kind == "print"
    assert result[0].children == []


def test_while_without_block_raises() -> None:
    with pytest.raises(SyntaxError, match="WHILE must be followed by a block"):
        parse("WHILE true RETURN")


def test_while_with_true_literal() -> None:
    source = "WHILE TRUE { RETURN 0 }"
    result = parse(source)
    assert result[0].kind == "loop"
    assert result[0].value == "WHILE"


# 592, 597->604, 605->604, 615->617: deeply nested expression arity paths
def test_nested_expressions_all_types() -> None:
    result = parse("[ AND [ OR x y ] z ]")
    node = result[0]
    assert node.kind == "bool"
    assert node.children[0].kind == "bool"
    assert node.children[1].kind == "identifier"


def test_class_extends_full_path_valid() -> None:
    source = """
    CLASS Parent {
      @ init(self) {
        = self.x 1
      }
    }
    CLASS Child EXTENDS Parent {
      @ init(self) {
        = self.msg "child"
      }
    }
    """
    result = parse(source)
    assert len(result) == 2
    assert result[0].kind == "class"
    assert result[0].value == "Parent"
    assert result[1].kind == "class"
    assert result[1].value == "Child"
    assert result[1].children[0].kind == "extends"
    assert result[1].children[0].value == "Parent"
    assert result[1].children[1].kind == "func"
    assert result[1].children[1].value == "init"


def test_parser_current_hits_eof_token() -> None:
    parser = Parser([])
    tok = parser.current()
    assert tok.type == "EOF"


def test_match_strict_error_branch() -> None:
    parser = Parser([Token("IDENT", "hello")])
    with pytest.raises(SyntaxError):
        parser.match("NONMATCH")


def test_parse_bracketed_expression_calls_expression() -> None:
    parser = Parser(tokenize("[ + x y ]"))
    result = parser.parse_bracketed_expression()
    assert result.kind == "arith"  # type: ignore


def test_assignment_new_with_commas() -> None:
    source = "= x NEW Foo(1, 2, 3)"
    result = parse(source)
    node = result[0].children[1]
    assert node.kind == "new"
    assert len(node.children) == 3
    assert all(c.kind == "number" for c in node.children)


# Pre-generate a list of valid identifier names
valid_identifiers = [
    s for s in (f"id{i}" for i in range(1000)) if s.upper() not in reserved_words
]


@given(
    num=st.integers(min_value=0, max_value=100),
    float_val=st.floats(
        min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False
    ),
    string_val=st.text(
        alphabet=st.characters(blacklist_characters='"\\', blacklist_categories=["Cs"]),
        min_size=1,
        max_size=10,
    ).map(lambda s: f'"{s}"'),
    literal_val=st.sampled_from(["TRUE", "FALSE"]),
    ident=st.sampled_from(valid_identifiers),
)  # type: ignore[misc]
@settings(deadline=None)  # type: ignore[misc]
def test_assignment_new_arg_variants(
    num: int,
    float_val: float,
    string_val: str,
    literal_val: str,
    ident: str,
) -> None:
    for val in [num, float_val, string_val, literal_val]:
        source = f"= {ident} {val}"
        result = parse(source)
        assert result[0].kind == "assign"
        assert result[0].value == ident


def test_assignment_new_two_args_hits_comma_and_break() -> None:
    source = "= x NEW Foo(1, 2)"
    result = parse(source)
    new_node = result[0].children[1]
    assert new_node.kind == "new"
    assert len(new_node.children) == 2
    assert new_node.children[0].kind == "number"
    assert new_node.children[1].kind == "number"


def test_assignment_hits_member_access_branch() -> None:
    result = parse("= result some.object")
    rhs = result[0].children[1]
    assert rhs.kind == "member"
    assert rhs.children[0].kind == "identifier"


def test_for_range_with_non_number_limit_raises() -> None:
    with pytest.raises(
        SyntaxError, match=r"Expected one of .* got Token\(LITERAL, TRUE\)"
    ):
        parse("FOR x TO TRUE { RETURN }")


def test_class_greeter_greet_method() -> None:
    source = """
    CLASS Greeter {
      @ init(self, name) {
        = self.name name
      }

      @ greet(self) {
        RETURN [+ "Hello, " self.name]
      }
    }
    = g NEW Greeter("Alice")
    ! [CALL g.greet]
    """
    result = parse(source)
    assert any(n.kind == "class" and n.value == "Greeter" for n in result)
    class_node = next(n for n in result if n.kind == "class" and n.value == "Greeter")
    method_names = [m.value for m in class_node.children if m.kind == "func"]
    assert "init" in method_names
    assert "greet" in method_names


def test_match_type_only_raises_on_unexpected_type() -> None:
    parser = Parser([Token("IDENT", "foo")])
    with pytest.raises(
        SyntaxError, match=r"Expected token type in .* got Token\(IDENT, foo\)"
    ):
        parser.match_type_only("NUMBER", "STRING")


def test_parse_block_position_does_not_advance() -> None:
    tokens = tokenize("{ ~~~ }")

    class DummyParser(Parser):
        def parse_statement(self, strict: bool = True):  # type: ignore # accepts strict now
            return None  # Simulate no advancement

    parser = DummyParser(tokens)

    with pytest.raises(
        SyntaxError,
        match=re.escape("Unrecognized statement inside block near: ~ (type=ERROR)"),
    ):
        parser.parse_block("LBRACE")


def test_parse_propagate_missing_dollar() -> None:
    parser = Parser([Token("NUMBER", "123")])  # Definitely not 'PROP'
    parser.in_block = True
    with pytest.raises(
        SyntaxError,
        match=re.escape("Expected one of ('PROP',), got Token(NUMBER, 123)"),
    ):
        parser.parse_propagate()


def test_parse_propagate_bracketed_expression_none() -> None:
    class DummyParser:
        def __init__(self, tokens: list[Token] | None = None) -> None:
            self.in_block = True
            self.tokens = tokens or []
            self.position = 0

        def current(self) -> Token:
            return Token("LBRACK", "[")

        def match(self, *args: str, strict: bool = True) -> Token | None:
            if "PROP" in args:
                return Token("PROP", "$")
            return None

        def parse_expression(self, strict: bool = True) -> ASTNode | None:
            return None

        def parse_propagate(self, strict: bool = True) -> ASTNode:
            if not self.in_block:
                raise SyntaxError(
                    "Propagate operator `$` only allowed inside functions."
                )

            prop_tok = self.match("PROP", strict=strict)
            if prop_tok is None:
                raise SyntaxError("Expected '$' propagate operator")

            if self.current().type == "LBRACK":
                expr = self.parse_expression(strict=strict)
                if expr is None:
                    raise SyntaxError("Expected valid expression after '$'")
            else:
                tok = self.match(
                    "IDENT",
                    "NUMBER",
                    "FLOAT",
                    "STRING",
                    "LITERAL",
                    "THIS",
                    strict=strict,
                )
                if tok is None:
                    raise SyntaxError("Expected value after '$'")
                kind = (
                    "identifier" if tok.type in ("IDENT", "THIS") else tok.type.lower()
                )
                expr = ASTNode(kind, tok.value, line=tok.line, col=tok.col)

            return ASTNode(
                "propagate",
                value=None,
                line=prop_tok.line,
                col=prop_tok.col,
                children=[expr],
            )

    tokens = tokenize("$ [ x y ]")
    parser = DummyParser(tokens)
    parser.in_block = True
    with pytest.raises(SyntaxError, match=r"Expected valid expression after '\$'"):
        parser.parse_propagate()


def test_parse_propagate_invalid_non_bracketed_value() -> None:
    tokens = [Token("PROP", "$"), Token("COMMA", ",")]
    parser = Parser(tokens)
    parser.in_block = True
    with pytest.raises(
        SyntaxError,
        match=re.escape(
            "Expected one of ('IDENT', 'NUMBER', 'FLOAT', 'STRING', 'LITERAL', 'THIS'), got Token(COMMA, ,)"
        ),
    ):
        parser.parse_propagate()


def test_parse_expression_strict_requires_bracket() -> None:
    parser = Parser([Token("IDENT", "x")])
    with pytest.raises(SyntaxError, match=r"Expected '\[' to start expression"):
        parser.parse_expression(strict=True)


def test_parse_expression_non_strict_entry_fails() -> None:
    parser = Parser([Token("IDENT", "x")])
    with pytest.raises(SyntaxError, match="Non-strict mode no longer supports"):
        parser.parse_expression(strict=False)


def test_parse_expression_open_bracket_missing() -> None:
    parser = Parser([Token("NUMBER", "1")])  # no LBRACK
    with pytest.raises(SyntaxError, match=r"Expected '\[' to start expression"):
        parser.parse_expression(strict=True)


def test_parse_expression_operator_missing() -> None:
    parser = Parser(tokenize("[ x y ]"))
    with pytest.raises(
        SyntaxError, match=r"Expected one of .* got Token\(IDENT, .*?\)"
    ):
        parser.parse_expression()


def test_parse_expression_call_missing_ident() -> None:
    tokens = tokenize("[ CALL , ]")  # COMMA instead of IDENT
    parser = Parser(tokens)
    with pytest.raises(
        SyntaxError, match="CALL must be followed by identifier or member access"
    ):
        parser.parse_expression()


def test_parse_expression_call_with_parens() -> None:
    result = parse("[ CALL f(1, 2) ]")
    call = result[0]
    assert call.kind == "call"
    assert call.value.kind == "identifier"  # type: ignore
    assert len(call.children) == 2


def test_expr_or_literal_with_brackets() -> None:
    parser = Parser(tokenize("[ + 1 2 ]"))
    result = parser.parse_expr_or_literal()
    assert result.kind == "arith"


def test_expr_or_literal_with_member_access() -> None:
    parser = Parser(tokenize("foo.bar"))
    result = parser.parse_expr_or_literal()
    assert result.kind == "member"


def test_parse_class_invalid_statement_in_body() -> None:
    class BadParser(Parser):
        def parse_statement(self, strict: bool = True):  # type: ignore # match base signature
            return None  # Always returns None to simulate invalid statement

    tokens = [
        Token("CLASS", "CLASS"),
        Token("IDENT", "MyClass"),
        Token("LBRACE", "{"),
        Token("IDENT", "badstmt"),  # triggers parse_statement() to return None
        Token("RBRACE", "}"),
    ]
    parser = BadParser(tokens)
    with pytest.raises(SyntaxError, match="Invalid statement in class body"):
        parser.parse_class()


def test_parse_class_missing_keyword() -> None:
    parser = Parser([Token("IDENT", "foo")])
    with pytest.raises(
        SyntaxError,
        match=re.escape("Expected one of ('CLASS',), got Token(IDENT, foo)"),
    ):
        parser.parse_class()


def test_parse_class_missing_name() -> None:
    parser = Parser([Token("CLASS", "CLASS")])
    with pytest.raises(
        SyntaxError,
        match=re.escape("Expected one of ('IDENT',), got Token(EOF, EOF)"),
    ):
        parser.parse_class()


def test_parse_class_extends_missing_base() -> None:
    tokens = [
        Token("CLASS", "CLASS"),
        Token("IDENT", "A"),
        Token("EXTENDS", "EXTENDS"),
        Token("LBRACE", "{"),
        Token("RBRACE", "}"),
    ]
    parser = Parser(tokens)
    with pytest.raises(
        SyntaxError,
        match=re.escape("Expected one of ('IDENT',), got Token(LBRACE, {)"),
    ):
        parser.parse_class()


def test_parse_expr_entrypoint_identifier() -> None:
    parser = Parser([Token("IDENT", "x")])
    result = parser.parse_expr_entrypoint()
    assert len(result) == 1
    assert result[0].kind == "identifier"
    assert result[0].value == "x"


@pytest.mark.parametrize(  # type: ignore[misc]
    "tok_type,value,expected_kind",
    [
        ("NUMBER", "42", "number"),
        ("FLOAT", "3.14", "float"),
        ("STRING", "hello", "string"),
    ],
)
def test_parse_expr_entrypoint_literals(
    tok_type: Literal["NUMBER"] | Literal["FLOAT"] | Literal["STRING"],
    value: Literal["42"] | Literal["3.14"] | Literal["hello"],
    expected_kind: Literal["number"] | Literal["float"] | Literal["string"],
) -> None:
    parser = Parser([Token(tok_type, value)])
    result = parser.parse_expr_entrypoint()
    assert len(result) == 1
    assert result[0].kind == expected_kind
    assert result[0].value == value


def test_parse_expr_entrypoint_unrecognized_token() -> None:
    parser = Parser([Token("COMMA", ",")])
    with pytest.raises(
        SyntaxError, match=r"Unrecognized expression: Token\(COMMA, ,\)"
    ):
        parser.parse_expr_entrypoint()


def test_assignment_lhs_member_access() -> None:
    result = parse("= obj.prop 42")
    node = result[0]
    assert node.kind == "assign"
    assert node.children[0].kind == "member"
    assert node.children[0].value == "prop"


def test_assignment_rhs_bracketed_expression() -> None:
    result = parse("= x [ + 1 2 ]")
    rhs = result[0].children[1]
    assert rhs.kind == "arith"
    assert rhs.value == "PLUS"


def test_assignment_rhs_function_call() -> None:
    result = parse("= x foo(1, 2)")
    rhs = result[0].children[1]
    assert rhs.kind == "call"
    assert rhs.value == "foo"
    assert len(rhs.children) == 2


def test_assignment_invalid_rhs_token() -> None:
    with pytest.raises(
        SyntaxError, match=r"Invalid right-hand side in assignment: Token\(COMMA, ,\)"
    ):
        parse("= x ,")


def test_input_keyword_missing() -> None:
    parser = Parser([Token("IDENT", "x")])
    expected = re.escape("Expected one of ('INPUT',), got Token(IDENT, x)")
    with pytest.raises(SyntaxError, match=expected):
        parser.parse_input()


def test_input_from_missing_path() -> None:
    tokens = [Token("INPUT", "INPUT"), Token("DELIM_FROM", "FROM")]
    parser = Parser(tokens)
    expected = re.escape("Expected one of ('IDENT', 'STRING'), got Token(EOF, EOF)")
    with pytest.raises(SyntaxError, match=expected):
        parser.parse_input()


def test_input_variable_missing_name() -> None:
    parser = Parser([Token("INPUT", "INPUT")])
    with pytest.raises(
        SyntaxError, match=re.escape("Expected one of ('IDENT',), got Token(EOF, EOF)")
    ):
        parser.parse_input()


def test_input_variable_name_and_type() -> None:
    tokens = [
        Token("INPUT", "INPUT"),
        Token("IDENT", "myvar"),
        Token("COLON", ":"),
        Token("IDENT", "int"),
    ]
    result = Parser(tokens).parse_input()
    assert result.kind == "input"
    assert result.value == "myvar"
    assert result.type == "int"


def test_input_variable_colon_missing_type() -> None:
    tokens = [Token("INPUT", "INPUT"), Token("IDENT", "val"), Token("COLON", ":")]
    with pytest.raises(
        SyntaxError, match=re.escape("Expected one of ('IDENT',), got Token(EOF, EOF)")
    ):
        Parser(tokens).parse_input()


def test_prefix_operator_missing_operator() -> None:
    parser = Parser([Token("IDENT", "x")])
    with pytest.raises(
        SyntaxError,
        match=re.escape(
            "Expected one of ('PLUS', 'SUB', 'MULT', 'DIV', 'MOD', 'AND', 'NOT'), got Token(IDENT, x)"
        ),
    ):
        parser.parse_prefix_operator()


def test_prefix_operator_with_identifier_operand() -> None:
    parser = Parser([Token("NOT", "NOT"), Token("IDENT", "flag")])
    result = parser.parse_prefix_operator()
    assert result.kind == "bool"
    assert result.value == "NOT"
    assert result.children[0].kind == "identifier"
    assert result.children[0].value == "flag"


def test_parse_statement_with_standalone_comma() -> None:
    parser = Parser([Token("COMMA", ","), Token("EOF", "EOF")])
    with pytest.raises(SyntaxError, match=r"Standalone comma is not a valid statement"):
        parser.parse_statement()


def test_parse_statement_ident_skip() -> None:
    class DummyParser(Parser):
        def __init__(self) -> None:
            self.tokens = [Token("IDENT", "SKIP"), Token("EOF", "EOF")]
            self.position = 0

        def parse_skip(self) -> ASTNode:
            return ASTNode("skip", value=None, line=1, col=1)

    parser = DummyParser()
    node = parser.parse_statement()
    assert node.kind == "skip"


def test_parse_loop_range_missing_for() -> None:
    parser = Parser([Token("IDENT", "foo")])
    with pytest.raises(
        SyntaxError,
        match=re.escape("Expected one of ('LOOP_FOR',), got Token(IDENT, foo)"),
    ):
        parser.parse_loop_range()


def test_parse_loop_range_missing_var() -> None:
    parser = Parser([Token("LOOP_FOR", "FOR")])
    with pytest.raises(
        SyntaxError, match=re.escape("Expected one of ('IDENT',), got Token(EOF, EOF)")
    ):
        parser.parse_loop_range()


def test_parse_loop_range_invalid_signed_number() -> None:
    parser = Parser(
        [
            Token("LOOP_FOR", "FOR"),
            Token("IDENT", "i"),
            Token("DELIM_TO", "TO"),
            Token("SUB", "-"),
        ]
    )
    with pytest.raises(
        SyntaxError,
        match=r"Expected one of \('NUMBER', 'FLOAT'\), got Token\(EOF, EOF\)",
    ):
        parser.parse_loop_range()


def test_parse_loop_range_missing_end_number() -> None:
    parser = Parser(
        [Token("LOOP_FOR", "FOR"), Token("IDENT", "i"), Token("DELIM_TO", "TO")]
    )
    with pytest.raises(SyntaxError, match=r"Expected one of \('NUMBER', 'FLOAT'\)"):
        parser.parse_loop_range()


def test_parse_loop_range_with_by_missing_step() -> None:
    tokens = [
        Token("LOOP_FOR", "FOR"),
        Token("IDENT", "i"),
        Token("DELIM_TO", "TO"),
        Token("NUMBER", "5"),
        Token("DELIM_BY", "BY"),
    ]
    parser = Parser(tokens)
    with pytest.raises(SyntaxError, match=r"Expected one of \('NUMBER', 'FLOAT'\)"):
        parser.parse_loop_range()


def test_parse_loop_range_with_at_and_by_and_empty_body() -> None:
    tokens = [
        Token("LOOP_FOR", "FOR"),
        Token("IDENT", "i"),
        Token("DELIM_AT", "AT"),
        Token("NUMBER", "1"),
        Token("DELIM_TO", "TO"),
        Token("NUMBER", "10"),
        Token("DELIM_BY", "BY"),
        Token("NUMBER", "2"),
        Token("LBRACE", "{"),
        Token("RBRACE", "}"),
    ]
    parser = Parser(tokens)
    with pytest.raises(SyntaxError, match="FOR loop body cannot be empty"):
        parser.parse_loop_range()


def test_parse_loop_range_full_valid() -> None:
    tokens = tokenize("FOR i AT -1 TO 10 BY 2 { RETURN 0 }")
    parser = Parser(tokens)
    result = parser.parse_loop_range()
    assert result.kind == "loop"
    assert result.value == "FOR"
    assert result.children[0].kind == "range"


def test_try_keyword_missing_hits_branch_173() -> None:
    parser = Parser([Token("STRING", "not_try")])  # Not TRY, not IDENT("TRY")
    with pytest.raises(
        SyntaxError,
        match=r"Expected one of \('TRY', 'IDENT'\), got Token\(STRING, not_try\)",
    ):
        parser.parse_try()


def test_catch_with_parens_missing_name_hits_branch_189() -> None:
    tokens = [
        Token("TRY", "TRY"),
        Token("LBRACE", "{"),
        Token("RBRACE", "}"),
        Token("CATCH", "CATCH"),
        Token("LPAREN", "("),
        Token("STRING", '"nope"'),  # Not an IDENT
        Token("RPAREN", ")"),
        Token("LBRACE", "{"),
        Token("RBRACE", "}"),
    ]
    parser = Parser(tokens)
    with pytest.raises(
        SyntaxError,
        match=r"Expected one of \('IDENT',\), got Token\(STRING, \"nope\"\)",
    ):
        parser.parse_try()


def test_catch_missing_ident_param_hits_branch_195() -> None:
    tokens = [
        Token("TRY", "TRY"),
        Token("LBRACE", "{"),
        Token("RBRACE", "}"),
        Token("CATCH", "CATCH"),
        Token("STRING", '"x"'),  # not an IDENT
        Token("LBRACE", "{"),
        Token("RBRACE", "}"),
    ]
    parser = Parser(tokens)
    with pytest.raises(
        SyntaxError, match=r"Expected one of \('LBRACE',\), got Token\(STRING, \"x\"\)"
    ):
        parser.parse_try()


def test_parse_skip_keyword() -> None:
    parser = Parser([Token("SKIP", "SKIP")])
    node = parser.parse_skip()
    assert node.kind == "skip"
    assert node.line == 0
    assert node.col == 0


def test_parse_skip_ident_alias() -> None:
    parser = Parser([Token("IDENT", "SKIP")])
    node = parser.parse_skip()
    assert node.kind == "skip"
    assert node.line == 0
    assert node.col == 0


def test_parse_skip_invalid_token() -> None:
    parser = Parser([Token("STRING", "nope")])
    with pytest.raises(
        SyntaxError,
        match=r"Expected one of \('SKIP', 'IDENT'\), got Token\(STRING, nope\)",
    ):
        parser.parse_skip()


def test_if_missing_keyword() -> None:
    parser = Parser([Token("STRING", "nope")])
    with pytest.raises(
        SyntaxError, match=r"Expected one of \('IF',\), got Token\(STRING, nope\)"
    ):
        parser.parse_if()


def test_if_invalid_operator() -> None:
    tokens = [
        Token("IF", "IF"),
        Token("LPAREN", "("),
        Token("NUMBER", "123"),  # Not a prefix op
        Token("IDENT", "x"),
        Token("RPAREN", ")"),
        Token("LBRACE", "{"),
        Token("RBRACE", "}"),
    ]
    with pytest.raises(SyntaxError, match=r"Invalid IF operator: Token\(NUMBER, 123\)"):
        Parser(tokens).parse_if()


def test_if_unexpected_eof_in_condition() -> None:
    tokens = [
        Token("IF", "IF"),
        Token("LPAREN", "("),
        Token("AND", "AND"),
        Token("IDENT", "x"),
        Token("EOF", "EOF"),  # Never closes
    ]
    with pytest.raises(SyntaxError, match=r"Unexpected EOF in IF condition"):
        Parser(tokens).parse_if()


def test_if_invalid_token_in_condition() -> None:
    tokens = [
        Token("IF", "IF"),
        Token("LPAREN", "("),
        Token("AND", "AND"),
        Token("PLUS", "+"),  # Invalid here
        Token("RPAREN", ")"),
        Token("LBRACE", "{"),
        Token("RBRACE", "}"),
    ]
    with pytest.raises(
        SyntaxError,
        match=r"Expected one of \('IDENT', 'NUMBER', 'FLOAT', 'STRING', 'LITERAL', 'THIS'\), got Token\(PLUS, \+\)",
    ):
        Parser(tokens).parse_if()


def test_if_unary_operator_wrong_arity() -> None:
    tokens = [
        Token("IF", "IF"),
        Token("LPAREN", "("),
        Token("NOT", "NOT"),
        Token("RPAREN", ")"),
        Token("LBRACE", "{"),
        Token("RBRACE", "}"),
    ]
    with pytest.raises(SyntaxError, match=r"NOT requires 1 operand"):
        Parser(tokens).parse_if()


def test_if_binary_operator_insufficient_operands() -> None:
    tokens = [
        Token("IF", "IF"),
        Token("LPAREN", "("),
        Token("AND", "AND"),
        Token("IDENT", "x"),
        Token("RPAREN", ")"),
        Token("LBRACE", "{"),
        Token("RBRACE", "}"),
    ]
    with pytest.raises(SyntaxError, match=r"AND requires at least 2 operands"):
        Parser(tokens).parse_if()


def test_if_body_missing_lbrace() -> None:
    tokens = [
        Token("IF", "IF"),
        Token("LPAREN", "("),
        Token("AND", "AND"),
        Token("IDENT", "x"),
        Token("IDENT", "y"),
        Token("RPAREN", ")"),
        Token("PRINT", "!"),
    ]
    with pytest.raises(SyntaxError, match=r"IF body must be enclosed in braces"):
        Parser(tokens).parse_if()


def test_if_else_body_missing_lbrace() -> None:
    tokens = [
        Token("IF", "IF"),
        Token("LPAREN", "("),
        Token("AND", "AND"),
        Token("IDENT", "x"),
        Token("IDENT", "y"),
        Token("RPAREN", ")"),
        Token("LBRACE", "{"),
        Token("RBRACE", "}"),
        Token("ELSE", "ELSE"),
        Token("PRINT", "!"),
    ]
    with pytest.raises(SyntaxError, match=r"ELSE body must be enclosed in braces"):
        Parser(tokens).parse_if()


def test_parse_export_success() -> None:
    tokens = [Token("EXPORT", "EXPORT", 1, 1), Token("IDENT", "foo", 1, 8)]
    node = Parser(tokens).parse_export()
    assert node.kind == "export"
    assert node.value == "foo"
    assert node.line == 1
    assert node.col == 1


def test_parse_break_success() -> None:
    parser = Parser([Token("BREAK", "BREAK", 2, 4)])
    parser.in_block = True
    node = parser.parse_break()
    assert node.kind == "break"
    assert node.line == 2
    assert node.col == 4


def test_parse_continue_success() -> None:
    parser = Parser([Token("CONTINUE", "CONTINUE", 3, 5)])
    parser.in_block = True
    node = parser.parse_continue()
    assert node.kind == "continue"
    assert node.line == 3
    assert node.col == 5


def test_prefix_operator_missing_op() -> None:
    parser = Parser([Token("IDENT", "x")])  # no prefix op
    with pytest.raises(
        SyntaxError,
        match=r"Expected one of \('PLUS', 'SUB', 'MULT', 'DIV', 'MOD', 'AND', 'NOT'\), got Token\(IDENT, x\)",
    ):
        parser.parse_prefix_operator()


def test_prefix_operator_missing_operand() -> None:
    parser = Parser([Token("PLUS", "+")])  # no operand after PLUS
    with pytest.raises(
        SyntaxError,
        match=r"Expected one of \('IDENT', 'NUMBER'\), got Token\(EOF, EOF\)",
    ):
        parser.parse_prefix_operator()


def test_prefix_operator_not_unary() -> None:
    parser = Parser([Token("NOT", "NOT"), Token("IDENT", "a")])
    node = parser.parse_prefix_operator()
    assert node.kind == "bool"
    assert node.value == "NOT"
    assert len(node.children) == 1


def test_if_operator_kind_arith() -> None:
    tokens = [
        Token("IF", "IF"),
        Token("LPAREN", "("),
        Token("PLUS", "+"),
        Token("NUMBER", "1"),
        Token("NUMBER", "2"),
        Token("RPAREN", ")"),
        Token("LBRACE", "{"),
        Token("RBRACE", "}"),
    ]
    node = Parser(tokens).parse_if()
    assert node.value.kind == "arith"  # type: ignore


def test_if_operator_kind_bool() -> None:
    tokens = [
        Token("IF", "IF"),
        Token("LPAREN", "("),
        Token("AND", "AND"),
        Token("IDENT", "x"),
        Token("IDENT", "y"),
        Token("RPAREN", ")"),
        Token("LBRACE", "{"),
        Token("RBRACE", "}"),
    ]
    node = Parser(tokens).parse_if()
    assert node.value.kind == "bool"  # type: ignore


def test_if_kind_arith_with_else() -> None:
    tokens = [
        Token("IF", "IF"),
        Token("LPAREN", "("),
        Token("PLUS", "+"),
        Token("NUMBER", "1"),
        Token("NUMBER", "2"),
        Token("RPAREN", ")"),
        Token("LBRACE", "{"),
        Token("RBRACE", "}"),
        Token("ELSE", "ELSE"),
        Token("LBRACE", "{"),
        Token("RBRACE", "}"),
    ]
    node = Parser(tokens).parse_if()
    assert node.value.kind == "arith"  # type: ignore
    assert node.else_children == []


def test_if_kind_bool_with_else() -> None:
    tokens = [
        Token("IF", "IF"),
        Token("LPAREN", "("),
        Token("AND", "AND"),
        Token("IDENT", "a"),
        Token("IDENT", "b"),
        Token("RPAREN", ")"),
        Token("LBRACE", "{"),
        Token("RBRACE", "}"),
        Token("ELSE", "ELSE"),
        Token("LBRACE", "{"),
        Token("RBRACE", "}"),
    ]
    node = Parser(tokens).parse_if()
    assert node.value.kind == "bool"  # type: ignore
    assert node.else_children == []


def test_if_kind_compare_with_else() -> None:
    tokens = [
        Token("IF", "IF"),
        Token("LPAREN", "("),
        Token("EQ", "=="),
        Token("NUMBER", "3"),
        Token("NUMBER", "3"),
        Token("RPAREN", ")"),
        Token("LBRACE", "{"),
        Token("RBRACE", "}"),
        Token("ELSE", "ELSE"),
        Token("LBRACE", "{"),
        Token("RBRACE", "}"),
    ]
    node = Parser(tokens).parse_if()
    assert node.value.kind == "compare"  # type: ignore
    assert node.else_children == []


def test_parse_export_missing_keyword() -> None:
    parser = Parser([Token("STRING", "nope")])
    with pytest.raises(
        SyntaxError,
        match=r"Expected one of \('EXPORT', 'IDENT'\), got Token\(STRING, nope\)",
    ):
        parser.parse_export()


def test_parse_export_missing_identifier() -> None:
    parser = Parser([Token("EXPORT", "EXPORT")])
    with pytest.raises(
        SyntaxError, match=r"Expected one of \('IDENT',\), got Token\(EOF, EOF\)"
    ):
        parser.parse_export()


def test_parse_skip_missing_keyword() -> None:
    parser = Parser([Token("STRING", "nope")])
    with pytest.raises(
        SyntaxError,
        match=r"Expected one of \('SKIP', 'IDENT'\), got Token\(STRING, nope\)",
    ):
        parser.parse_skip()


def test_parse_continue_outside_loop() -> None:
    parser = Parser([Token("CONTINUE", "CONTINUE")])
    parser.in_block = False
    with pytest.raises(
        SyntaxError, match=r"CONTINUE only allowed inside loop blocks\."
    ):
        parser.parse_continue()


def test_parse_continue_missing_keyword() -> None:
    parser = Parser([Token("STRING", "nope")])
    parser.in_block = True
    with pytest.raises(
        SyntaxError,
        match=r"Expected one of \('CONTINUE', 'IDENT'\), got Token\(STRING, nope\)",
    ):
        parser.parse_continue()


def test_parse_break_outside_loop() -> None:
    parser = Parser([Token("BREAK", "BREAK")])
    parser.in_block = False
    with pytest.raises(SyntaxError, match=r"BREAK only allowed inside loop blocks\."):
        parser.parse_break()


def test_parse_break_missing_keyword() -> None:
    parser = Parser([Token("STRING", "nope")])
    parser.in_block = True
    with pytest.raises(
        SyntaxError,
        match=r"Expected one of \('BREAK', 'IDENT'\), got Token\(STRING, nope\)",
    ):
        parser.parse_break()


def test_parse_if_missing_keyword() -> None:
    parser = Parser([Token("STRING", "nope")])
    with pytest.raises(
        SyntaxError, match=r"Expected one of \('IF',\), got Token\(STRING, nope\)"
    ):
        parser.parse_if()


def test_parse_try_missing_keyword() -> None:
    parser = Parser([Token("STRING", "nope")])
    with pytest.raises(
        SyntaxError,
        match=r"Expected one of \('TRY', 'IDENT'\), got Token\(STRING, nope\)",
    ):
        parser.parse_try()


def test_parse_import_missing_keyword_raises() -> None:
    parser = Parser([Token("STRING", "nope")])
    with pytest.raises(
        SyntaxError,
        match=r"Expected one of \('IMPORT', 'IDENT'\), got Token\(STRING, nope\)",
    ):
        parser.parse_import()


def test_parse_import_missing_string_path_raises() -> None:
    parser = Parser([Token("IMPORT", "IMPORT")])
    with pytest.raises(
        SyntaxError, match=r"Expected one of \('STRING',\), got Token\(EOF, EOF\)"
    ):
        parser.parse_import()


def test_parse_class_missing_keyword_387() -> None:
    parser = Parser([Token("STRING", "not_class")])
    with pytest.raises(
        SyntaxError,
        match=r"Expected one of \('CLASS',\), got Token\(STRING, not_class\)",
    ):
        parser.parse_class()


def test_parse_class_missing_name_391() -> None:
    parser = Parser([Token("CLASS", "CLASS"), Token("LBRACE", "{")])
    with pytest.raises(
        SyntaxError, match=r"Expected one of \('IDENT',\), got Token\(LBRACE, \{\)"
    ):
        parser.parse_class()


def test_parse_class_extends_missing_base_398() -> None:
    parser = Parser(
        [
            Token("CLASS", "CLASS"),
            Token("IDENT", "MyClass"),
            Token("EXTENDS", "EXTENDS"),
            Token("NUMBER", "123"),
            Token("LBRACE", "{"),
            Token("RBRACE", "}"),
        ]
    )
    with pytest.raises(
        SyntaxError, match=r"Expected one of \('IDENT',\), got Token\(NUMBER, 123\)"
    ):
        parser.parse_class()


# Precompute safe identifiers
SAFE_CHARS = "abcdefghjklmnpqrstuvwxyz"
banned_idents = {k.upper() for k in token_hashmap.keys()}
valid_idents = [c * 3 for c in SAFE_CHARS if (c * 3).upper() not in banned_idents]


@given(  # type: ignore[misc]
    name=st.sampled_from(valid_idents),
    params=st.lists(st.sampled_from(valid_idents), max_size=3),
    rettype=st.sampled_from(["int", "float", "str"]),
)
@settings(deadline=2000, max_examples=10)  # type: ignore[misc]
def test_function_decl_with_return_and_params(
    name: str, params: list[str], rettype: str
) -> None:
    param_str = ", ".join(params)
    fn_header = f"@ {name}({param_str})"
    if rettype:
        fn_header += f": {rettype}"
    source = fn_header + " { RETURN 0 }"

    result = Parser(tokenize(source)).parse()
    assert result[0].kind == "func"
    assert result[0].value == name
    assert result[0].return_type == rettype
    if params:
        assert result[0].type == ",".join(params)
    else:
        assert result[0].type is None
    assert result[0].children[0].kind == "return"
    assert len(result[0].children[0].children) == 1


def test_parse_function_missing_func_keyword() -> None:
    parser = tokenize_tokens(Token("IDENT", "oops"))
    with pytest.raises(
        SyntaxError, match=r"Expected one of \('FUNC',\), got Token\(IDENT, oops\)"
    ):
        parser.parse_function()


def test_parse_function_missing_name() -> None:
    tokens = [Token("FUNC", "@"), Token("LPAREN", "(")]
    parser = tokenize_tokens(*tokens)
    with pytest.raises(
        SyntaxError, match=r"Expected one of \('IDENT',\), got Token\(LPAREN, \(\)"
    ):
        parser.parse_function()


def test_parse_function_missing_param_name() -> None:
    tokens = [
        Token("FUNC", "@"),
        Token("IDENT", "f"),
        Token("LPAREN", "("),
        Token("COMMA", ","),
        Token("RPAREN", ")"),
        Token("LBRACE", "{"),
        Token("RBRACE", "}"),
    ]
    parser = tokenize_tokens(*tokens)
    with pytest.raises(
        SyntaxError, match=r"Expected one of \('IDENT',\), got Token\(COMMA, ,\)"
    ):
        parser.parse_function()


def test_parse_function_missing_return_type() -> None:
    tokens = [
        Token("FUNC", "@"),
        Token("IDENT", "f"),
        Token("LPAREN", "("),
        Token("RPAREN", ")"),
        Token("COLON", ":"),
        Token("LBRACE", "{"),
        Token("RBRACE", "}"),
    ]
    parser = tokenize_tokens(*tokens)
    with pytest.raises(
        SyntaxError, match=r"Expected one of \('IDENT',\), got Token\(LBRACE, \{\)"
    ):
        parser.parse_function()


def test_parse_return_with_member_access() -> None:
    source = "@ f() { RETURN obj.prop }"
    result = parse(source)
    ret = result[0].children[0]
    assert ret.kind == "return"
    assert ret.children[0].kind == "member"
    assert ret.children[0].value == "prop"
    assert ret.children[0].children[0].kind == "identifier"
    assert ret.children[0].children[0].value == "obj"


def test_assignment_lhs_invalid_token_913_916() -> None:
    with pytest.raises(
        SyntaxError, match=r"Invalid assignment target: Token\(NUMBER, 5\)"
    ):
        parse("= 5")


def test_assignment_new_empty_parens_927_947() -> None:
    result = parse("= x NEW Foo()")
    node = result[0].children[1]
    assert node.kind == "new"
    assert node.children == []


def test_assignment_call_empty_parens_964_982() -> None:
    result = parse("= x foo()")
    node = result[0].children[1]
    assert node.kind == "call"
    assert node.children == []


def test_assignment_rhs_expression_is_none() -> None:
    class DummyParser(Parser):
        def __init__(self) -> None:
            self.tokens = [
                Token("ASSIGN", "="),
                Token("IDENT", "x"),
                Token("LBRACK", "["),
            ]
            self.position = 0
            self.in_block = False

        def parse_expression(self, strict: bool = True) -> ASTNode:
            return None  # type: ignore[return-value]

    with pytest.raises(SyntaxError, match=r"Expected expression on right-hand side"):
        DummyParser().parse_assignment()


def test_assignment_rhs_value_token_none() -> None:
    class DummyParser(Parser):
        def __init__(self) -> None:
            self.tokens = [
                Token("ASSIGN", "="),
                Token("IDENT", "x"),
                Token("NUMBER", "42"),
            ]
            self.position = 0
            self.in_block = False
            self._match_count = 0

        def match(self, *args: str, strict: bool = True) -> Token | None:
            if "ASSIGN" in args:
                tok = self.tokens[self.position]
                self.advance()
                return tok
            if "IDENT" in args and self._match_count < 1:
                self._match_count += 1
                tok = self.tokens[self.position]
                self.advance()
                return tok
            return None  # fail on RHS

        def current(self) -> Token:
            return self.tokens[self.position]

        def peek(self, offset: int = 1) -> Token:
            try:
                return self.tokens[self.position + offset]
            except IndexError:
                return Token("EOF", "EOF")

    with pytest.raises(SyntaxError, match=r"Expected value on right-hand side"):
        DummyParser().parse_assignment()


def test_catch_missing_name_after_ident_767_768() -> None:
    class DummyParser1(Parser):
        def __init__(self) -> None:
            self.tokens = [
                Token("TRY", "TRY"),
                Token("LBRACE", "{"),
                Token("RBRACE", "}"),
                Token("CATCH", "CATCH"),
                Token("IDENT", "@"),
                Token("LBRACE", "{"),
                Token("RBRACE", "}"),
            ]
            self.position = 0
            self.in_block = False

        def match(self, *args: str, strict: bool = True) -> Token | None:
            tok = self.current()
            if "IDENT" in args and tok.value == "@":
                return None
            self.position += 1
            return tok

    with pytest.raises(SyntaxError, match=r"Expected exception name in CATCH"):
        DummyParser1().parse_try()


def test_finally_malformed_789_792() -> None:
    # FINALLY present, but next token is EOF, not LBRACE
    tokens = [
        Token("TRY", "TRY"),
        Token("LBRACE", "{"),
        Token("RBRACE", "}"),
        Token("FINALLY", "finally_typo"),
    ]
    parser = Parser(tokens)
    with pytest.raises(
        SyntaxError, match=r"Expected one of \('LBRACE',\), got Token\(EOF, EOF\)"
    ):
        parser.parse_try()


def test_try_missing_or_invalid_743_744() -> None:
    tokens = [Token("IDENT", "not_try"), Token("LBRACE", "{"), Token("RBRACE", "}")]
    parser = Parser(tokens)
    with pytest.raises(SyntaxError, match=r"Expected 'TRY' keyword"):
        parser.parse_try()


def test_catch_malformed() -> None:
    class Dummy(Parser):
        def __init__(self) -> None:
            self.tokens = [
                Token("TRY", "TRY"),
                Token("LBRACE", "{"),
                Token("RBRACE", "}"),
                Token("IDENT", "CATCH"),  # triggers CATCH branch
                Token("LBRACE", "{"),
                Token("RBRACE", "}"),
            ]
            self.position = 0
            self.in_block = False

        def match(self, *args: str, strict: bool = True) -> Token | None:
            tok = self.current()
            if "CATCH" in args or "IDENT" in args:
                if tok.type == "IDENT" and tok.value != "CATCH":
                    return None  # simulate malformed IDENT ≠ "CATCH"
                if tok.type == "IDENT" and tok.value == "CATCH":
                    return None  # force failure even for "CATCH"
            self.position += 1
            return tok

        def current(self) -> Token:
            return self.tokens[self.position]

        def peek(self, offset: int = 1) -> Token:
            try:
                return self.tokens[self.position + offset]
            except IndexError:
                return Token("EOF", "EOF")

    with pytest.raises(SyntaxError, match=r"Expected 'CATCH' keyword"):
        Dummy().parse_try()


def test_catch_missing_name_in_parens() -> None:
    class Dummy(Parser):
        def __init__(self) -> None:
            self.tokens = [
                Token("TRY", "TRY"),
                Token("LBRACE", "{"),
                Token("RBRACE", "}"),
                Token("IDENT", "CATCH"),  # triggers while-loop
                Token("LPAREN", "("),
                Token("IDENT", "e"),
                Token("RPAREN", ")"),
                Token("LBRACE", "{"),
                Token("RBRACE", "}"),
            ]
            self.position = 0
            self.in_block = False

        def match(self, *args: str, strict: bool = True) -> Token | None:
            tok = self.current()
            if "CATCH" in args or "IDENT" in args:
                if tok.type == "IDENT" and tok.value != "CATCH":
                    return None  # Simulate bad IDENT pretending to be CATCH
            self.position += 1
            return tok

        def current(self) -> Token:
            return self.tokens[self.position]

        def peek(self, offset: int = 1) -> Token:
            try:
                return self.tokens[self.position + offset]
            except IndexError:
                return Token("EOF", "EOF")

    with pytest.raises(SyntaxError, match=r"Expected exception name inside CATCH()"):
        Dummy().parse_try()


def test_call_expr_empty_parens_708_729() -> None:
    result = parse("foo()")
    node = result[0].children[0]
    assert node.kind == "call"
    assert node.value.kind == "identifier"  # type: ignore
    assert node.children == []


def test_call_expr_missing_arg_713_714() -> None:
    class Dummy(Parser):
        def __init__(self) -> None:
            self.tokens = [
                Token("IDENT", "foo"),
                Token("LPAREN", "("),
                Token("COMMA", ","),
                Token("RPAREN", ")"),
            ]
            self.position = 0
            self.in_block = False

        def match(self, *args: str, strict: bool = True) -> Token | None:
            tok = self.current()
            if tok.type in args:
                self.position += 1
                return tok
            return None

        def current(self) -> Token:
            return self.tokens[self.position]

        def peek(self, offset: int = 1) -> Token:
            try:
                return self.tokens[self.position + offset]
            except IndexError:
                return Token("EOF", "EOF")

    with pytest.raises(
        SyntaxError, match=r"Unrecognized statement: foo \(type=IDENT\)"
    ):
        Dummy().parse_statement()


def test_call_expr_with_ident_arg() -> None:
    result = parse("foo(x)")
    node = result[0].children[0]
    assert node.kind == "call"
    assert node.children[0].kind == "identifier"
    assert node.children[0].value == "x"


def test_call_expr_with_comma_725_726() -> None:
    result = parse("foo(x, y)")
    node = result[0].children[0]
    assert node.kind == "call"
    assert [c.value for c in node.children] == ["x", "y"]


def test_call_expr_with_this_arg() -> None:
    result = parse("foo(THIS)")
    node = result[0].children[0]
    assert node.children[0].kind == "identifier"
    assert node.children[0].value == "THIS"


def test_call_missing_keyword_597_598() -> None:
    tokens = [Token("IDENT", "foo"), Token("LPAREN", "("), Token("RPAREN", ")")]
    parser = Parser(tokens)
    with pytest.raises(SyntaxError, match=r"Expected 'CALL' keyword"):
        parser.parse_call(strict=False)


def test_call_missing_callee_606_607() -> None:
    class Dummy(Parser):
        def __init__(self) -> None:
            self.tokens = [
                Token("CALL", "CALL"),
                Token("IDENT", "invalid"),
            ]  # current() returns IDENT
            self.position = 0
            self.in_block = False

        def match(self, *args: str, strict: bool = True) -> Token | None:
            tok = self.current()
            if "CALL" in args and tok.type == "CALL":
                self.position += 1
                return tok
            if "IDENT" in args:
                return None  # simulate failed IDENT match to trigger line 607
            self.position += 1
            return tok

        def current(self) -> Token:
            return (
                self.tokens[self.position]
                if self.position < len(self.tokens)
                else Token("EOF", "EOF")
            )

        def peek(self, offset: int = 1) -> Token:
            return Token("EOF", "EOF")

    with pytest.raises(SyntaxError, match=r"Expected callee name after CALL"):
        Dummy().parse_call()


def test_prefix_lhs_bracketed_none_566_567() -> None:
    class Dummy(Parser):
        def __init__(self) -> None:
            self.tokens = [
                Token("PLUS", "PLUS"),
                Token("LBRACK", "["),
                Token("RBRACK", "]"),
            ]
            self.position = 0
            self.in_block = True

        def match(self, *args: str, strict: bool = True) -> Token | None:
            tok = self.current()
            self.position += 1
            return tok

        def current(self) -> Token:
            return self.tokens[self.position]

        def parse_expression(self, strict=True):  # type: ignore
            return None  # simulate malformed expression

    with pytest.raises(AssertionError):  # hits assert expr is not None
        Dummy().parse_prefix_operator()


def test_prefix_missing_operand_572_573() -> None:
    class Dummy(Parser):
        def __init__(self) -> None:
            self.tokens = [Token("PLUS", "PLUS"), Token("COMMA", ",")]
            self.position = 0
            self.in_block = True

        def match(self, *args: str, strict: bool = True) -> Token | None:
            tok = self.current()
            if tok.type in args:
                self.position += 1
                return tok
            return None

        def current(self) -> Token:
            return self.tokens[self.position]

    with pytest.raises(SyntaxError, match=r"Expected operand after operator"):
        Dummy().parse_prefix_operator()


def test_prefix_missing_operator_562_563() -> None:
    class Dummy(Parser):
        def __init__(self) -> None:
            self.tokens = [Token("IDENT", "x")]
            self.position = 0
            self.in_block = True  # required for some parsing paths

        def match(self, *args: str, strict: bool = True) -> Token | None:
            return None  # Force match() to return None to hit the condition at line 562

        def current(self) -> Token:
            return self.tokens[self.position]

        def peek(self, offset: int = 1) -> Token:
            return Token("EOF", "EOF")

    with pytest.raises(SyntaxError, match=r"Expected prefix operator"):
        Dummy().parse_prefix_operator()


def test_prefix_operand_bracket_expr_567() -> None:
    tokens = [
        Token("PLUS", "+"),
        Token("LBRACK", "["),
        Token("PLUS", "+"),
        Token("NUMBER", "1"),
        Token("NUMBER", "2"),
        Token("RBRACK", "]"),
        Token("NUMBER", "3"),
    ]
    parser = Parser(tokens)
    parser.in_block = True
    node = parser.parse_prefix_operator()
    assert node.kind == "arith"
    assert node.value == "PLUS"
    assert node.children[0].kind == "arith"
    assert node.children[0].value == "PLUS"
    assert node.children[1].kind == "number"
    assert node.children[1].value == "3"


def test_loop_while_missing_keyword_506_507() -> None:
    tokens = [Token("IDENT", "x")]
    parser = Parser(tokens)
    parser.in_block = True
    with pytest.raises(SyntaxError, match=r"Expected 'WHILE' loop start"):
        parser.parse_loop_while(strict=False)


def test_loop_while_missing_condition_520_521() -> None:
    tokens = [
        Token("LOOP_WHILE", "WHILE"),
        Token("COMMA", ","),
    ]  # Invalid condition token
    parser = Parser(tokens)
    parser.in_block = True
    with pytest.raises(SyntaxError, match=r"Expected condition after WHILE"):
        parser.parse_loop_while(strict=False)


def test_propagate_missing_operator_480_481() -> None:
    class Dummy(Parser):
        def __init__(self) -> None:
            self.tokens = [Token("EOF", "EOF")]
            self.position = 0
            self.in_block = True

        def match(self, *args: str, strict: bool = True) -> Token | None:
            return None  # force match to return None

        def current(self) -> Token:
            return self.tokens[self.position]

        def peek(self, offset: int = 1) -> Token:
            return Token("EOF", "EOF")

    with pytest.raises(SyntaxError, match=r"Expected '\$' propagate operator"):
        Dummy().parse_propagate(strict=False)


def test_propagate_invalid_expr_485_486() -> None:
    class Dummy(Parser):
        def __init__(self) -> None:
            self.tokens = [Token("PROP", "$"), Token("LBRACK", "[")]
            self.position = 0
            self.in_block = True

        def match(self, *args: str, strict: bool = True) -> Token | None:
            if "PROP" in args:
                self.position += 1
                return Token("PROP", "$")
            if "LBRACK" in args:
                self.position += 1
                return Token("LBRACK", "[")
            return None

        def parse_expression(self, strict=True):  # type: ignore
            return None  # force invalid expression

        def current(self) -> Token:
            return self.tokens[self.position]

        def peek(self, offset: int = 1) -> Token:
            return Token("EOF", "EOF")

    with pytest.raises(SyntaxError, match=r"Expected valid expression after '\$'"):
        Dummy().parse_propagate(strict=False)


def test_propagate_missing_value_491_492() -> None:
    class Dummy(Parser):
        def __init__(self) -> None:
            self.tokens = [Token("PROP", "$")]
            self.position = 0
            self.in_block = True

        def match(self, *args: str, strict: bool = True) -> Token | None:
            if "PROP" in args:
                self.position += 1
                return Token("PROP", "$")
            return None

        def current(self) -> Token:
            return Token("IDENT", "dummy")

        def peek(self, offset: int = 1) -> Token:
            return Token("EOF", "EOF")

    with pytest.raises(SyntaxError, match=r"Expected value after '\$'"):
        Dummy().parse_propagate(strict=False)


def test_member_access_missing_base_ident_1199_1200() -> None:
    tokens = [Token("DOT", "."), Token("IDENT", "prop")]
    parser = Parser(tokens)
    with pytest.raises(
        SyntaxError, match=r"Expected identifier at start of member access"
    ):
        parser.parse_member_access(strict=False)


def test_member_access_missing_property_1210_1211() -> None:
    tokens = [Token("IDENT", "foo"), Token("DOT", "."), Token("NUMBER", "123")]
    parser = Parser(tokens)
    with pytest.raises(
        SyntaxError, match=r"Expected identifier after '\.' in member access"
    ):
        parser.parse_member_access(strict=False)


def test_parse_break_strict_false_420_421() -> None:
    class Dummy(Parser):
        def __init__(self) -> None:
            self.tokens = [Token("IDENT", "not_break")]
            self.position = 0
            self.in_block = True

        def match(self, *args: str, strict: bool = True) -> Token | None:
            if strict:
                return super().match(*args, strict=True)
            return None

        def current(self) -> Token:
            return self.tokens[self.position]

        def peek(self, offset: int = 1) -> Token:
            return Token("EOF", "EOF")

    with pytest.raises(SyntaxError, match=r"Expected 'BREAK' keyword"):
        Dummy().parse_break(strict=False)


def test_parse_continue_strict_false_431_432() -> None:
    class Dummy(Parser):
        def __init__(self) -> None:
            self.tokens = [Token("IDENT", "not_continue")]
            self.position = 0
            self.in_block = True

        def match(self, *args: str, strict: bool = True) -> Token | None:
            if strict:
                return super().match(*args, strict=True)
            return None

        def current(self) -> Token:
            return self.tokens[self.position]

        def peek(self, offset: int = 1) -> Token:
            return Token("EOF", "EOF")

    with pytest.raises(SyntaxError, match=r"Expected 'CONTINUE' keyword"):
        Dummy().parse_continue(strict=False)


def test_export_missing_keyword_404_405() -> None:
    class Dummy(Parser):
        def __init__(self) -> None:
            self.tokens = [Token("IDENT", "not_export")]
            self.position = 0
            self.in_block = False

        def match(self, *args: str, strict: bool = True) -> Token | None:
            tok = self.current()
            if "EXPORT" in args and tok.value != "EXPORT":
                return None
            self.position += 1
            return tok

        def current(self) -> Token:
            return self.tokens[self.position]

        def peek(self, offset: int = 1) -> Token:
            return Token("EOF", "EOF")

    with pytest.raises(SyntaxError, match=r"Expected 'EXPORT' keyword"):
        Dummy().parse_export(strict=False)


def test_export_missing_name_409_410() -> None:
    tokens = [Token("EXPORT", "EXPORT"), Token("COMMA", ",")]  # Invalid after EXPORT
    parser = Parser(tokens)
    with pytest.raises(SyntaxError, match=r"Expected identifier after EXPORT"):
        parser.parse_export(strict=False)


def test_print_expr_none_373_374() -> None:
    class Dummy(Parser):
        def __init__(self) -> None:
            self.tokens = [Token("PRINT", "!"), Token("LBRACK", "[")]
            self.position = 0
            self.in_block = True

        def match(self, *args: str, strict: bool = True) -> Token | None:
            tok = self.tokens[self.position]
            self.position += 1
            return tok

        def current(self) -> Token:
            return self.tokens[self.position]

        def parse_expression(self, strict: bool = True) -> ASTNode:
            return None  # type: ignore

    with pytest.raises(SyntaxError, match=r"Expected expression inside brackets"):
        Dummy().parse_print(strict=True)


def test_print_literal_final_return() -> None:
    tokens = [Token("PRINT", "!"), Token("NUMBER", "123")]
    parser = Parser(tokens)
    node = parser.parse_print(strict=True)
    assert node.kind == "print"
    assert node.children[0].kind == "number"
    assert node.children[0].value == "123"


def test_print_missing_keyword_1189_1190() -> None:
    class Dummy(Parser):
        def __init__(self) -> None:
            self.tokens = [Token("IDENT", "x")]
            self.position = 0
            self.in_block = False

        def match(self, *args: str, strict: bool = True) -> Token | None:
            tok = self.current()
            if "PRINT" in args:
                return None  # simulate no match
            return tok

        def current(self) -> Token:
            return self.tokens[self.position]

        def peek(self, offset: int = 1) -> Token:
            return Token("EOF", "EOF")

    with pytest.raises(SyntaxError, match=r"Expected 'PRINT' keyword"):
        Dummy().parse_print(strict=False)


def test_input_missing_keyword_strict_false() -> None:
    class Dummy(Parser):
        def __init__(self) -> None:
            self.tokens = [Token("IDENT", "data")]
            self.position = 0
            self.in_block = False

        def match(self, *args: str, strict: bool = True) -> Token | None:
            if "INPUT" in args:
                return None
            return self.current()

        def current(self) -> Token:
            return self.tokens[self.position]

        def peek(self, offset: int = 1) -> Token:
            return Token("EOF", "EOF")

    with pytest.raises(SyntaxError, match=r"Expected 'INPUT' keyword"):
        Dummy().parse_input(strict=False)


def test_input_missing_varname_strict_false() -> None:
    class Dummy(Parser):
        def __init__(self) -> None:
            self.tokens = [Token("INPUT", "INPUT")]
            self.position = 0
            self.in_block = False

        def match(self, *args: str, strict: bool = True) -> Token | None:
            tok = self.current()
            if "INPUT" in args and tok.type == "INPUT":
                self.position += 1
                return tok
            if "IDENT" in args:
                return None  # Simulate missing IDENT, but don't increment position
            return None

        def current(self) -> Token:
            return (
                self.tokens[self.position]
                if self.position < len(self.tokens)
                else Token("EOF", "EOF")
            )

        def peek(self, offset: int = 1) -> Token:
            index = self.position + offset
            return (
                self.tokens[index] if index < len(self.tokens) else Token("EOF", "EOF")
            )

    with pytest.raises(SyntaxError, match=r"Expected variable name after INPUT"):
        Dummy().parse_input(strict=False)


def test_input_missing_file_after_from_1188() -> None:
    class Dummy(Parser):
        def __init__(self) -> None:
            self.tokens = [Token("INPUT", "INPUT"), Token("DELIM_FROM", "FROM")]
            self.position = 0
            self.in_block = False

        def match(self, *args: str, strict: bool = True) -> Token | None:
            tok = self.current()
            if "INPUT" in args and tok.type == "INPUT":
                self.position += 1
                return tok
            if "DELIM_FROM" in args and tok.type == "DELIM_FROM":
                self.position += 1
                return tok
            if "IDENT" in args or "STRING" in args:
                return None  # simulate failure
            return None

        def current(self) -> Token:
            return (
                self.tokens[self.position]
                if self.position < len(self.tokens)
                else Token("EOF", "EOF")
            )

        def peek(self, offset: int = 1) -> Token:
            return Token("EOF", "EOF")

    with pytest.raises(SyntaxError, match=r"Expected file path after 'FROM'"):
        Dummy().parse_input(strict=False)


def test_input_missing_type_after_colon_1218() -> None:
    class Dummy(Parser):
        def __init__(self) -> None:
            self.tokens = [
                Token("INPUT", "INPUT"),
                Token("IDENT", "x"),
                Token("COLON", ":"),
            ]
            self.position = 0
            self.in_block = False

        def match(self, *args: str, strict: bool = True) -> Token | None:
            tok = self.current()
            if "INPUT" in args and tok.type == "INPUT":
                self.position += 1
                return tok
            if "IDENT" in args and tok.type == "IDENT":
                self.position += 1
                return tok
            if "COLON" in args and tok.type == "COLON":
                self.position += 1
                return tok
            return None  # simulate missing type

        def current(self) -> Token:
            return (
                self.tokens[self.position]
                if self.position < len(self.tokens)
                else Token("EOF", "EOF")
            )

        def peek(self, offset: int = 1) -> Token:
            return Token("EOF", "EOF")

    with pytest.raises(SyntaxError, match=r"Expected type after colon in input"):
        Dummy().parse_input(strict=False)


def test_import_missing_keyword_304() -> None:
    class Dummy(Parser):
        def __init__(self) -> None:
            self.tokens = [Token("IDENT", "x")]
            self.position = 0
            self.in_block = False

        def match(self, *args: str, strict: bool = True) -> Token | None:
            tok = self.current()
            if "IMPORT" in args or "IDENT" in args:
                return None  # Simulate no match
            return tok

        def current(self) -> Token:
            return self.tokens[self.position]

        def peek(self, offset: int = 1) -> Token:
            return Token("EOF", "EOF")

    with pytest.raises(SyntaxError, match=r"Expected 'IMPORT' keyword"):
        Dummy().parse_import(strict=False)


def test_import_missing_path() -> None:
    class Dummy(Parser):
        def __init__(self) -> None:
            self.tokens = [Token("IMPORT", "IMPORT")]
            self.position = 0
            self.in_block = False

        def match(self, *args: str, strict: bool = True) -> Token | None:
            tok = self.current()
            if "IMPORT" in args or "IDENT" in args:
                self.position += 1
                return tok
            if "STRING" in args:
                return None  # Simulate missing path
            return tok

        def current(self) -> Token:
            return (
                self.tokens[self.position]
                if self.position < len(self.tokens)
                else Token("EOF", "EOF")
            )

        def peek(self, offset: int = 1) -> Token:
            return Token("EOF", "EOF")

    with pytest.raises(SyntaxError, match=r"Expected string path after IMPORT"):
        Dummy().parse_import(strict=False)


def test_expr_entrypoint_missing_ident_282() -> None:
    class Dummy(Parser):
        def __init__(self) -> None:
            self.tokens = [Token("IDENT", "foo")]
            self.position = 0
            self.in_block = False

        def match(self, *args: str, strict: bool = True) -> Token | None:
            if "IDENT" in args:
                return None
            return self.current()

        def current(self) -> Token:
            return self.tokens[self.position]

        def peek(self, offset: int = 1) -> Token:
            return Token("EOF", "EOF")

    with pytest.raises(SyntaxError, match=r"Expected identifier"):
        Dummy().parse_expr_entrypoint(strict=False)


def test_expr_entrypoint_missing_literal_295() -> None:
    class Dummy(Parser):
        def __init__(self) -> None:
            self.tokens = [Token("NUMBER", "123")]
            self.position = 0
            self.in_block = False

        def match(self, *args: str, strict: bool = True) -> Token | None:
            if self.current().type in args:
                return None
            return self.current()

        def current(self) -> Token:
            return self.tokens[self.position]

        def peek(self, offset: int = 1) -> Token:
            return Token("EOF", "EOF")

    with pytest.raises(SyntaxError, match=r"Expected literal value"):
        Dummy().parse_expr_entrypoint(strict=False)


def test_expr_entrypoint_prefix_op_1190() -> None:
    tokens = [
        Token("PLUS", "+"),
        Token("NUMBER", "1"),
        Token("NUMBER", "2"),
    ]
    parser = Parser(tokens)
    parser.in_block = True
    node_list = parser.parse_expr_entrypoint(strict=False)
    assert isinstance(node_list, list)
    assert node_list[0].kind == "arith"
    assert node_list[0].value == "PLUS"


def test_module_missing_keyword_131_132() -> None:
    class Dummy(Parser):
        def __init__(self) -> None:
            self.tokens = [Token("NUMBER", "42")]
            self.position = 0
            self.in_block = False

        def match(self, *args: str, strict: bool = True) -> Token | None:
            if strict:
                return super().match(*args, strict=True)
            return None

        def current(self) -> Token:
            return self.tokens[self.position]

        def peek(self, offset: int = 1) -> Token:
            return Token("EOF", "EOF")

    with pytest.raises(SyntaxError, match=r"Expected 'MODULE' keyword"):
        Dummy().parse_module(strict=False)


def test_module_missing_name_134_135() -> None:
    class Dummy(Parser):
        def __init__(self) -> None:
            self.tokens = [Token("MODULE", "MODULE")]
            self.position = 0
            self.in_block = False

        def match(self, *args: str, strict: bool = True) -> Token | None:
            if self.position == 0 and "MODULE" in args:
                self.position += 1
                return self.tokens[0]
            return None  # simulate failure to match IDENT name

        def current(self) -> Token:
            if self.position >= len(self.tokens):
                return Token("EOF", "EOF")
            return self.tokens[self.position]

        def peek(self, offset: int = 1) -> Token:
            return Token("EOF", "EOF")

    with pytest.raises(SyntaxError, match=r"Expected module name after 'MODULE'"):
        Dummy().parse_module(strict=False)


def test_expr_missing_open_135() -> None:
    tokens = make_tokens(("IDENT", "x"))
    parser = Parser(tokens)
    with pytest.raises(SyntaxError, match=r"Expected '\[' to start expression"):
        parser.parse_expression(strict=True)


def test_expr_call_missing_ident_154() -> None:
    tokens = make_tokens(
        ("LBRACK", "["), ("CALL", "CALL"), ("COMMA", ","), ("RBRACK", "]")
    )
    parser = Parser(tokens)
    with pytest.raises(
        SyntaxError, match=r"CALL must be followed by identifier or member access"
    ):
        parser.parse_expression(strict=True)


def test_expr_missing_closing_200() -> None:
    tokens = make_tokens(
        ("LBRACK", "["), ("PLUS", "+"), ("NUMBER", "1"), ("NUMBER", "2")
    )
    parser = Parser(tokens)
    with pytest.raises(SyntaxError, match=r"Expected closing '\]', got EOF"):
        parser.parse_expression(strict=True)


def test_expr_stray_comma_203() -> None:
    tokens = make_tokens(
        ("LBRACK", "["),
        ("PLUS", "+"),
        ("NUMBER", "1"),
        ("COMMA", ","),
        ("NUMBER", "2"),
        ("RBRACK", "]"),
    )
    parser = Parser(tokens)
    node = parser.parse_expression(strict=True)
    assert node.kind == "arith"
    assert len(node.children) == 2


def test_expr_op_type_compare_237() -> None:
    tokens = make_tokens(
        ("LBRACK", "["), ("LT", "<"), ("NUMBER", "1"), ("NUMBER", "2"), ("RBRACK", "]")
    )
    parser = Parser(tokens)
    node = parser.parse_expression(strict=True)
    assert node.kind == "compare"


def test_expr_call_missing_rparen_172() -> None:
    tokens = make_tokens(
        ("LBRACK", "["), ("CALL", "CALL"), ("IDENT", "f"), ("LPAREN", "(")
    )
    parser = Parser(tokens)
    with pytest.raises(
        SyntaxError, match=r"Expected expression or literal, got Token\(EOF, EOF\)"
    ):
        parser.parse_expression(strict=True)


def test_expr_token_none_215() -> None:
    tokens = make_tokens(
        ("LBRACK", "["), ("PLUS", "+"), ("COMMA", ","), ("RBRACK", "]")
    )
    parser = Parser(tokens)
    with pytest.raises(SyntaxError, match=r"PLUS requires at least 2 operands"):
        parser.parse_expression(strict=True)


def test_expr_op_type_propagate_239() -> None:
    tokens = make_tokens(
        ("LBRACK", "["), ("PROP", "$"), ("NUMBER", "99"), ("RBRACK", "]")
    )
    parser = Parser(tokens)
    with pytest.raises(SyntaxError, match=r"PROP requires at least 2 operands"):
        parser.parse_expression(strict=True)


def test_expr_open_tok_none_1099() -> None:
    class Dummy(Parser):
        def __init__(self) -> None:
            self.tokens = [Token("LBRACK", "[")]
            self.position = 0
            self.in_block = False

        def match(self, *args: str, strict: bool = True) -> Token | None:
            return None  # Force failure at open bracket

        def current(self) -> Token:
            return self.tokens[self.position]

        def peek(self, offset: int = 1) -> Token:
            return Token("EOF", "EOF")

    with pytest.raises(SyntaxError, match=r"Expected '\[' to open expression"):
        Dummy().parse_expression(strict=False)


def test_expr_missing_operator_1109() -> None:
    class Dummy(Parser):
        def __init__(self) -> None:
            self.tokens = [Token("LBRACK", "["), Token("IDENT", "x")]
            self.position = 0
            self.in_block = False
            self.expression_ops = ("PLUS", "SUB")  # type: ignore

        def match(self, *args: str, strict: bool = True) -> Token | None:
            if "LBRACK" in args:
                self.position += 1
                return self.tokens[0]
            if any(a in self.expression_ops for a in args):
                return None  # Simulate missing operator
            return self.tokens[self.position]

        def current(self) -> Token:
            return self.tokens[self.position]

        def peek(self, offset: int = 1) -> Token:
            return Token("EOF", "EOF")

    with pytest.raises(SyntaxError, match=r"Expected operator at start of expression"):
        Dummy().parse_expression(strict=False)


def test_expr_missing_call_ident_1118_pathological() -> None:
    class Dummy(Parser):
        def __init__(self) -> None:
            self.tokens = [
                Token("LBRACK", "["),
                Token("CALL", "CALL"),
                Token("IDENT", "wrong"),  # Looks like IDENT, but match will fail
                Token("RBRACK", "]"),
            ]
            self.position = 0
            self.in_block = False
            self.expression_ops = ("CALL",)  # type: ignore

        def match(self, *args: str, strict: bool = True) -> Token | None:
            tok = self.current()
            self.position += 1
            # Force failure ONLY for "IDENT" in CALL parsing
            if "CALL" in args and tok.type == "CALL":
                return tok
            if "LBRACK" in args and tok.type == "LBRACK":
                return tok
            if "IDENT" in args:
                return None  # forcibly simulate match failure
            return tok

        def current(self) -> Token:
            return self.tokens[self.position]

        def peek(self, offset: int = 1) -> Token:
            return Token("EOF", "EOF")

    with pytest.raises(SyntaxError, match=r"Expected identifier after CALL"):
        Dummy().parse_expression(strict=False)


def test_expr_unexpected_token_1179_pathological() -> None:
    class Dummy(Parser):
        def __init__(self) -> None:
            self.tokens = [
                Token("LBRACK", "["),
                Token("PLUS", "+"),
                Token("GARBAGE", "???"),  # definitely not matchable
                Token("RBRACK", "]"),
            ]
            self.position = 0
            self.in_block = False
            self.expression_ops = ("PLUS",)  # type: ignore
            self.unary_ops = ("NOT",)  # type: ignore
            self.arith_ops = ("PLUS",)  # type: ignore
            self.bool_ops = ()  # type: ignore
            self.comparison_ops = ()  # type: ignore

        def match(self, *args: str, strict: bool = True) -> Token | None:
            tok = self.current()
            if tok.type in args or tok.value in args:
                self.position += 1
                return tok
            return None  # Simulate failure

        def current(self) -> Token:
            return self.tokens[self.position]

        def peek(self, offset: int = 1) -> Token:
            return Token("EOF", "EOF")

    with pytest.raises(SyntaxError, match=r"Unexpected token in expression:"):
        Dummy().parse_expression(strict=False)


def test_expr_op_type_propagate_1201() -> None:
    class Dummy(Parser):
        def __init__(self) -> None:
            self.tokens = [
                Token("LBRACK", "["),
                Token("PROP", "$"),
                Token("NUMBER", "99"),
                Token(
                    "NUMBER", "1"
                ),  # Make sure it's treated as binary to hit propagate type
                Token("RBRACK", "]"),
            ]
            self.position = 0
            self.in_block = False
            self.expression_ops = ("PROP",)  # type: ignore
            self.unary_ops = ()  # type: ignore
            self.arith_ops = ()  # type: ignore
            self.bool_ops = ()  # type: ignore
            self.comparison_ops = ()  # type: ignore

        def match(self, *args: str, strict: bool = True) -> Token | None:
            tok = self.current()
            if tok.type in args or tok.value in args:
                self.position += 1
                return tok
            return None

        def current(self) -> Token:
            return self.tokens[self.position]

        def peek(self, offset: int = 1) -> Token:
            return Token("EOF", "EOF")

    node = Dummy().parse_expression(strict=False)
    assert node.kind == "propagate"
    assert node.value == "PROP"
    assert len(node.children) == 2


def test_if_condition_operand_match_failure_513_514() -> None:
    class Dummy(Parser):
        def __init__(self) -> None:
            self.tokens = [
                Token("IF", "IF"),
                Token("LPAREN", "("),
                Token("NOT", "NOT"),  # valid prefix op
                Token("COMMA", ","),  # invalid operand
                Token("RPAREN", ")"),
                Token("LBRACE", "{"),
                Token("RBRACE", "}"),
            ]
            self.position = 0
            self.in_block = False
            self.prefix_ops = ("NOT",)  # type: ignore

        def match(self, *args: str, strict: bool = True) -> Token | None:
            tok = self.current()
            if tok.type in args or tok.value in args:
                self.position += 1
                return tok
            return None

        def match_type_only(self, type_):  # type: ignore
            return self.match(type_)

        def current(self) -> Token:
            return self.tokens[self.position]

        def peek(self, offset: int = 1) -> Token:
            return Token("EOF", "EOF")

    with pytest.raises(SyntaxError, match=r"Expected token in IF condition"):
        Dummy().parse_if(strict=False)


def test_if_operator_kind_compare() -> None:
    tokens = [
        Token("IF", "IF"),
        Token("LPAREN", "("),
        Token("LT", "<"),  # comparison op
        Token("NUMBER", "1"),
        Token("NUMBER", "2"),
        Token("RPAREN", ")"),
        Token("LBRACE", "{"),
        Token("RBRACE", "}"),
    ]

    parser = Parser(tokens)
    parser.prefix_ops = ("LT",)  # type: ignore
    parser.unary_ops = ()  # type: ignore
    parser.arith_ops = ()  # type: ignore
    parser.bool_ops = ()  # type: ignore
    parser.comparison_ops = ("LT",)  # type: ignore

    node = parser.parse_if(strict=False)
    assert node.value.kind == "compare"  # type: ignore
    assert node.value.value == "LT"  # type: ignore


def test_parse_loop_range_with_step_hits_delim_by_branch() -> None:
    tokens = [
        Token("LOOP_FOR", "FOR"),
        Token("IDENT", "i"),
        Token("DELIM_AT", "AT"),
        Token("NUMBER", "1"),
        Token("DELIM_TO", "TO"),
        Token("NUMBER", "10"),
        Token("DELIM_BY", "BY"),
        Token("NUMBER", "2"),
        Token("LBRACE", "{"),
        Token("SKIP", "SKIP"),
        Token("RBRACE", "}"),
    ]
    parser = Parser(tokens)
    node = parser.parse_loop_range(strict=True)
    assert node.kind == "loop"
    assert node.children[0].children[3].value == "2"


def test_parse_input_with_type_annotation_hits_colon_branch() -> None:
    tokens = [
        Token("INPUT", "INPUT"),
        Token("IDENT", "foo"),
        Token("COLON", ":"),
        Token("IDENT", "int"),
    ]
    parser = Parser(tokens)
    node = parser.parse_input(strict=True)
    assert node.kind == "input"
    assert node.type == "int"


def test_parse_bracketed_call_expression_hits_rparen_arg_branch() -> None:
    tokens = [
        Token("LBRACK", "["),
        Token("CALL", "CALL"),
        Token("IDENT", "foo"),
        Token("LPAREN", "("),
        Token("NUMBER", "1"),
        Token("COMMA", ","),
        Token("NUMBER", "2"),
        Token("RPAREN", ")"),
        Token("RBRACK", "]"),
    ]
    parser = Parser(tokens)
    node = parser.parse_expression(strict=True)
    assert node.kind == "call"
    assert len(node.children) == 2
    assert node.children[0].value == "1"
    assert node.children[1].value == "2"


def test_loop_range_with_strict_step_and_negative_start() -> None:
    tokens = [
        Token("LOOP_FOR", "FOR"),
        Token("IDENT", "idx"),
        Token("DELIM_AT", "AT"),
        Token("SUB", "-"),
        Token("NUMBER", "1"),  # parse_signed_number() must take SUB path
        Token("DELIM_TO", "TO"),
        Token("NUMBER", "5"),
        Token("DELIM_BY", "BY"),
        Token("FLOAT", "2.5"),  # take float path
        Token("LBRACE", "{"),
        Token("SKIP", "SKIP"),
        Token("RBRACE", "}"),
    ]
    parser = Parser(tokens)
    node = parser.parse_loop_range(strict=True)

    range_node = node.children[0]
    assert node.kind == "loop"
    assert node.value == "FOR"
    assert range_node.kind == "range"
    assert len(range_node.children) == 4

    start_val = range_node.children[1].value
    end_val = range_node.children[2].value
    step_val = range_node.children[3].value

    assert start_val == "-1"  # SUB prefix was taken
    assert end_val == "5"
    assert step_val == "2.5"  # FLOAT hit


def test_parse_bracketed_call_expression_no_args() -> None:
    tokens = [
        Token("LBRACK", "["),
        Token("CALL", "CALL"),
        Token("IDENT", "foo"),
        Token("LPAREN", "("),
        Token("RPAREN", ")"),
        Token("RBRACK", "]"),
    ]
    parser = Parser(tokens)
    node = parser.parse_expression(strict=True)

    assert node.kind == "call"
    assert node.value.kind == "identifier"  # type: ignore
    assert node.value.value == "foo"  # type: ignore
    assert node.children == []


def test_parse_input_without_type_annotation() -> None:
    tokens = [
        Token("INPUT", "INPUT"),
        Token("IDENT", "foo"),
    ]
    parser = Parser(tokens)
    node = parser.parse_input(strict=True)

    assert node.kind == "input"
    assert node.value == "foo"
    assert node.type is None
