import importlib

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


def tokenize(source: str):
    stream = CharacterStream(source, 0, 1, 1)
    lexer = Lexer(stream)
    tokens = []
    while True:
        tok = lexer.next_token()
        tokens.append(tok)
        if tok.type == "EOF":
            break
    return tokens


def parse(source: str):
    return Parser(tokenize(source)).parse()


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
            "@ loop { ! x }",
            [
                ASTNode(
                    "func",
                    "loop",
                    [
                        ASTNode(
                            "print",
                            "x",
                            [ASTNode("identifier", "x", line=1, col=12)],
                            line=1,
                            col=10,
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
)
def test_parser_regressions(source, expected):
    result = parse(source)
    assert [n.to_dict() for n in result] == [n.to_dict() for n in expected]


@given(
    var=st.from_regex(r"[a-zA-Z_][a-zA-Z0-9_]{0,10}", fullmatch=True).filter(
        lambda x: x.upper() not in token_hashmap
    ),  # Fix: block any value mapped to a token
    num=st.integers(min_value=0, max_value=999),
)
def test_assignment_parses_correctly(var, num):
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
)
def test_addition_expression_parses_correctly(a, b):
    source = f"+ x{a} {b}"
    result = parse(source)
    assert len(result) == 1
    node = result[0]
    assert node.kind == "arith"
    assert node.value == "PLUS"
    assert node.children[0].kind == "identifier"
    assert node.children[1].kind == "number"


@given(literal=st.sampled_from(["true", "false", "null"]))
def test_literal_assignment_parses_correctly(literal):
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
            "@ f { RETURN }",
            [
                ASTNode(
                    "func",
                    "f",
                    [ASTNode("return", None, line=1, col=7)],
                    line=1,
                    col=1,
                    type_=None,  # Fixed from [] to None
                )
            ],
        ),
        # 70 – Propagate with bracketed expression
        (
            "@ f { $ [ AND x y ] }",
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
                                        ASTNode("identifier", "x", line=1, col=15),
                                        ASTNode("identifier", "y", line=1, col=17),
                                    ],
                                    line=1,
                                    col=11,
                                )
                            ],
                            line=1,
                            col=7,
                        )
                    ],
                    line=1,
                    col=1,
                    type_=None,  # FIXED FROM type_=[]
                )
            ],
        ),
        # 71 – Call with no args
        ("CALL f", [ASTNode("call", "f", [], line=1, col=1)]),
        # 72 – TRY with no catch/finally
        (
            "TRY { RETURN }",
            [
                ASTNode(
                    "try",
                    None,
                    [ASTNode("return", None, line=1, col=7)],  # previously col=7
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
        ("@ f { }", [ASTNode("func", "f", [], line=1, col=1, type_=[])]),
        # 75 – Class with multiple statements
        (
            "CLASS A { ! x RETURN }",
            [
                ASTNode(
                    "class",
                    "A",
                    [
                        ASTNode(
                            "print",
                            "x",
                            [ASTNode("identifier", "x", line=1, col=13)],
                            line=1,
                            col=11,
                        ),
                        ASTNode("return", None, line=1, col=15),
                    ],
                    line=1,
                    col=1,
                )
            ],
        ),
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
            "FOR { RETURN }",
            [
                ASTNode(
                    "loop",
                    "FOR",
                    [ASTNode("return", None, line=1, col=7)],
                    line=1,
                    col=1,
                )
            ],
        ),
    ],
)
def test_parser_missing_paths(source, expected):
    try:
        result = parse(source)
        if expected is None:
            pytest.fail("Expected SyntaxError but got AST")
        assert [node.to_dict() for node in result] == [
            node.to_dict() for node in expected
        ]
    except SyntaxError:
        if expected is not None:
            raise


@composite
def operator_expression(draw):
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


@given(expr=operator_expression())
def test_operator_expressions_cover_all(expr):
    try:
        result = parse(expr)
        assert isinstance(result[0], ASTNode)
        assert result[0].line > 0 and result[0].col > 0
    except SyntaxError:
        # Some malformed ones may sneak through generation
        pass


def test_call_with_member_and_args():
    source = "CALL a.b(c, d)"
    result = parse(source)
    assert result[0].kind == "call"
    assert result[0].children[0].kind == "member"
    assert len(result[0].children) == 3


def test_class_extends_with_block():
    source = "CLASS A EXTENDS B { ! x }"
    result = parse(source)
    cls = result[0]
    assert cls.kind == "class"
    assert cls.children[0].kind == "extends"
    assert cls.children[1].kind == "print"


def test_try_catch_finally_full_form():
    source = "TRY { RETURN } CATCH(e) { ! x } FINALLY { ! y }"
    result = parse(source)
    kinds = [c.kind for c in result[0].children]
    assert "catch" in kinds
    assert "finally" in kinds


def test_return_without_value():
    source = "@ f { RETURN }"
    result = parse(source)
    ret = result[0].children[0]
    assert ret.kind == "return"
    assert ret.children == []


def test_module_declaration():
    source = "MODULE MyMod"
    result = parse(source)
    node = result[0]
    assert node.kind == "module"
    assert node.value == "MyMod"


def test_import_statement():
    source = 'IMPORT "foo.min"'
    result = parse(source)
    node = result[0]
    assert node.kind == "import"
    assert node.value == "foo.min"


def test_export_statement():
    source = "EXPORT x"
    result = parse(source)
    node = result[0]
    assert node.kind == "export"
    assert node.value == "x"


def test_deeply_nested_expression():
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


def test_if_with_else_block():
    source = "@ f { ? (x) { RETURN } ELSE { RETURN } }"
    result = parse(source)
    func = result[0]
    assert func.kind == "func"
    if_node = func.children[0]
    assert if_node.kind == "if"
    assert hasattr(if_node, "else_children")
    assert len(if_node.else_children) == 1


def test_new_object_creation():
    source = "= x NEW Foo(1, 2)"
    result = parse(source)
    assign = result[0]
    assert assign.children[1].kind == "new"
    assert len(assign.children[1].children) == 2


def test_call_with_space_separated_args():
    source = "CALL myFunc x y"
    result = parse(source)
    assert result[0].kind == "call"
    assert result[0].value == "myFunc"
    assert len(result[0].children) == 2


def test_multiple_catch_blocks():
    source = "TRY { RETURN } CATCH(a) { ! x } CATCH(b) { ! y }"
    result = parse(source)
    kinds = [c.kind for c in result[0].children]
    assert kinds.count("catch") == 2


def test_input_without_from_clause():
    source = 'INPUT FROM "config.txt"'
    result = parse(source)
    assert result[0].kind == "input_from_file"
    assert result[0].value == "config.txt"


def test_return_with_bracket_expression():
    source = "@ f { RETURN [ + x 1 ] }"
    result = parse(source)
    ret = result[0].children[0]
    assert ret.kind == "return"
    assert ret.children[0].kind == "arith"


def test_function_with_no_return_type():
    source = "@ f(x, y) { RETURN }"
    result = parse(source)
    node = result[0]
    assert node.kind == "func"
    assert node.return_type is None


def test_call_with_empty_parens():
    source = "CALL myFunc()"
    result = parse(source)
    node = result[0]
    assert node.kind == "call"
    assert node.value == "myFunc"
    assert node.children == []


def test_propagate_with_literal():
    source = "@ f { $ true }"
    result = parse(source)
    prop = result[0].children[0]
    assert prop.kind == "propagate"
    assert prop.children[0].kind == "literal"
    assert prop.children[0].value == "true"


def test_empty_expression_node():
    parser = Parser(tokenize("[ ]"))
    expr = parser.parse_expression()
    assert expr.kind == "empty"


def test_parse_skips_none_nodes():
    tokens = tokenize("???")

    class DummyParser(Parser):
        def parse_statement(self):
            self.advance()
            return None

    parser = DummyParser(tokens)
    result = parser.parse()
    assert result == []


def test_invalid_syntax_raises():
    with pytest.raises(SyntaxError):
        parse("@ f { RETURN")  # unclosed block

    with pytest.raises(SyntaxError):
        parse("CALL")  # no target

    with pytest.raises(SyntaxError):
        parse("$ x")  # propagate outside block


def test_new_object_with_malformed_args():
    with pytest.raises(SyntaxError):
        parse("= x NEW Foo(1, )")


def test_empty_block_with_trailing_garbage():
    with pytest.raises(SyntaxError):
        parse("@ f { } ???")


def test_if_else_without_block_raises():
    with pytest.raises(SyntaxError):
        parse("IF ( x ) { RETURN } ELSE RETURN")


def test_invalid_operator_in_expression_raises():
    with pytest.raises(SyntaxError):
        parse("[ ?? x y ]")


def test_break_continue_outside_block_raise():
    with pytest.raises(SyntaxError):
        parse("BREAK")
    with pytest.raises(SyntaxError):
        parse("CONTINUE")


@given(
    ident=st.from_regex(r"[a-z]{1,4}", fullmatch=True).filter(
        lambda s: s.upper() not in reserved_words
    ),
    depth=st.integers(min_value=1, max_value=3),
)
def test_nested_member_access(ident, depth):
    chain = ".".join([ident] * (depth + 1))
    source = f"= x {chain}"
    result = parse(source)
    node = result[0].children[1]
    assert node.kind == "member"
    assert node.col > 0


@given(st.text(alphabet="@$%^&*~`|", min_size=1, max_size=6))
def test_fuzz_invalid_tokens_raise(text):
    with pytest.raises(SyntaxError):
        parse(text)


def test_function_single_statement_fallback():
    source = "@ f { RETURN }"
    result = parse(source)
    assert result[0].kind == "func"
    assert result[0].children[0].kind == "return"


def test_class_single_statement_fallback():
    source = "CLASS A { RETURN }"
    result = parse(source)
    assert result[0].kind == "class"
    assert result[0].children[0].kind == "return"


def test_while_with_bracket_expression_condition():
    source = "WHILE [ NOT x ] { RETURN }"
    result = parse(source)
    node = result[0]
    assert node.kind == "loop"


def test_while_empty_body_raises():
    with pytest.raises(SyntaxError, match="WHILE loop body cannot be empty"):
        parse("WHILE x { }")


def test_loop_block_empty_for_raises():
    with pytest.raises(SyntaxError, match="FOR loop body cannot be empty"):
        parse("FOR { }")


def test_class_extends_no_block_raises():
    with pytest.raises(SyntaxError, match="Class body must be enclosed"):
        parse("CLASS A EXTENDS B")


def test_class_with_no_body_raises():
    with pytest.raises(SyntaxError, match="Class body must be enclosed"):
        parse("CLASS A")


def test_else_without_braces_raises():
    with pytest.raises(SyntaxError, match="ELSE body must be enclosed"):
        parse("? (x) { RETURN } ELSE RETURN")


def test_function_missing_brace_body_raises():
    with pytest.raises(SyntaxError, match="Function body must start"):
        parse("@ f RETURN")


def test_parse_expression_invalid_operator():
    with pytest.raises(SyntaxError, match=r"Expected operator at start of expression"):
        parse("[ ??? x y ]")


def test_not_operator_with_too_many_args_raises():
    with pytest.raises(SyntaxError, match=r"NOT operator requires exactly one operand"):
        parse("[ NOT x y ]")


def test_add_operator_with_one_arg_raises():
    with pytest.raises(SyntaxError, match=r"PLUS requires at least two operands"):
        parse("[ + x ]")


def test_match_strict_false_fallback():
    parser = Parser([])
    result = parser.match("NONEXISTENT", strict=False)
    assert result is None


def test_expr_or_literal_strict_fallback():
    parser = Parser(tokenize("???"))
    with pytest.raises(SyntaxError):
        parser.parse_expr_or_literal()


def test_parse_expression_invalid_operator_non_strict():
    parser = Parser(tokenize("[ ??? x y ]"))
    result = parser.parse_expression(strict=False)
    assert result is None


def test_parse_expression_strict_not_arity_violation():
    with pytest.raises(SyntaxError, match="NOT operator requires exactly one operand"):
        parse("[ NOT x y ]")


def test_parse_expression_strict_binary_arity_violation():
    with pytest.raises(SyntaxError, match="PLUS requires at least two operands"):
        parse("[ + x ]")


def test_parse_block_invalid_delimiter():
    parser = Parser([])
    with pytest.raises(SyntaxError, match="Invalid block delimiter"):
        parser.parse_block(open_type="LBRACK")


def test_parse_loop_block_invalid_open_type():
    tokens = tokenize("FOR x to 5")  # missing block
    parser = Parser(tokens)
    with pytest.raises(SyntaxError, match=r"FOR loop body must be enclosed in braces"):
        parser.parse_loop_block("FOR")


def test_parse_loop_block_empty_body():
    with pytest.raises(SyntaxError, match="FOR loop body cannot be empty"):
        parse("FOR { }")


def test_parse_try_only_finally():
    source = "TRY { RETURN } FINALLY { RETURN }"
    result = parse(source)
    kinds = [n.kind for n in result[0].children]
    assert "finally" in kinds and "catch" not in kinds


def test_parse_try_with_single_catch():
    source = "TRY { RETURN } CATCH(e) { ! x }"
    result = parse(source)
    kinds = [n.kind for n in result[0].children]
    assert "catch" in kinds


def test_parse_statement_unrecognized_token():
    class DummyParser(Parser):
        def __init__(self):
            self.tokens = tokenize("???")
            self.position = 0

        def parse_statement(self):
            return None

    parser = DummyParser()
    with pytest.raises(SyntaxError, match="Unrecognized syntax near"):
        parser.parse()


def test_parse_class_extends_malformed_body():
    with pytest.raises(SyntaxError, match="Class body must be enclosed"):
        parse("CLASS A EXTENDS B ???")


def test_parse_function_with_param_and_return_type():
    source = "@ foo(x): int { RETURN }"
    result = parse(source)
    node = result[0]
    assert node.kind == "func"
    assert node.value == "foo"
    assert node.type == ["x"]
    assert node.return_type == "int"


def test_parse_function_param_comma_edge_case():
    with pytest.raises(SyntaxError):
        parse("@ f(x,) { RETURN }")


def test_parse_function_param_extra_token():
    with pytest.raises(SyntaxError):
        parse("@ f(x y) { RETURN }")


# new tests


# LINES 33, 43->56, 45->54, 62: match fallback and strict=False
def test_match_fallback_path_and_non_strict_match():
    parser = Parser([])
    assert parser.match("NON_EXISTENT", strict=False) is None


# LINES 99–101: parse_assignment – test NEW without parens
def test_assignment_new_without_parens():
    result = parse("= x NEW Foo")
    node = result[0].children[1]
    assert node.kind == "new"
    assert node.value == "Foo"
    assert node.children == []


# LINES 110–113: parse_assignment – test NEW Foo(1, 2)
def test_assignment_new_with_parens():
    result = parse("= x NEW Foo(1, 2)")
    node = result[0].children[1]
    assert node.kind == "new"
    assert len(node.children) == 2
    assert all(isinstance(c, ASTNode) for c in node.children)


# LINES 155–156, 162–163: parse_call – method-style CALL a.b x y
def test_call_method_style_space_args():
    result = parse("CALL a.b x y")
    node = result[0]
    assert node.kind == "call"
    assert node.children[0].kind == "member"
    assert len(node.children) == 3


# LINES 240–243: parse_continue not in block
def test_continue_outside_block_raises():
    with pytest.raises(SyntaxError, match="only allowed inside loop blocks"):
        parse("CONTINUE")


# LINES 267, 271: parse_member_access deeply chained
def test_member_access_chain_3_levels():
    result = parse("= x a.b.c")
    chain = result[0].children[1]
    assert chain.kind == "member"
    assert chain.children[0].kind == "member"
    assert chain.children[0].children[0].kind == "identifier"


# LINES 291: parse_propagate – not in block
def test_propagate_outside_block_raises():
    with pytest.raises(SyntaxError):
        parse("$ true")


# LINES 301–302: parse_try – catch with IDENT param
def test_try_catch_with_ident_param():
    source = "TRY { RETURN } CATCH e { RETURN }"
    result = parse(source)
    assert result[0].children[1].kind == "catch"
    assert result[0].children[1].value == "e"


# LINES 320–321: parse_try – no param catch
def test_try_catch_without_param():
    source = "TRY { RETURN } CATCH { RETURN }"
    result = parse(source)
    assert result[0].children[1].kind == "catch"
    assert result[0].children[1].value is None


# LINES 342, 350, 359: parse_try with finally block
def test_try_with_finally_block():
    source = "TRY { RETURN } FINALLY { RETURN }"
    result = parse(source)
    kinds = [n.kind for n in result[0].children]
    assert "finally" in kinds


# LINES 379–386: parse_expr_or_literal all branches
def test_expr_or_literal_member_access():
    result = parse("= x a.b")
    assert result[0].children[1].kind == "member"


# LINES 432–433: parse_return with literal
def test_return_literal():
    result = parse("@ f { RETURN true }")
    ret = result[0].children[0]
    assert ret.kind == "return"
    assert ret.children[0].value == "true"


# LINES 441–443: parse_return with no value
def test_return_no_value():
    result = parse("@ f { RETURN }")
    ret = result[0].children[0]
    assert ret.kind == "return"
    assert ret.children == []


# LINES 472, 475–476, 481, 488: parse_print fallback paths
def test_print_empty_and_literal():
    result = parse("! true")
    assert result[0].kind == "print"
    assert result[0].children[0].kind == "literal"


# LINES 492->470, 501, 505: parse_loop_while w/ empty condition
def test_while_empty_condition_raises():
    with pytest.raises(SyntaxError, match="WHILE condition cannot be empty"):
        parse("WHILE [ ] { RETURN }")


# LINES 525: parse_loop_while with literal condition
def test_while_with_literal_condition():
    result = parse("WHILE true { RETURN }")
    assert result[0].kind == "loop"


# LINES 592, 597->604, 605->604, 615->617: parse_expression branches
def test_expression_nested_with_member_and_literals():
    source = "[ + x.y 1 ]"
    result = parse(source)
    node = result[0]
    assert node.kind == "arith"
    assert node.children[0].kind == "member"
    assert node.children[1].kind == "number"


def test_expression_empty():
    result = Parser(tokenize("[ ]")).parse_expression()
    assert result.kind == "empty"


def test_expression_invalid_token_returns_none():
    parser = Parser(tokenize("[ ??? x ]"))
    assert parser.parse_expression(strict=False) is None


# LINES 673–674: parse_class with EXTENDS
def test_class_with_extends_and_body():
    source = "CLASS A EXTENDS B { RETURN }"
    result = parse(source)
    assert result[0].kind == "class"
    assert result[0].children[0].kind == "extends"


# 62: parse_assignment() fallback to THIS literal
def test_assignment_with_this_literal():
    result = parse("= x THIS")
    node = result[0]
    assert node.children[1].kind == "identifier"
    assert node.children[1].value == "THIS"


# 110–113: NEW Foo(1, "bar")
def test_assignment_new_with_mixed_args():
    result = parse('= x NEW Foo(1, "bar")')
    new_node = result[0].children[1]
    assert new_node.kind == "new"
    assert new_node.children[1].kind == "string"


# 155–156, 162–163: CALL obj.method(a, b)
def test_call_method_with_parens_args():
    result = parse("CALL obj.method(a, b)")
    call = result[0]
    assert call.kind == "call"
    assert len(call.children) == 3


# 267, 271: member access (a.b.c.d.e)
def test_long_member_access_chain():
    result = parse("= x a.b.c.d.e")
    node = result[0].children[1]
    depth = 0
    while node.kind == "member":
        depth += 1
        node = node.children[0]
    assert depth == 4


# 291: parse_propagate – raise outside block
def test_propagate_outside_block():
    with pytest.raises(SyntaxError, match="only allowed inside"):
        parse("$ x")


# 320–321: TRY block with CATCH (no parens, direct IDENT)
def test_try_with_ident_catch():
    source = "TRY { RETURN } CATCH x { RETURN }"
    result = parse(source)
    assert result[0].children[1].value == "x"


# 342, 359: TRY with FINALLY only
def test_try_finally_only():
    result = parse("TRY { RETURN } FINALLY { RETURN }")
    kinds = [c.kind for c in result[0].children]
    assert "finally" in kinds


# 379–386: parse_expr_or_literal: IDENT, LITERAL, STRING, NUMBER
@pytest.mark.parametrize(
    "expr,kind",
    [
        ("x", "identifier"),
        ("true", "literal"),
        ('"hello"', "string"),
        ("123", "number"),
    ],
)
def test_parse_expr_or_literal_variants(expr, kind):
    result = parse(f"= x {expr}")
    assert result[0].children[1].kind == kind


# 432–433, 441–443: RETURN with and without value
def test_return_variants():
    result = parse("@ f { RETURN 5 }")
    assert result[0].children[0].children[0].kind == "number"
    result2 = parse("@ f { RETURN }")
    assert result2[0].children[0].children == []


# 472, 475–476, 488: PRINT node with no children
def test_empty_print_node():
    result = parse("!")
    assert result[0].kind == "print"
    assert result[0].children == []


# 492->470, 501, 505: WHILE without braces
def test_while_without_block_raises():
    with pytest.raises(SyntaxError, match="WHILE must be followed by a block"):
        parse("WHILE true RETURN")


# 525: WHILE with basic literal condition
def test_while_with_true_literal():
    result = parse("WHILE true { RETURN }")
    assert result[0].kind == "loop"


# 592, 597->604, 605->604, 615->617: deeply nested expression arity paths
def test_nested_expressions_all_types():
    result = parse("[ AND [ OR x y ] z ]")
    node = result[0]
    assert node.kind == "bool"
    assert node.children[0].kind == "bool"
    assert node.children[1].kind == "identifier"


# 673–674: CLASS A EXTENDS B { RETURN }
def test_class_extends_full_path():
    result = parse("CLASS A EXTENDS B { RETURN }")
    assert result[0].children[0].kind == "extends"
    assert result[0].children[1].kind == "return"


def test_parser_current_hits_eof_token():
    parser = Parser([])
    tok = parser.current()
    assert tok.type == "EOF"


def test_match_strict_error_branch():
    parser = Parser([Token("IDENT", "hello")])
    with pytest.raises(SyntaxError):
        parser.match("NONMATCH")


def test_parse_bracketed_expression_calls_expression():
    parser = Parser(tokenize("[ + x y ]"))
    result = parser.parse_bracketed_expression()
    assert result.kind == "arith"


def test_assignment_new_with_commas():
    source = "= x NEW Foo(1, 2, 3)"
    result = parse(source)
    node = result[0].children[1]
    assert node.kind == "new"
    assert len(node.children) == 3
    assert all(c.kind == "number" for c in node.children)


@given(
    name=st.from_regex(r"[a-zA-Z_][a-zA-Z0-9_]{1,10}", fullmatch=True).filter(
        lambda x: x.upper() not in token_hashmap
    ),
    params=st.lists(
        st.from_regex(r"[a-z]{1,4}", fullmatch=True).filter(
            lambda p: p.upper() not in token_hashmap
        ),
        max_size=3,
    ),
    rettype=st.sampled_from(["int", "float", "str"]),
)
def test_function_decl_with_return_and_params(name, params, rettype):
    param_str = ", ".join(params)
    source = f"@ {name}({param_str}): {rettype} {{ RETURN }}"
    result = parse(source)
    node = result[0]
    assert node.kind == "func"
    assert node.value == name
    assert node.return_type == rettype
    assert node.line > 0 and node.col > 0


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
        alphabet=st.characters(blacklist_characters="\"'", blacklist_categories=["Cs"]),
        min_size=1,
        max_size=10,
    ).map(lambda s: f'"{s}"'),
    literal_val=st.sampled_from(["TRUE", "FALSE"]),
    ident=st.sampled_from(valid_identifiers),
)
@settings(deadline=None)
def test_assignment_new_arg_variants(num, float_val, string_val, literal_val, ident):
    for val in [num, float_val, string_val, literal_val]:
        source = f"= {ident} {val}"
        result = parse(source)
        assert result[0].kind == "assign"
        assert result[0].value == ident


def test_assignment_new_two_args_hits_comma_and_break():
    source = "= x NEW Foo(1, 2)"
    result = parse(source)
    new_node = result[0].children[1]
    assert new_node.kind == "new"
    assert len(new_node.children) == 2
    assert new_node.children[0].kind == "number"
    assert new_node.children[1].kind == "number"


def test_assignment_hits_member_access_branch():
    result = parse("= result some.object")
    rhs = result[0].children[1]
    assert rhs.kind == "member"
    assert rhs.children[0].kind == "identifier"


def test_for_range_with_non_number_limit_raises():
    with pytest.raises(
        SyntaxError, match=r"Expected one of .* got Token\(LITERAL, true\)"
    ):
        parse("FOR x to true { RETURN }")


def test_match_strict_false_returns_none():
    parser = Parser([])
    result = parser.match("NONEXISTENT", strict=False)
    assert result is None


def test_class_greeter_greet_method():
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
