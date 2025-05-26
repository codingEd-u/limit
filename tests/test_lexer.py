import pytest
from hypothesis import given
from hypothesis import strategies as st

from limit.limit_lexer import CharacterStream, Lexer, Token


def tokenize(source: str) -> list[Token]:
    stream = CharacterStream(source, 0, 1, 1)
    lexer = Lexer(stream)
    tokens = []
    while True:
        tok = lexer.next_token()
        if tok.type == "EOF":
            break
        tokens.append(tok)
    return tokens


def test_single_char_tokens() -> None:
    code = "= + - * / % ? @ ! [ ] { } ( ) , : ."
    expected = [
        "ASSIGN",
        "PLUS",
        "SUB",
        "MULT",
        "DIV",
        "MOD",
        "IF",
        "FUNC",
        "PRINT",
        "LBRACK",
        "RBRACK",
        "LBRACE",
        "RBRACE",
        "LPAREN",
        "RPAREN",
        "COMMA",
        "COLON",
        "DOT",
    ]
    lexer = Lexer(CharacterStream(code))
    types = [lexer.next_token().type for _ in expected]
    assert types == expected


def test_string_token() -> None:
    lexer = Lexer(CharacterStream('"hello world"'))
    tok = lexer.next_token()
    assert tok.type == "STRING"
    assert tok.value == "hello world"


def test_number_token() -> None:
    lexer = Lexer(CharacterStream("123"))
    tok = lexer.next_token()
    assert tok.type == "NUMBER"
    assert tok.value == "123"


def test_float_token() -> None:
    lexer = Lexer(CharacterStream("123.456"))
    tok = lexer.next_token()
    assert tok.type == "FLOAT"
    assert tok.value == "123.456"


def test_identifier_token() -> None:
    lexer = Lexer(CharacterStream("myVar"))
    tok = lexer.next_token()
    assert tok.type == "IDENT"
    assert tok.value == "myVar"


def test_reserved_token() -> None:
    lexer = Lexer(CharacterStream("RETURN"))
    tok = lexer.next_token()
    assert tok.type == "RETURN"
    assert tok.value == "RETURN"


def test_reserved_token_disambiguation() -> None:
    lexer = Lexer(CharacterStream("return"))
    tok = lexer.next_token()
    assert tok.type == "IDENT"
    assert tok.value == "return"


def test_literal_token() -> None:
    for literal in ["NULL", "FALSE", "TRUE"]:
        lexer = Lexer(CharacterStream(literal))
        tok = lexer.next_token()
        assert tok.type == "LITERAL"
        assert tok.value == literal


def test_line_and_column_tracking() -> None:
    code = "x = 1\ny = 2"
    lexer = Lexer(CharacterStream(code))
    tokens = [lexer.next_token() for _ in range(5)]  # x, =, 1, y, =
    assert tokens[3].line == 2
    assert tokens[3].col == 1


def test_skip_whitespace_and_comments() -> None:
    tokens = list(tokenize("   \n  # a comment\n123"))
    assert tokens[0].type == "NUMBER"
    assert tokens[0].value == "123"


def test_invalid_token_returns_error_token() -> None:
    source = "`"
    tokens = list(tokenize(source))
    assert any(tok.type == "ERROR" and tok.value == "`" for tok in tokens)


def test_token_eof() -> None:
    lexer = Lexer(CharacterStream(""))
    tok = lexer.next_token()
    assert tok.type == "EOF"
    assert tok.value == "EOF"


def test_character_stream_methods() -> None:
    stream = CharacterStream("abc")
    assert stream.peek() == "a"
    assert stream.next() == "a"
    assert stream.current() == "b"
    assert not stream.end_of_file()
    stream.next()
    stream.next()
    assert stream.end_of_file()


def test_token_repr_and_eq() -> None:
    t1 = Token("NUMBER", "42", 1, 2)
    t2 = Token("NUMBER", "42", 1, 2)
    t3 = Token("IDENT", "x")

    assert repr(t1) == "Token(NUMBER, 42)"
    assert t1 == t2
    assert t1 != t3

    # Trigger __hash__
    token_set = {t1, t2, t3}
    assert t1 in token_set
    assert len(token_set) == 2  # t1 and t2 are equal, t3 is different


def test_character_stream_next_past_eof_raises() -> None:
    stream = CharacterStream("")
    with pytest.raises(
        Exception, match="CharacterStreamError: Attempted to read past end of source"
    ):
        stream.next()


def test_unclosed_string_raises() -> None:
    # Input has an opening quote and at least one character inside
    lexer = Lexer(CharacterStream('"abc'))
    with pytest.raises(SyntaxError, match="Unterminated string"):
        lexer.next_token()


@given(st.text(min_size=1, max_size=100))  # type: ignore[misc]
def test_lexer_does_not_crash_on_random_input(input_str: str) -> None:
    stream = CharacterStream(input_str)
    lexer = Lexer(stream)
    try:
        while True:
            tok = lexer.next_token()
            if tok.type == "EOF":
                break
    except SyntaxError as e:
        # Accept any known syntax error type
        assert "Unterminated string" in str(e) or "Invalid float format" in str(e)


def tokenize_all(text: str) -> list[Token]:
    lexer = Lexer(CharacterStream(text))
    tokens = []
    while True:
        tok = lexer.next_token()
        tokens.append(tok)
        if tok.type == "EOF":
            break
    return tokens


@given(st.text(alphabet=st.characters(blacklist_categories=["Cs"]), min_size=1))  # type: ignore[misc]
def test_unicode_survival(text: str) -> None:
    stream = CharacterStream(text)
    lexer = Lexer(stream)
    try:
        while True:
            tok = lexer.next_token()
            if tok.type == "EOF":
                break
    except SyntaxError:
        # Acceptable: malformed strings are part of valid input space
        pass
    except Exception as e:
        raise AssertionError(
            f"Lexer crashed on input: {repr(text)} with error: {e}"
        ) from e


def test_malformed_float_token() -> None:
    with pytest.raises(SyntaxError) as excinfo:
        tokenize_all("123..456")

    # Check exact message match
    msg = str(excinfo.value)
    assert "Invalid float format at line 1, col 1" in msg  # adjust col if needed


def test_multiple_dots_in_float() -> None:
    with pytest.raises(SyntaxError) as excinfo:
        tokenize_all("1.2.3")
    msg = str(excinfo.value)
    assert "Invalid float format at line 1, col 1" in msg  # adjust col if needed


def test_escape_sequences_ignored_in_string() -> None:
    # Currently lexer treats escape slashes literally
    toks = tokenize_all('"line\\nbreak"')
    assert toks[0].type == "STRING"
    assert toks[0].value == "line\\nbreak"


def test_peek_beyond_end_returns_empty() -> None:
    stream = CharacterStream("abc")
    stream.next()
    stream.next()
    stream.next()  # At EOF
    assert stream.peek() == ""
    assert stream.peek(5) == ""


def test_single_char_operator_token() -> None:
    lexer = Lexer(CharacterStream("+"))
    token = lexer.next_token()
    assert token.type == "PLUS"
    assert token.value == "+"


def test_unrecognized_character_returns_error() -> None:
    lexer = Lexer(CharacterStream("~"))
    token = lexer.next_token()
    assert token.type == "ERROR"
    assert token.value == "~"


def test_empty_input_returns_eof() -> None:
    lexer = Lexer(CharacterStream(""))
    token = lexer.next_token()
    assert token.type == "EOF"


def test_unterminated_string_with_trailing_escape() -> None:
    src = '"abc\\'
    lexer = Lexer(CharacterStream(src))
    with pytest.raises(SyntaxError, match=r"Unterminated string"):
        while True:
            tok = lexer.next_token()
            if tok.type == "EOF":
                break


def test_eof_via_fallback_path() -> None:
    # A token that skips all rules and hits the final return
    lexer = Lexer(CharacterStream(""))
    tok = lexer.next_token()
    assert tok.type == "EOF"
