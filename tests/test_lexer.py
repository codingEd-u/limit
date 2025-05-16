import pytest
from hypothesis import given
from hypothesis import strategies as st

from limit.limit_lexer import CharacterStream, Lexer, Token


def tokenize(source: str):
    stream = CharacterStream(source, 0, 1, 1)
    lexer = Lexer(stream)
    tokens = []
    while True:
        tok = lexer.next_token()
        if tok.type == "EOF":
            break
        tokens.append(tok)
    return tokens


def test_single_char_tokens():
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


def test_string_token():
    lexer = Lexer(CharacterStream('"hello world"'))
    tok = lexer.next_token()
    assert tok.type == "STRING"
    assert tok.value == "hello world"


def test_number_token():
    lexer = Lexer(CharacterStream("123"))
    tok = lexer.next_token()
    assert tok.type == "NUMBER"
    assert tok.value == "123"


def test_float_token():
    lexer = Lexer(CharacterStream("123.456"))
    tok = lexer.next_token()
    assert tok.type == "FLOAT"
    assert tok.value == "123.456"


def test_identifier_token():
    lexer = Lexer(CharacterStream("myVar"))
    tok = lexer.next_token()
    assert tok.type == "IDENT"
    assert tok.value == "myVar"


def test_reserved_token():
    lexer = Lexer(CharacterStream("return"))
    tok = lexer.next_token()
    assert tok.type == "RETURN"
    assert tok.value == "return"


def test_literal_token():
    for literal in ["true", "false", "null"]:
        lexer = Lexer(CharacterStream(literal))
        tok = lexer.next_token()
        assert tok.type == "LITERAL"
        assert tok.value == literal


def test_line_and_column_tracking():
    code = "x = 1\ny = 2"
    lexer = Lexer(CharacterStream(code))
    tokens = [lexer.next_token() for _ in range(5)]  # x, =, 1, y, =
    assert tokens[3].line == 2
    assert tokens[3].col == 1


def test_skip_whitespace_and_comments():
    tokens = list(tokenize("   \n  # a comment\n123"))
    assert tokens[0].type == "NUMBER"
    assert tokens[0].value == "123"


def test_invalid_token_returns_error_token():
    source = "`"
    tokens = list(tokenize(source))
    assert any(tok.type == "ERROR" and tok.value == "`" for tok in tokens)


def test_token_eof():
    lexer = Lexer(CharacterStream(""))
    tok = lexer.next_token()
    assert tok.type == "EOF"
    assert tok.value == "EOF"


def test_character_stream_methods():
    stream = CharacterStream("abc")
    assert stream.peek() == "a"
    assert stream.next() == "a"
    assert stream.current() == "b"
    assert not stream.end_of_file()
    stream.next()
    stream.next()
    assert stream.end_of_file()


def test_token_repr_and_eq():
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


def test_character_stream_next_past_eof_raises():
    stream = CharacterStream("")
    with pytest.raises(
        Exception, match="CharacterStreamError: Attempted to read past end of source"
    ):
        stream.next()


def test_unclosed_string_raises():
    # Input has an opening quote and at least one character inside
    lexer = Lexer(CharacterStream('"abc'))
    with pytest.raises(SyntaxError, match="Unterminated string"):
        lexer.next_token()


@given(st.text(min_size=1, max_size=100))
def test_lexer_does_not_crash_on_random_input(input_str):
    stream = CharacterStream(input_str)
    lexer = Lexer(stream)
    try:
        while True:
            tok = lexer.next_token()
            if tok.type == "EOF":
                break
    except SyntaxError as e:
        # This is OK: the lexer is supposed to raise SyntaxError on malformed strings
        assert "Unterminated string" in str(e)
    except Exception as e:
        # All other exceptions are unexpected
        raise AssertionError(
            f"Lexer crashed unexpectedly on input: {repr(input_str)}\nError: {e}"
        ) from e


def tokenize_all(text):
    lexer = Lexer(CharacterStream(text))
    tokens = []
    while True:
        tok = lexer.next_token()
        tokens.append(tok)
        if tok.type == "EOF":
            break
    return tokens


@given(st.text(alphabet=st.characters(blacklist_categories=["Cs"]), min_size=1))
def test_unicode_survival(text):
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


def test_malformed_float_token():
    toks = tokenize_all("123..456")
    assert any(tok.type == "FLOAT" for tok in toks) or any(
        tok.type == "ERROR" for tok in toks
    )


def test_multiple_dots_in_float():
    toks = tokenize_all("1.2.3")
    types = [tok.type for tok in toks]
    assert "FLOAT" in types or "ERROR" in types


def test_escape_sequences_ignored_in_string():
    # Currently lexer treats escape slashes literally
    toks = tokenize_all('"line\\nbreak"')
    assert toks[0].type == "STRING"
    assert toks[0].value == "line\\nbreak"


def test_peek_beyond_end_returns_empty():
    stream = CharacterStream("abc")
    stream.next()
    stream.next()
    stream.next()  # At EOF
    assert stream.peek() == ""
    assert stream.peek(5) == ""
