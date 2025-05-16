# limit_lexer.py

from typing import Any

from limit.limit_constants import token_hashmap


class CharacterStream:
    def __init__(self, source: str, position: int = 0, line: int = 1, column: int = 1):
        self.source = source
        self.position = position
        self.line = line
        self.column = column

    def next(self) -> str:
        if self.position >= len(self.source):
            raise Exception(
                f"CharacterStreamError: Attempted to read past end of source at position=<{self.position}>, line=<{self.line}>"
            )
        char = self.source[self.position]
        if char == "\n":
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        self.position += 1
        return char

    def peek(self, offset: int = 0) -> str:
        index = self.position + offset
        if index < 0 or index >= len(self.source):
            return ""
        return self.source[index]

    def current(self) -> str | None:
        return self.source[self.position] if self.position < len(self.source) else None

    def end_of_file(self) -> bool:
        return self.position >= len(self.source)


class Token:
    def __init__(self, type_: str, value: str, line: int = 0, col: int = 0):
        self.type = type_
        self.value = value
        self.line = line
        self.col = col

    def __repr__(self) -> str:
        return f"Token({self.type}, {self.value})"

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, Token)
            and self.type == other.type
            and self.value == other.value
            and self.line == other.line
            and self.col == other.col
        )

    def __hash__(self) -> int:
        return hash((self.type, self.value, self.line, self.col))


class Lexer:
    def __init__(self, stream: CharacterStream):
        self.stream = stream

    def peek(self) -> str:
        return self.stream.peek()

    def advance(self) -> str:
        return self.stream.next()

    def skip_whitespace(self) -> None:
        while not self.stream.end_of_file():
            if self.peek() in " \t\r\n":
                self.advance()
            elif self.peek() == "#":
                self.skip_comment()
            else:
                break

    def skip_comment(self) -> None:
        while not self.stream.end_of_file() and self.peek() != "\n":
            self.advance()

    def next_token(self) -> Token:
        # First EOF check
        if self.stream.end_of_file():
            return Token("EOF", "EOF", self.stream.line, self.stream.column)

        self.skip_whitespace()

        # Second EOF check
        if self.stream.end_of_file():
            return Token("EOF", "EOF", self.stream.line, self.stream.column)

        line, col = self.stream.line, self.stream.column
        ch = self.peek()

        # Match multi-char symbols (non-alphanumeric only)
        for symbol in sorted(token_hashmap.keys(), key=len, reverse=True):
            if (
                not symbol.isalnum()
                and self.stream.source[
                    self.stream.position : self.stream.position + len(symbol)
                ]
                == symbol
            ):
                for _ in range(len(symbol)):
                    self.advance()
                return Token(token_hashmap[symbol], symbol, line, col)

        # Identifiers and keywords
        if ch.isalpha() or ch == "_":
            word = ""
            while not self.stream.end_of_file() and (
                self.peek().isalnum() or self.peek() == "_"
            ):
                word += self.advance()
            lookup = word.upper()
            tok_type = token_hashmap.get(lookup, "IDENT")
            return Token(tok_type, word, line, col)

        # Numeric literal (int or float)
        if ch.isdigit():
            value = ""
            has_dot = False
            while not self.stream.end_of_file() and (
                self.peek().isdigit() or (self.peek() == "." and not has_dot)
            ):
                if self.peek() == ".":
                    has_dot = True
                value += self.advance()
            token_type = "FLOAT" if has_dot else "NUMBER"
            return Token(token_type, value, line, col)

        # String literal
        if ch in ('"', "'"):
            quote = self.advance()
            string_value = ""
            while not self.stream.end_of_file():
                if self.peek() == quote:
                    break
                string_value += self.advance()
            if self.peek() == quote:
                self.advance()
                return Token("STRING", string_value, line, col)
            else:
                raise SyntaxError(f"Unterminated string at line {line}, col {col}")

        # Unknown character fallback
        if not self.stream.end_of_file():
            return Token("ERROR", self.advance(), line, col)

        return Token("EOF", "EOF", line, col)
