"""
Lexical analyzer for the LIMIT programming language.

This module provides core components for converting raw source code into token streams:

Classes:
    CharacterStream: Stream abstraction for reading characters with line/column tracking.
    Token: Represents a single token with type, value, and source location.
    Lexer: Converts a CharacterStream into a sequence of tokens.

Features:
    - Skips whitespace and single-line comments (`#`)
    - Supports longest-match recognition of operators and keywords
    - Recognizes:
        * Identifiers and keywords
        * Numbers (integer and float)
        * Strings (with escape sequences)
        * Operators and punctuation

Raises:
    SyntaxError: If invalid floats or unterminated strings are encountered.

Example:
    >>> stream = CharacterStream("PRINT 42")
    >>> lexer = Lexer(stream)
    >>> token = lexer.next_token()
    >>> print(token)
    Token(PRINT, PRINT)

Exports:
    - CharacterStream
    - Token
    - Lexer
    - token_hashmap
"""

from typing import Any

from limit.limit_constants import token_hashmap


class CharacterStream:
    """
    A utility for reading characters from a string source with line and column tracking.

    This stream is used by the LIMIT lexer to support character-by-character parsing
    with precise source location metadata for error reporting.

    Attributes:
        source (str): The input source string.
        position (int): Current index in the source.
        line (int): Current line number (1-indexed).
        column (int): Current column number (1-indexed).
    """

    def __init__(self, source: str, position: int = 0, line: int = 1, column: int = 1):
        """
        Initializes the character stream.

        Args:
            source (str): The input source code.
            position (int, optional): Starting position index. Defaults to 0.
            line (int, optional): Starting line number. Defaults to 1.
            column (int, optional): Starting column number. Defaults to 1.
        """
        self.source = source
        self.position = position
        self.line = line
        self.column = column

    def next(self) -> str:
        """
        Consumes and returns the next character in the stream.

        Returns:
            str: The next character.

        Raises:
            Exception: If reading past the end of the source.
        """
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
        """
        Returns the character at the given offset from the current position without advancing.

        Args:
            offset (int, optional): Number of characters to look ahead. Defaults to 0.

        Returns:
            str: The character at the offset, or an empty string if out of bounds.
        """
        index = self.position + offset
        if index < 0 or index >= len(self.source):
            return ""
        return self.source[index]

    def current(self) -> str | None:
        """Returns the current character at the stream's position.

        Returns:
            str | None: The current character, or None if the stream has reached EOF.
        """
        return self.source[self.position] if self.position < len(self.source) else None

    def end_of_file(self) -> bool:
        """Checks if the stream has reached the end of the source input.

        Returns:
            bool: True if the stream has consumed all characters, False otherwise.
        """
        return self.position >= len(self.source)


class Token:
    """Represents a single lexical token in the LIMIT language.

    Attributes:
        type (str): The canonical token type (e.g. 'IDENT', 'NUMBER', 'EOF').
        value (str): The raw string value associated with the token.
        line (int): The 1-based line number where the token appears.
        col (int): The 1-based column number where the token starts.
    """

    def __init__(self, type_: str, value: str, line: int = 0, col: int = 0):
        """Initializes a new Token instance.

        Args:
            type_ (str): The token's type.
            value (str): The literal value of the token.
            line (int, optional): The line number (default is 0).
            col (int, optional): The column number (default is 0).
        """
        self.type = type_
        self.value = value
        self.line = line
        self.col = col

    def __repr__(self) -> str:
        """Returns a string representation of the token.

        Returns:
            str: A concise summary of the token's type and value.
        """
        return f"Token({self.type}, {self.value})"

    def __eq__(self, other: Any) -> bool:
        """Compares two Token instances for equality.

        Args:
            other (Any): The object to compare against.

        Returns:
            bool: True if equal, False otherwise.
        """
        return (
            isinstance(other, Token)
            and self.type == other.type
            and self.value == other.value
            and self.line == other.line
            and self.col == other.col
        )

    def __hash__(self) -> int:
        """Computes the hash of the token for use in sets and dicts.

        Returns:
            int: The token's hash value.
        """
        return hash((self.type, self.value, self.line, self.col))


class Lexer:
    """Lexical analyzer for the LIMIT language.

    The Lexer takes a CharacterStream and converts it into a stream of Token objects.
    It supports identifiers, numbers, strings, operators, and comments.

    Attributes:
        stream (CharacterStream): The source stream to tokenize.
    """

    def __init__(self, stream: CharacterStream) -> None:
        """Initializes the Lexer with a given character stream.

        Args:
            stream (CharacterStream): The input character stream to lex.
        """
        self.stream = stream

    def peek(self) -> str:
        """Returns the next character in the stream without consuming it.

        Returns:
            str: The upcoming character, or an empty string if EOF.
        """
        return self.stream.peek()

    def advance(self) -> str:
        """Consumes and returns the next character from the stream.

        Returns:
            str: The next character.
        """
        return self.stream.next()

    def skip_whitespace(self) -> None:
        """Skips all whitespace and comments in the stream."""
        while not self.stream.end_of_file():
            if self.peek() in " \t\r\n":
                self.advance()
            elif self.peek() == "#":
                self.skip_comment()
            else:
                break

    def skip_comment(self) -> None:
        """Advances through the stream until the end of a comment line."""
        while not self.stream.end_of_file() and self.peek() != "\n":
            self.advance()

    def match_operator(self) -> Token | None:
        """Attempts to match the longest valid operator from the current position.

        Returns:
            Token | None: A Token if a match is found, otherwise None.
        """
        line, col = self.stream.line, self.stream.column
        max_token = None
        match_len = 0
        candidate = ""

        for i in range(32):  # reasonable cap
            ch = self.stream.peek(i)
            if ch == "":
                break
            candidate += ch
            if candidate in token_hashmap:
                max_token = candidate
                match_len = i + 1

        if max_token:
            for _ in range(match_len):
                self.advance()
            return Token(token_hashmap[max_token], max_token, line, col)

        return None

    def next_token(self) -> Token:
        """Consumes and returns the next Token from the stream.

        Returns:
            Token: The next token parsed from the stream.

        Raises:
            SyntaxError: If a malformed token is encountered (e.g., unterminated string or malformed float).
        """
        if self.stream.end_of_file():
            return Token("EOF", "EOF", self.stream.line, self.stream.column)

        self.skip_whitespace()

        if self.stream.end_of_file():
            return Token("EOF", "EOF", self.stream.line, self.stream.column)

        ch = self.peek()
        line, col = self.stream.line, self.stream.column

        # 1. Identifier or keyword
        if ch.isalpha() or ch == "_":
            ident = ""
            while not self.stream.end_of_file() and (
                self.peek().isalnum() or self.peek() == "_"
            ):
                ident += self.advance()
            if ident in token_hashmap:
                return Token(token_hashmap[ident], ident, line, col)
            return Token("IDENT", ident, line, col)

        # 2. Number or float
        if ch.isdigit():
            num = ""
            has_dot = False
            while not self.stream.end_of_file() and (
                self.peek().isdigit() or self.peek() == "."
            ):
                if self.peek() == ".":
                    if has_dot:
                        raise SyntaxError(
                            f"Invalid float format at line {line}, col {col}"
                        )
                    has_dot = True
                num += self.advance()
            return Token("FLOAT" if has_dot else "NUMBER", num, line, col)

        # 3. String
        if ch in ('"', "'"):
            quote = self.advance()
            val = ""
            while not self.stream.end_of_file():
                if self.peek() == "\\":
                    val += self.advance()
                    if not self.stream.end_of_file():
                        val += self.advance()
                elif self.peek() == quote:
                    break
                else:
                    val += self.advance()
            if self.peek() == quote:
                self.advance()
                return Token("STRING", val, line, col)
            raise SyntaxError(f"Unterminated string at line {line}, col {col}")

        # 4. Compound or symbolic operator
        token = self.match_operator()
        if token:
            return token

        # 5. Unknown character → error
        if not self.stream.end_of_file():
            return Token("ERROR", self.advance(), line, col)
        else:
            return Token("EOF", "EOF", line, col)  # pragma: no cover


__all__ = ["CharacterStream", "Lexer", "Token", "token_hashmap"]

## For dev
# def print_tokens(source: str) -> None:
#     lexer = Lexer(CharacterStream(source))
#     while True:
#         tok = lexer.next_token()
#         print(tok)
#         if tok.type == "EOF":
#             break

# if __name__ == "__main__":
# sample = "TO0"
# print_tokens(sample)
