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
    def __init__(self, stream: CharacterStream) -> None:
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

    def match_operator(self) -> Token | None:
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
