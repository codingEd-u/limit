"""
LIMIT Language Parser

Parses LIMIT language source tokens into structured abstract syntax trees (ASTs).

This module implements the core LIMIT parser logic, which transforms a flat list of
lexer-generated `Token` objects into a validated tree of `ASTNode` instances. The parser
fully supports the LIMIT language syntax, including nested expressions, block scoping,
and control flow constructs. It serves as the backbone for all downstream analysis,
transpilation, or interpretation.

Supported Constructs
--------------------
- Expressions:
    * Prefix notation: `[PLUS x 1]`, `[CALL foo 1 2]`
    * Nested expressions with proper bracket handling
    * Full operator support: arithmetic, boolean, comparison, member access, call, etc.

- Statements:
    * Assignments: `= x 5`, `= obj.method(...)`
    * Control flow: `IF`, `WHILE`, `FOR`, `TRY`/`CATCH`/`FINALLY`
    * Functions: `FUNC name(...) { ... }` with optional return types
    * Classes: `CLASS Foo EXTENDS Bar { ... }`
    * I/O: `INPUT`, `PRINT`
    * Modules: `MODULE`, `IMPORT`, `EXPORT`
    * Special: `SKIP`, `PROPAGATE`, `RETURN`, `BREAK`, `CONTINUE`

Parser Behavior
---------------
- Operates in strict mode by default: raises `SyntaxError` on invalid or malformed input.
- Performs multi-token lookahead and supports nested block structures.
- Supports both statement parsing and expression parsing via dedicated entry points.
- Expression parsing is exclusively prefix-based and requires square brackets `[]`.

Entry Points
------------
- `parse()`: Parse a full program into a list of top-level AST nodes.
- `parse_expression()`: Parse a single prefix expression (must be bracketed).
- `parse_statement()`: Parse a single statement (assignment, loop, control flow, etc.).
- `parse_expr_entrypoint()`: Parse one-liner expressions or literals (REPL mode).

Returns
-------
list[ASTNode]
    A list of parsed and validated ASTNode objects, representing the full abstract
    syntax tree (AST) of the LIMIT program or subexpression.

Raises
------
SyntaxError
    Raised when malformed input is encountered, unexpected tokens appear,
    or grammar rules are violated.
"""

from __future__ import annotations

from limit.limit_ast import ASTNode
from limit.limit_constants import operator_tokens, token_hashmap
from limit.limit_lexer import Token


class Parser:
    """
    LIMIT Parser Class

    Responsible for transforming a list of lexical tokens into structured abstract
    syntax trees (`ASTNode` objects) that represent a valid LIMIT program.

    This parser handles both top-level program structure and nested constructs such as
    function bodies, loop blocks, conditionals, and expressions. It supports full
    prefix-style expression parsing and enforces strict grammar rules unless explicitly
    configured otherwise.

    Attributes
    ----------
    tokens : list[Token]
        The input token stream to be parsed.
    position : int
        Current index into the token stream.
    in_block : bool
        Tracks whether the parser is currently inside a scoped block (e.g., function or loop).
    unary_ops : set[str]
        Set of operator token types considered unary (e.g., NOT, TRUTHY).
    arith_ops : set[str]
        Set of arithmetic operator token types (e.g., PLUS, SUB, MULT).
    bool_ops : set[str]
        Set of boolean operator token types (e.g., AND, OR, NOT).
    comparison_ops : set[str]
        Set of comparison operator token types (e.g., LT, EQ, GT).
    binary_ops : set[str]
        Union of arithmetic, boolean, and comparison operators.
    prefix_ops : set[str]
        All operator tokens allowed as the first token inside bracketed expressions.
    expression_ops : set[str]
        All valid expression-start tokens, including CALL and PROP.

    Methods
    -------
    parse() -> list[ASTNode]
        Parse a complete LIMIT program into an AST.
    parse_statement(strict: bool = True) -> ASTNode
        Parse a single LIMIT statement.
    parse_expression(strict: bool = True) -> ASTNode
        Parse a bracketed prefix expression.
    parse_expr_or_literal(strict: bool = True) -> ASTNode
        Parse a literal, identifier, or expression.
    parse_function(...) -> ASTNode
        Parse a function definition, including parameters and optional return type.
    parse_class(...) -> ASTNode
        Parse a class definition with optional inheritance.
    parse_block(...) -> list[ASTNode]
        Parse a `{}`-enclosed block of statements.
    parse_try(...) -> ASTNode
        Parse a `TRY`/`CATCH`/`FINALLY` block.
    parse_loop_while(...) -> ASTNode
        Parse a `WHILE` loop with optional bracketed condition.
    parse_loop_range(...) -> ASTNode
        Parse a `FOR` loop with `TO`, optional `AT` and `BY` range components.
    parse_assignment(...) -> ASTNode
        Parse an assignment statement.
    parse_call(...) -> ASTNode
        Parse a `CALL` expression or method call.
    parse_return(...) -> ASTNode
        Parse a `RETURN` statement with optional value.
    parse_input(...) -> ASTNode
        Parse an `INPUT` statement, optionally from file.
    parse_print(...) -> ASTNode
        Parse a `PRINT` statement.
    parse_export(...) -> ASTNode
        Parse an `EXPORT` declaration.
    parse_import(...) -> ASTNode
        Parse an `IMPORT` directive.
    parse_module(...) -> ASTNode
        Parse a `MODULE` declaration.
    parse_prefix_operator(...) -> ASTNode
        Parse a standalone prefix operator and its operands.
    parse_member_access(...) -> ASTNode
        Parse chained `.` access (e.g., `obj.x.y`).
    parse_propagate(...) -> ASTNode
        Parse a `PROPAGATE` (`$`) statement.
    parse_skip() -> ASTNode
        Parse a `SKIP` placeholder.

    Raises
    ------
    SyntaxError
        When an invalid construct or malformed syntax is encountered during parsing.
    """

    def __init__(self, tokens: list[Token]) -> None:
        self.tokens: list[Token] = tokens
        self.position: int = 0
        self.in_block: bool = False

        # Aliases
        PLUS = operator_tokens[0]
        SUB = operator_tokens[1]
        MULT = operator_tokens[2]
        DIV = operator_tokens[3]
        MOD = operator_tokens[4]

        AND = operator_tokens[5]
        OR = operator_tokens[6]
        NOT = operator_tokens[7]
        TRUTHY = operator_tokens[8]

        LT = operator_tokens[9]
        LE = operator_tokens[10]
        GT = operator_tokens[11]
        GE = operator_tokens[12]
        EQ = operator_tokens[13]
        NE = operator_tokens[14]

        CALL = operator_tokens[15]
        PROP = operator_tokens[16]

        # TOKEN MAPPINGS (PARSER)

        self.unary_ops: set[str] = {NOT, TRUTHY}

        self.arith_ops: set[str] = {
            PLUS,
            SUB,
            MULT,
            DIV,
            MOD,
        }

        self.bool_ops: set[str] = {AND, OR, NOT, TRUTHY}  # ✅ Include TRUTHY

        self.comparison_ops: set[str] = {
            LT,
            LE,
            GT,
            GE,
            EQ,
            NE,
        }

        self.binary_ops: set[str] = self.arith_ops | self.bool_ops | self.comparison_ops

        # Build prefix_ops by collecting all keys in token_hashmap that map to known op types
        canonical_ops = self.unary_ops | self.binary_ops
        self.prefix_ops: set[str] = {
            v for k, v in token_hashmap.items() if v in canonical_ops
        }

        self.expression_ops: set[str] = self.prefix_ops | {CALL, PROP}

        self.CALL: str = CALL
        self.PROP: str = PROP

    def current(self) -> Token:
        return (
            self.tokens[self.position]
            if self.position < len(self.tokens)
            else Token("EOF", "EOF")
        )

    def peek(self, offset: int = 1) -> Token:
        index = self.position + offset
        return self.tokens[index] if index < len(self.tokens) else Token("EOF", "EOF")

    def advance(self) -> Token:
        self.position += 1
        return self.current()

    def match_type_only(self, *types: str) -> Token:
        tok = self.current()
        if tok.type in types:
            self.advance()
            return tok
        raise SyntaxError(f"Expected token type in {types}, got {tok}")
        raise AssertionError("Unreachable")  # for mypy

    def match(self, *types: str, strict: bool = True) -> Token | None:
        tok = self.current()
        if tok.type in types or (tok.type == "IDENT" and tok.value in types):
            self.advance()
            return tok
        if strict:
            raise SyntaxError(f"Expected one of {types}, got {tok}")
        return None

    def parse_bracketed_expression(self) -> ASTNode | None:
        return self.parse_expression()

    def parse(self) -> list[ASTNode]:
        """Parse a full LIMIT program and return a list of top-level AST nodes."""
        ast: list[ASTNode] = []
        while self.current().type != "EOF":
            pos_before = self.position
            try:
                node = self.parse_statement()
                if node is not None:
                    ast.append(node)
                elif self.position == pos_before and self.current().type != "EOF":
                    raise SyntaxError(
                        f"Unrecognized syntax near: {self.current().value}"
                    )  # pragma: no cover
            except SyntaxError as e:
                raise e  # fail-fast
        return ast

    def parse_skip(self) -> ASTNode:
        """Parse a SKIP statement used as a placeholder stub."""
        tok = self.match("SKIP", "IDENT")
        assert tok is not None  # for mypy

        return ASTNode("skip", line=tok.line, col=tok.col)

    def parse_propagate(self, strict: bool = True) -> ASTNode:
        """Parse a PROPAGATE (`$`) statement to forward errors."""
        if not self.in_block:
            raise SyntaxError("Propagate operator `$` only allowed inside functions.")

        prop_tok = self.match("PROP", strict=strict)
        if prop_tok is None:
            raise SyntaxError("Expected '$' propagate operator")

        if self.current().type == "LBRACK":
            expr = self.parse_expression(strict=strict)
            if expr is None:
                raise SyntaxError("Expected valid expression after '$'")
        else:
            tok = self.match(
                "IDENT", "NUMBER", "FLOAT", "STRING", "LITERAL", "THIS", strict=strict
            )
            if tok is None:
                raise SyntaxError("Expected value after '$'")
            kind = "identifier" if tok.type in ("IDENT", "THIS") else tok.type.lower()
            expr = ASTNode(kind, tok.value, line=tok.line, col=tok.col)

        return ASTNode(
            "propagate",
            value=None,
            line=prop_tok.line,
            col=prop_tok.col,
            children=[expr],
        )

    def parse_loop_while(self, strict: bool = True) -> ASTNode:
        """Parse a WHILE loop with condition and body block."""
        loop_tok = self.match("LOOP_WHILE", strict=strict)
        if loop_tok is None:
            raise SyntaxError("Expected 'WHILE' loop start")

        # Parse condition
        if self.current().type == "LBRACK":
            if self.peek().type == "RBRACK":
                raise SyntaxError("WHILE condition cannot be empty")
            cond_node = self.parse_expression(strict=strict)
            assert cond_node is not None  # for mypy
        else:
            tok = self.match(
                "IDENT", "NUMBER", "FLOAT", "STRING", "LITERAL", strict=strict
            )
            if tok is None:
                raise SyntaxError("Expected condition after WHILE")
            kind = "identifier" if tok.type == "IDENT" else tok.type.lower()
            cond_node = ASTNode(kind, tok.value, line=tok.line, col=tok.col)

        if self.current().type not in ("LBRACK", "LBRACE"):
            raise SyntaxError("WHILE must be followed by a block")

        body = self.parse_block()
        if not body:
            raise SyntaxError("WHILE loop body cannot be empty")

        return ASTNode(
            "loop",
            value="WHILE",
            line=loop_tok.line,
            col=loop_tok.col,
            children=[cond_node] + body,
        )

    def parse_expr_or_literal(self, strict: bool = True) -> ASTNode:
        """Parse either a bracketed expression or a single literal/identifier."""
        if self.current().type == "LBRACK":
            expr = self.parse_expression(strict=strict)
            return expr

        if self.current().type == "IDENT" and self.peek().type == "DOT":
            return self.parse_member_access()

        tok = self.current()
        if tok.type not in ("IDENT", "NUMBER", "FLOAT", "STRING", "LITERAL", "THIS"):
            raise SyntaxError(f"Expected expression or literal, got {tok}")
        tok = self.match(tok.type, strict=strict)  # type: ignore[assignment]
        assert tok is not None  # for mypy

        kind = "identifier" if tok.type in ("IDENT", "THIS") else tok.type.lower()
        return ASTNode(kind, tok.value, line=tok.line, col=tok.col)

    def parse_prefix_operator(self, strict: bool = True) -> ASTNode:
        """Parse a prefix operator expression like `+ x 1` or `NOT flag`."""
        op_tok = self.match(
            "PLUS", "SUB", "MULT", "DIV", "MOD", "AND", "NOT", strict=strict
        )
        if op_tok is None:
            raise SyntaxError("Expected prefix operator")

        def parse_operand() -> ASTNode:
            if self.current().type == "LBRACK":
                expr = self.parse_expression(strict=strict)
                assert expr is not None  # for mypy
                return expr

            tok = self.match("IDENT", "NUMBER", strict=strict)
            if tok is None:
                raise SyntaxError("Expected operand after operator")
            kind = "number" if tok.type == "NUMBER" else "identifier"
            return ASTNode(kind, tok.value, line=tok.line, col=tok.col)

        lhs_node = parse_operand()
        children = [lhs_node]

        if op_tok.type != "NOT":
            rhs_node = parse_operand()
            children.append(rhs_node)

        kind = (
            "arith" if op_tok.type in ("PLUS", "SUB", "MULT", "DIV", "MOD") else "bool"
        )

        return ASTNode(
            kind, value=op_tok.type, line=op_tok.line, col=op_tok.col, children=children
        )

    def parse_call(self, strict: bool = True) -> ASTNode:
        """Parse a CALL expression with function or method syntax and arguments."""
        call_tok = self.match("CALL", strict=strict)
        if call_tok is None:
            raise SyntaxError("Expected 'CALL' keyword")

        # Handle method-style member access: CALL a.b(c, d)
        if self.current().type == "IDENT":
            if self.peek().type == "DOT":
                callee = self.parse_member_access()
            else:
                ident_tok = self.match("IDENT", strict=strict)
                if ident_tok is None:
                    raise SyntaxError("Expected callee name after CALL")
                callee = ASTNode(
                    "identifier",
                    ident_tok.value,
                    line=ident_tok.line,
                    col=ident_tok.col,
                )

            args = []
            if self.current().type == "LPAREN":
                self.match("LPAREN")
                if self.current().type != "RPAREN":
                    while True:
                        args.append(self.parse_expr_or_literal(strict=strict))
                        if self.current().type == "COMMA":
                            self.match("COMMA")
                        else:
                            break
                self.match("RPAREN")
            else:
                while self.current().type in (
                    "IDENT",
                    "NUMBER",
                    "FLOAT",
                    "STRING",
                    "LITERAL",
                    "THIS",
                    "LBRACK",
                ):
                    args.append(self.parse_expr_or_literal(strict=strict))

            return ASTNode(
                "call",
                value=callee,
                children=args,
                line=call_tok.line,
                col=call_tok.col,
            )

        raise SyntaxError("CALL must be followed by identifier or member access")

    def parse_statement(self, strict: bool = True) -> ASTNode:
        """Parse a single top-level or block-level LIMIT statement."""
        tok = self.current()

        if tok.type == "COMMA":
            self.advance()
            raise SyntaxError("Standalone comma is not a valid statement")

        if tok.type == "PROP":
            return self.parse_propagate(strict=strict)
        if tok.type in ("TRY", "IDENT") and tok.value == "TRY":
            return self.parse_try(strict=strict)
        if tok.type == "ASSIGN":
            return self.parse_assignment()
        if tok.type == "LBRACK":
            return self.parse_expression(strict=strict)
        if tok.type == "IF":
            return self.parse_if()
        if tok.type in ("SKIP", "IDENT") and tok.value == "SKIP":
            return self.parse_skip()
        if tok.type == "FUNC":
            return self.parse_function()
        if tok.type == "CALL":
            return self.parse_call(strict=strict)
        if tok.type == "INPUT":
            return self.parse_input()
        if tok.type == "PRINT":
            return self.parse_print()
        if tok.type == "RETURN":
            return self.parse_return()
        if tok.type in ("BREAK", "IDENT") and tok.value == "BREAK":
            return self.parse_break()
        if tok.type in ("CONTINUE", "IDENT") and tok.value == "CONTINUE":
            return self.parse_continue()
        if tok.type in ("MODULE", "IDENT") and tok.value == "MODULE":
            return self.parse_module()
        if tok.type in ("IMPORT", "IDENT") and tok.value == "IMPORT":
            return self.parse_import()
        if tok.type in ("EXPORT", "IDENT") and tok.value == "EXPORT":
            return self.parse_export()
        if tok.type == "LOOP_FOR":
            if (
                self.peek().type == "IDENT"
                and self.peek(2).type in ("DELIM_TO", "DELIM_AT")
                or self.peek(2).value.lower() in ("to", "at")
            ):
                return self.parse_loop_range()
            return self.parse_loop_block("FOR")
        if tok.type == "LOOP_WHILE":
            return self.parse_loop_while(strict=strict)
        if tok.type == "CLASS":
            return self.parse_class()
        if tok.type in ("PLUS", "SUB", "MULT", "DIV", "MOD", "AND", "NOT"):
            return self.parse_prefix_operator(strict=strict)

        try:
            expr = self.parse_expr_or_literal(strict=strict)

            while self.current().type == "LPAREN":
                self.match("LPAREN")
                args: list[ASTNode] = []
                if self.current().type != "RPAREN":
                    while True:
                        arg_tok = self.match(
                            "IDENT", "NUMBER", "FLOAT", "STRING", "LITERAL", "THIS"
                        )
                        if arg_tok is None:
                            raise SyntaxError("Expected argument in function call")
                        kind = (
                            "identifier"
                            if arg_tok.type in ("IDENT", "THIS")
                            else arg_tok.type.lower()
                        )
                        args.append(
                            ASTNode(
                                kind, arg_tok.value, line=arg_tok.line, col=arg_tok.col
                            )
                        )
                        if self.current().type == "COMMA":
                            self.match("COMMA")
                        else:
                            break
                self.match("RPAREN")
                expr = ASTNode(
                    "call", value=expr, children=args, line=expr.line, col=expr.col
                )

            return ASTNode("expr_stmt", children=[expr], line=expr.line, col=expr.col)

        except SyntaxError as e:
            raise SyntaxError(
                f"Unrecognized statement: {tok.value} (type={tok.type})"
            ) from e

    def parse_try(self, strict: bool = True) -> ASTNode:
        """Parse a TRY block with CATCH handlers and optional FINALLY."""
        try_tok = self.match("TRY", "IDENT", strict=strict)
        if try_tok is None or (try_tok.type == "IDENT" and try_tok.value != "TRY"):
            raise SyntaxError("Expected 'TRY' keyword")

        try_block = self.parse_block()
        catches = []

        while self.current().type == "CATCH" or (
            self.current().type == "IDENT" and self.current().value == "CATCH"
        ):
            catch_tok = self.match("CATCH", "IDENT", strict=strict)
            if catch_tok is None or (
                catch_tok.type == "IDENT" and catch_tok.value != "CATCH"
            ):
                raise SyntaxError("Expected 'CATCH' keyword")

            if self.current().type == "LPAREN":
                self.match("LPAREN")
                exc_name_tok = self.match("IDENT", strict=strict)
                if exc_name_tok is None:
                    raise SyntaxError("Expected exception name inside CATCH()")
                exc_name = exc_name_tok.value
                self.match("RPAREN")
            elif self.current().type == "IDENT":
                exc_name_tok = self.match("IDENT", strict=strict)
                if exc_name_tok is None:
                    raise SyntaxError("Expected exception name in CATCH")
                exc_name = exc_name_tok.value
            else:
                exc_name = None

            catch_block = self.parse_block()
            catches.append(
                ASTNode(
                    "catch",
                    exc_name,
                    catch_block,
                    line=catch_tok.line,
                    col=catch_tok.col,
                )
            )

        finally_node = None
        if self.current().type == "FINALLY" or (
            self.current().type == "IDENT" and self.current().value == "FINALLY"
        ):
            finally_tok = self.match("FINALLY", "IDENT", strict=strict)
            finally_block = self.parse_block()
            finally_node = ASTNode(
                "finally",
                None,
                finally_block,
                line=finally_tok.line,  # type: ignore
                col=finally_tok.col,  # type: ignore
            )

        children = try_block
        children.extend(catches)
        if finally_node:
            children.append(finally_node)

        return ASTNode("try", None, children, line=try_tok.line, col=try_tok.col)

    def parse_block(
        self, open_type: str = "LBRACE", strict: bool = True
    ) -> list[ASTNode]:
        """Parse a `{}`-enclosed scoped block of LIMIT statements."""
        if open_type != "LBRACE":
            raise SyntaxError(
                f"Invalid block delimiter: {open_type}. Blocks must use '{{' and '}}'."
            )

        expected_close = "RBRACE"
        self.match("LBRACE", strict=strict)

        stmts = []
        prev_in_block = self.in_block
        self.in_block = True

        if self.current().type == expected_close:
            self.match(expected_close, strict=strict)
            self.in_block = prev_in_block
            return []

        while self.current().type != expected_close:
            if self.current().type == "EOF":
                raise SyntaxError("Expected closing '}', got EOF")

            pos_before = self.position
            stmt = self.parse_statement(strict=strict)
            pos_after = self.position

            if stmt is not None:
                stmts.append(stmt)
            elif pos_after == pos_before:
                raise SyntaxError(
                    f"Unrecognized statement inside block near: {self.current().value} (type={self.current().type})"
                )  # pragma: no cover

        self.match(expected_close, strict=strict)
        self.in_block = prev_in_block
        return stmts

    def parse_if(self, strict: bool = True) -> ASTNode:
        """Parse an IF condition with optional ELSE block."""
        if_tok = self.match("IF", strict=strict)
        assert if_tok is not None  # for mypy

        self.match("LPAREN", strict=strict)

        op_tok = self.current()
        if op_tok.type not in self.prefix_ops:
            raise SyntaxError(f"Invalid IF operator: {op_tok}")
        op_tok = self.match_type_only(op_tok.type)
        assert op_tok is not None  # for mypy

        children: list[ASTNode] = []
        while self.current().type != "RPAREN":
            if self.current().type == "EOF":
                raise SyntaxError("Unexpected EOF in IF condition")

            tok = self.match(
                "IDENT", "NUMBER", "FLOAT", "STRING", "LITERAL", "THIS", strict=strict
            )
            if tok is None:
                raise SyntaxError("Expected token in IF condition")
            kind = "identifier" if tok.type in ("IDENT", "THIS") else tok.type.lower()
            children.append(ASTNode(kind, tok.value, line=tok.line, col=tok.col))
        self.match("RPAREN", strict=strict)

        if op_tok.type in self.unary_ops and len(children) != 1:
            raise SyntaxError(f"{op_tok.type} requires 1 operand")
        elif op_tok.type not in self.unary_ops and len(children) < 2:
            raise SyntaxError(f"{op_tok.type} requires at least 2 operands")

        if op_tok.type in self.arith_ops:
            kind = "arith"
        elif op_tok.type in self.bool_ops:
            kind = "bool"
        elif op_tok.type in self.comparison_ops:
            kind = "compare"
        else:
            raise AssertionError(f"Unexpected operator: {op_tok}")  # pragma: no cover

        cond_expr = ASTNode(
            kind, value=op_tok.type, line=op_tok.line, col=op_tok.col, children=children
        )

        if self.current().type != "LBRACE":
            raise SyntaxError("IF body must be enclosed in braces '{ }'")
        then_block = self.parse_block("LBRACE", strict=strict)

        else_block: list[ASTNode] = []
        if self.current().type == "ELSE":
            self.match("ELSE", strict=strict)
            if self.current().type != "LBRACE":
                raise SyntaxError("ELSE body must be enclosed in braces '{ }'")
            else_block = self.parse_block("LBRACE", strict=strict)

        node = ASTNode(
            "if", value=cond_expr, line=if_tok.line, col=if_tok.col, children=then_block
        )
        if else_block:
            node.else_children = else_block
        return node

    def parse_assignment(self, strict: bool = True) -> ASTNode:
        """Parse a variable assignment with identifier and right-hand value."""
        assign_tok = self.match("ASSIGN", strict=strict)
        assert assign_tok is not None  # for mypy

        # LEFT-HAND SIDE
        if self.current().type == "IDENT":
            name_node = self.parse_member_access()
        else:
            raise SyntaxError(f"Invalid assignment target: {self.current()}")

        # RIGHT-HAND SIDE
        if self.current().type == "NEW":
            new_tok = self.match("NEW", strict=strict)
            assert new_tok is not None  # for mypy
            class_tok = self.match("IDENT", strict=strict)
            assert class_tok is not None  # for mypy
            args = []
            if self.current().type == "LPAREN":
                self.match("LPAREN", strict=strict)
                if self.current().type != "RPAREN":
                    while True:
                        arg_tok = self.match(
                            "IDENT",
                            "NUMBER",
                            "FLOAT",
                            "STRING",
                            "LITERAL",
                            "THIS",
                            strict=strict,
                        )
                        assert arg_tok is not None  # for mypy
                        kind = (
                            "identifier"
                            if arg_tok.type in ("IDENT", "THIS")
                            else arg_tok.type.lower()
                        )
                        args.append(
                            ASTNode(
                                kind, arg_tok.value, line=arg_tok.line, col=arg_tok.col
                            )
                        )
                        if self.current().type == "COMMA":
                            self.match("COMMA", strict=strict)
                        else:
                            break
                self.match("RPAREN", strict=strict)
            value_node = ASTNode(
                "new",
                class_tok.value,
                children=args,
                line=new_tok.line,
                col=new_tok.col,
            )

        elif self.current().type == "IDENT" and self.peek().type == "DOT":
            value_node = self.parse_member_access()

        elif self.current().type == "IDENT" and self.peek().type == "LPAREN":
            func_tok = self.match("IDENT", strict=strict)
            assert func_tok is not None  # for mypy
            args = []
            self.match("LPAREN", strict=strict)
            if self.current().type != "RPAREN":
                while True:
                    arg_tok = self.match(
                        "IDENT",
                        "NUMBER",
                        "FLOAT",
                        "STRING",
                        "LITERAL",
                        "THIS",
                        strict=strict,
                    )
                    assert arg_tok is not None  # for mypy
                    kind = (
                        "identifier"
                        if arg_tok.type in ("IDENT", "THIS")
                        else arg_tok.type.lower()
                    )
                    args.append(
                        ASTNode(kind, arg_tok.value, line=arg_tok.line, col=arg_tok.col)
                    )
                    if self.current().type == "COMMA":
                        self.match("COMMA", strict=strict)
                    else:
                        break
            self.match("RPAREN", strict=strict)
            value_node = ASTNode(
                "call",
                func_tok.value,
                children=args,
                line=func_tok.line,
                col=func_tok.col,
            )

        elif self.current().type == "LBRACK":
            value_node = self.parse_expression(strict=strict)
            if value_node is None:
                raise SyntaxError("Expected expression on right-hand side")

        elif self.current().type in (
            "NUMBER",
            "FLOAT",
            "STRING",
            "LITERAL",
            "IDENT",
            "THIS",
        ):
            value_tok = self.match(
                "NUMBER", "FLOAT", "STRING", "LITERAL", "IDENT", "THIS", strict=strict
            )
            if value_tok is None:
                raise SyntaxError("Expected value on right-hand side")
            kind = (
                "identifier"
                if value_tok.type in ("IDENT", "THIS")
                else value_tok.type.lower()
            )
            value_node = ASTNode(
                kind, value_tok.value, line=value_tok.line, col=value_tok.col
            )

        else:
            raise SyntaxError(
                f"Invalid right-hand side in assignment: {self.current()}"
            )

        return ASTNode(
            "assign",
            value=name_node.value if name_node.kind == "identifier" else None,
            children=[name_node, value_node],
            line=assign_tok.line,
            col=assign_tok.col,
            type_="local" if self.in_block else "global",
        )

    def parse_class(self, strict: bool = True) -> ASTNode:
        """Parse a CLASS definition with optional EXTENDS and class body."""
        class_tok = self.match("CLASS", strict=strict)
        assert class_tok is not None  # for mypy
        name_tok = self.match("IDENT", strict=strict)
        assert name_tok is not None  # for mypy

        extends_node: ASTNode | None = None
        if self.current().type == "EXTENDS":
            self.match("EXTENDS", strict=strict)
            base_tok = self.match("IDENT", strict=strict)
            extends_node = ASTNode(
                "extends", value=base_tok.value, line=base_tok.line, col=base_tok.col  # type: ignore
            )

        self.match("LBRACE", strict=strict)
        body: list[ASTNode] = []
        while self.current().type != "RBRACE":
            stmt = self.parse_statement(strict=strict)
            if stmt is None:
                raise SyntaxError("Invalid statement in class body")
            body.append(stmt)
        self.match("RBRACE", strict=strict)

        children: list[ASTNode] = [extends_node] + body if extends_node else body
        return ASTNode(
            "class",
            value=name_tok.value,
            children=children,
            line=name_tok.line,
            col=name_tok.col,
        )

    def parse_loop_block(self, kind: str, strict: bool = True) -> ASTNode:
        """Parse a generic loop block enclosed in braces (used by FOR/WHILE fallback)."""
        loop_tok = self.match(
            "LOOP_FOR" if kind == "FOR" else "LOOP_WHILE", strict=strict
        )
        assert loop_tok is not None  # for mypy

        if self.current().type != "LBRACE":
            raise SyntaxError(f"{kind} loop body must be enclosed in braces '{{}}'")

        body = self.parse_block("LBRACE", strict=strict)
        if not body:
            raise SyntaxError(f"{kind} loop body cannot be empty")

        return ASTNode(
            "loop", value=kind, line=loop_tok.line, col=loop_tok.col, children=body
        )

    def parse_function(self, strict: bool = True) -> ASTNode:
        """Parse a FUNC definition including parameters, return type, and body."""
        func_tok = self.match("FUNC", strict=strict)

        name_tok = self.match("IDENT", strict=strict)

        self.match("LPAREN", strict=strict)

        params: list[str] = []
        if self.current().type != "RPAREN":
            while True:
                param_tok = self.match("IDENT", strict=strict)
                params.append(param_tok.value)  # type: ignore

                if self.current().type == "COMMA":
                    self.match("COMMA", strict=strict)
                else:
                    break

        self.match("RPAREN", strict=strict)

        return_type: str | None = None
        if self.current().type == "COLON":
            self.match("COLON", strict=strict)
            rt_tok = self.match("IDENT", strict=strict)
            return_type = rt_tok.value  # type: ignore

        if self.current().type != "LBRACE":
            raise SyntaxError("Function body must start with '{'")
        block = self.parse_block("LBRACE", strict=strict)

        node = ASTNode("func", value=name_tok.value, line=func_tok.line, col=func_tok.col, children=block)  # type: ignore
        node.type = ",".join(params) if params else None
        node.return_type = return_type
        return node

    def parse_return(self, strict: bool = True) -> ASTNode:
        """Parse a RETURN statement with optional expression or value."""
        if not self.in_block:
            raise SyntaxError("RETURN is only allowed inside function blocks.")

        tok = self.match("RETURN", strict=strict)

        # Bracketed expression: RETURN [EXPR]
        if self.current().type == "LBRACK":
            expr = self.parse_expression(strict=strict)
            return ASTNode(
                "return",
                children=[expr],
                line=tok.line,  # type: ignore
                col=tok.col,  # type: ignore
            )

        # Member access: RETURN obj.member
        if self.current().type == "IDENT" and self.peek().type == "DOT":
            member_node = self.parse_member_access()
            return ASTNode(
                "return",
                children=[member_node],
                line=tok.line,  # type: ignore
                col=tok.col,  # type: ignore
            )

        # Literal or identifier
        if self.current().type in (
            "IDENT",
            "NUMBER",
            "FLOAT",
            "STRING",
            "LITERAL",
            "THIS",
        ):
            val = self.match(
                "IDENT", "NUMBER", "FLOAT", "STRING", "LITERAL", "THIS", strict=strict
            )
            kind = "identifier" if val.type in ("IDENT", "THIS") else val.type.lower()  # type: ignore
            return ASTNode(
                "return",
                children=[ASTNode(kind, val.value, line=val.line, col=val.col)],  # type: ignore
                line=tok.line,  # type: ignore
                col=tok.col,  # type: ignore
            )

        # No return value (just RETURN)
        return ASTNode(
            "return",
            value=None,
            line=tok.line,  # type: ignore
            col=tok.col,  # type: ignore
        )

    def parse_member_access(self, strict: bool = True) -> ASTNode:
        """
        Parse chained member access expressions of the form:
            obj.y.z
        Consumes IDENT, then any number of (DOT IDENT) pairs.
        """
        ident_tok = self.match("IDENT", strict=strict)
        if ident_tok is None:
            raise SyntaxError("Expected identifier at start of member access")
        assert ident_tok is not None  # for mypy

        node = ASTNode(
            "identifier", ident_tok.value, line=ident_tok.line, col=ident_tok.col
        )

        while self.current().type == "DOT":
            self.match("DOT", strict=strict)
            member_tok = self.match("IDENT", strict=strict)
            if member_tok is None:
                raise SyntaxError("Expected identifier after '.' in member access")
            assert member_tok is not None  # for mypy

            node = ASTNode(
                "member",
                member_tok.value,
                [node],
                line=member_tok.line,
                col=member_tok.col,
            )

        return node

    def parse_break(self, strict: bool = True) -> ASTNode:
        if not self.in_block:
            raise SyntaxError("BREAK only allowed inside loop blocks.")

        tok = self.match("BREAK", "IDENT", strict=strict)
        if tok is None:
            raise SyntaxError("Expected 'BREAK' keyword")
        assert tok is not None  # for mypy

        return ASTNode("break", line=tok.line, col=tok.col)

    def parse_continue(self, strict: bool = True) -> ASTNode:
        if not self.in_block:
            raise SyntaxError("CONTINUE only allowed inside loop blocks.")

        tok = self.match("CONTINUE", "IDENT", strict=strict)
        if tok is None:
            raise SyntaxError("Expected 'CONTINUE' keyword")
        assert tok is not None  # for mypy

        return ASTNode("continue", line=tok.line, col=tok.col)

    def parse_export(self, strict: bool = True) -> ASTNode:
        """Parse an EXPORT statement that exposes an identifier."""
        tok = self.match("EXPORT", "IDENT", strict=strict)
        if tok is None:
            raise SyntaxError("Expected 'EXPORT' keyword")
        assert tok is not None  # for mypy

        name_tok = self.match("IDENT", strict=strict)
        if name_tok is None:
            raise SyntaxError("Expected identifier after EXPORT")
        assert name_tok is not None  # for mypy

        return ASTNode("export", name_tok.value, line=tok.line, col=tok.col)

    def parse_print(self, strict: bool = True) -> ASTNode:
        """
        Print statement can be:
        ! x
        ! 123
        ! "hello"
        ! [ PLUS x 1 ]
        ! [CALL sum 2 3]
        """
        print_tok = self.match("PRINT", strict=strict)
        if print_tok is None:
            raise SyntaxError("Expected 'PRINT' keyword")

        # Bracketed expression: ! [ ... ]
        if self.current().type == "LBRACK":
            expr = self.parse_expression(strict=strict)
            if expr is None:
                raise SyntaxError("Expected expression inside brackets")
            return ASTNode(
                "print",
                value=None,
                children=[expr],
                line=print_tok.line,
                col=print_tok.col,
            )

        # Otherwise, a single literal or identifier
        tok = self.match(
            "IDENT", "NUMBER", "FLOAT", "STRING", "LITERAL", "THIS", strict=False
        )
        if tok is None:
            return ASTNode(
                "print", value=None, children=[], line=print_tok.line, col=print_tok.col
            )

        kind = "identifier" if tok.type in ("IDENT", "THIS") else tok.type.lower()
        child = ASTNode(kind, tok.value, line=tok.line, col=tok.col)
        return ASTNode(
            "print",
            value=child.value,
            children=[child],
            line=print_tok.line,
            col=print_tok.col,
        )

    def parse_input(self, strict: bool = True) -> ASTNode:
        """Parse an INPUT statement from stdin or FROM file path."""
        tok = self.match("INPUT", strict=strict)
        if tok is None:
            raise SyntaxError("Expected 'INPUT' keyword")

        if self.current().type == "DELIM_FROM":
            self.match("DELIM_FROM", strict=strict)
            f = self.match("IDENT", "STRING", strict=strict)
            if f is None:
                raise SyntaxError("Expected file path after 'FROM'")

            varname = (
                f.value.strip('"')
                .strip("'")
                .split("/")[-1]
                .split("\\")[-1]
                .split(".")[0]
            )

            return ASTNode(
                "input_from_file",
                value=varname,
                children=[ASTNode("file", f.value, line=f.line, col=f.col)],
                line=tok.line,
                col=tok.col,
            )

        var_tok = self.match("IDENT", strict=strict)
        if var_tok is None:
            raise SyntaxError("Expected variable name after INPUT")

        node = ASTNode(
            "input", value=var_tok.value, children=[], line=tok.line, col=tok.col
        )

        if self.current().type == "COLON":
            self.match("COLON", strict=strict)
            type_tok = self.match("IDENT", strict=strict)
            if type_tok is None:
                raise SyntaxError("Expected type after colon in input")
            node.type = type_tok.value.lower()

        return node

    def parse_import(self, strict: bool = True) -> ASTNode:
        """Parse an IMPORT statement to load a module from string path."""
        tok = self.match("IMPORT", "IDENT", strict=strict)
        if tok is None:
            raise SyntaxError("Expected 'IMPORT' keyword")

        path_tok = self.match("STRING", strict=strict)
        if path_tok is None:
            raise SyntaxError("Expected string path after IMPORT")

        return ASTNode("import", path_tok.value, line=tok.line, col=tok.col)

    def parse_expr_entrypoint(self, strict: bool = True) -> list[ASTNode]:
        """Parse a single top-level expression like `x` or `+ x 10`."""
        if self.current().type in (
            "PLUS",
            "SUB",
            "MULT",
            "DIV",
            "MOD",
            "AND",
            "OR",
            "NOT",
        ):
            return [self.parse_prefix_operator(strict=strict)]

        elif self.current().type == "IDENT":
            ident_tok = self.match("IDENT", strict=strict)
            if ident_tok is None:
                raise SyntaxError("Expected identifier")
            return [
                ASTNode(
                    "identifier",
                    ident_tok.value,
                    line=ident_tok.line,
                    col=ident_tok.col,
                )
            ]

        elif self.current().type in ("NUMBER", "FLOAT", "STRING"):
            tok = self.match(self.current().type, strict=strict)
            if tok is None:
                raise SyntaxError("Expected literal value")
            return [ASTNode(tok.type.lower(), tok.value, line=tok.line, col=tok.col)]

        else:
            raise SyntaxError(f"Unrecognized expression: {self.current()}")

    def parse_module(self, strict: bool = True) -> ASTNode:
        """
        Parse a module declaration of the form:

            MODULE Name
            Or, if the lexer emitted it as a generic IDENT “MODULE”:
            IDENT('MODULE') Name
        """
        tok = self.match("MODULE", "IDENT", strict=strict)
        if tok is None:
            raise SyntaxError("Expected 'MODULE' keyword")

        name_tok = self.match("IDENT", strict=strict)
        if name_tok is None:
            raise SyntaxError("Expected module name after 'MODULE'")

        return ASTNode("module", value=name_tok.value, line=tok.line, col=tok.col)

    def parse_expression(self, strict: bool = True) -> ASTNode:
        """Parse a prefix-style bracketed expression like `[PLUS x 1]`."""
        if self.current().type != "LBRACK":
            if strict:
                raise SyntaxError(
                    f"Expected '[' to start expression, got {self.current()}"
                )
            raise SyntaxError(
                "Non-strict mode no longer supports non-bracketed expression entry"
            )

        open_tok = self.match("LBRACK", strict=strict)
        if open_tok is None:
            raise SyntaxError("Expected '[' to open expression")

        close_type = "RBRACK"

        if self.current().type == close_type:
            self.match(close_type, strict=strict)
            return ASTNode("empty", None, line=open_tok.line, col=open_tok.col)

        op_tok = self.match(*self.expression_ops, strict=strict)
        if op_tok is None:
            raise SyntaxError("Expected operator at start of expression")

        if op_tok.type == "CALL":
            if self.current().type == "IDENT":
                if self.peek().type == "DOT":
                    callee = self.parse_member_access(strict=strict)
                else:
                    ident_tok = self.match("IDENT", strict=strict)
                    if ident_tok is None:
                        raise SyntaxError("Expected identifier after CALL")
                    callee = ASTNode(
                        "identifier",
                        ident_tok.value,
                        line=ident_tok.line,
                        col=ident_tok.col,
                    )

                if self.current().type == "LPAREN":
                    self.match("LPAREN", strict=strict)
                    args = []
                    if self.current().type != "RPAREN":
                        while True:
                            args.append(self.parse_expr_or_literal(strict=strict))
                            if self.current().type == "COMMA":
                                self.match("COMMA", strict=strict)
                            else:
                                break
                    self.match("RPAREN", strict=strict)
                    self.match(close_type, strict=strict)
                    return ASTNode(
                        "call",
                        value=callee,
                        children=args,
                        line=op_tok.line,
                        col=op_tok.col,
                    )

                args = []
                while self.current().type != close_type:
                    args.append(self.parse_expr_or_literal(strict=strict))

                self.match(close_type, strict=strict)
                return ASTNode(
                    "call",
                    value=callee,
                    children=args,
                    line=op_tok.line,
                    col=op_tok.col,
                )

            raise SyntaxError("CALL must be followed by identifier or member access")

        children: list[ASTNode] = []
        while self.current().type != close_type:
            if self.current().type == "EOF":
                raise SyntaxError("Expected closing ']', got EOF")

            if self.current().type == "COMMA":
                self.match("COMMA", strict=strict)
                continue

            if self.current().type == "LBRACK":
                child = self.parse_expression(strict=strict)
            elif self.current().type == "IDENT" and self.peek().type == "DOT":
                child = self.parse_member_access(strict=strict)
            else:
                tok = self.match(
                    "NUMBER",
                    "FLOAT",
                    "STRING",
                    "LITERAL",
                    "IDENT",
                    "THIS",
                    strict=strict,
                )
                if tok is None:
                    raise SyntaxError(
                        f"Unexpected token in expression: {self.current()}"
                    )
                kind = (
                    "identifier" if tok.type in ("IDENT", "THIS") else tok.type.lower()
                )
                child = ASTNode(kind, tok.value, line=tok.line, col=tok.col)

            children.append(child)

        self.match(close_type, strict=strict)

        if op_tok.type in self.unary_ops and len(children) != 1:
            raise SyntaxError(f"{op_tok.type} requires 1 operand")
        elif op_tok.type not in self.unary_ops and len(children) < 2:
            raise SyntaxError(f"{op_tok.type} requires at least 2 operands")

        if op_tok.type in self.arith_ops:
            kind = "arith"
        elif op_tok.type in self.bool_ops:
            kind = "bool"
        elif op_tok.type in self.comparison_ops:
            kind = "compare"
        elif op_tok.type == "PROP":
            kind = "propagate"
        else:
            kind = "unknown"

        return ASTNode(
            kind,
            value=op_tok.type,
            line=op_tok.line,
            col=op_tok.col,
            children=children,
        )

    def parse_loop_range(self, strict: bool = True) -> ASTNode:
        """Parse a FOR loop with optional AT (start) and BY (step) clauses."""
        loop_tok = self.match("LOOP_FOR", strict=strict)
        assert loop_tok is not None  # for mypy

        var_tok = self.match("IDENT", strict=strict)
        assert var_tok is not None  # for mypy

        def parse_signed_number() -> Token:
            if self.current().type == "SUB":
                self.match("SUB", strict=strict)
                num_tok = self.match("NUMBER", "FLOAT", strict=strict)
                assert num_tok is not None  # for mypy
                num_tok.value = f"-{num_tok.value}"
                return num_tok
            num_tok = self.match("NUMBER", "FLOAT", strict=strict)
            assert num_tok is not None  # for mypy
            return num_tok

        if self.current().type == "DELIM_AT":
            self.match("DELIM_AT", strict=strict)
            start_tok = parse_signed_number()
        else:
            start_tok = Token("NUMBER", "0", var_tok.line, var_tok.col)

        self.match("DELIM_TO", strict=strict)
        end_tok = parse_signed_number()

        step_tok: Token | None = None

        is_delim_by = self.current().type == "DELIM_BY"  # split conditional
        if is_delim_by:
            print(">> HIT 703: Found DELIM_BY")  # Debug marker for coverage
            self.match("DELIM_BY", strict=strict)
            step_tok = parse_signed_number()
            assert step_tok is not None  # for mypy
        else:
            pass  # pragma: no cover

        body = self.parse_block(strict=strict)
        if not body:
            raise SyntaxError("FOR loop body cannot be empty")

        range_children = [
            ASTNode("identifier", var_tok.value, line=var_tok.line, col=var_tok.col),
            ASTNode(
                "number" if start_tok.type == "NUMBER" else "float",
                start_tok.value,
                line=start_tok.line,
                col=start_tok.col,
            ),
            ASTNode(
                "number" if end_tok.type == "NUMBER" else "float",
                end_tok.value,
                line=end_tok.line,
                col=end_tok.col,
            ),
        ]

        has_step_tok = step_tok is not None  # split conditional
        if has_step_tok:
            assert step_tok is not None  # for mypy
            print(">> HIT 726: step_tok assigned")  # Debug marker for coverage
            range_children.append(
                ASTNode(
                    "number" if step_tok.type == "NUMBER" else "float",
                    step_tok.value,
                    line=step_tok.line,
                    col=step_tok.col,
                )
            )
        else:
            pass  # pragma: no cover

        return ASTNode(
            "loop",
            value="FOR",
            line=loop_tok.line,
            col=loop_tok.col,
            children=[
                ASTNode("range", None, range_children),
                *body,
            ],
        )
