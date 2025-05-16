# limit_parser.py
from __future__ import annotations

from limit.limit_ast import ASTNode
from limit.limit_constants import operator_tokens, token_hashmap
from limit.limit_lexer import Token


class Parser:
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

        LT = operator_tokens[8]
        LE = operator_tokens[9]
        GT = operator_tokens[10]
        GE = operator_tokens[11]
        EQ = operator_tokens[12]
        NE = operator_tokens[13]

        CALL = operator_tokens[14]
        PROP = operator_tokens[15]

        # TOKEN MAPPINGS (PARSER)

        self.unary_ops: set[str] = {NOT}

        self.arith_ops: set[str] = {
            PLUS,
            SUB,
            MULT,
            DIV,
            MOD,
        }

        self.bool_ops: set[str] = {AND, OR, NOT}

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
        self.expression_ops: set[str] = self.prefix_ops | {
            CALL,
            PROP,
        }
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

    def parse_block(self, open_type: str = "LBRACE") -> list[ASTNode]:
        if open_type != "LBRACE":
            raise SyntaxError(
                f"Invalid block delimiter: {open_type}. Blocks must use '{{' and '}}'."
            )

        expected_close = "RBRACE"
        self.match("LBRACE")

        stmts = []
        prev_in_block = self.in_block
        self.in_block = True

        if self.current().type == expected_close:
            self.match(expected_close)
            self.in_block = prev_in_block
            return []

        while self.current().type != expected_close:
            if self.current().type == "EOF":
                raise SyntaxError("Expected closing '}', got EOF")

            pos_before = self.position
            stmt = self.parse_statement()
            pos_after = self.position

            if stmt is not None:
                stmts.append(stmt)
            elif pos_after == pos_before:
                raise SyntaxError(
                    f"Unrecognized statement inside block near: {self.current().value} (type={self.current().type})"
                )

        self.match(expected_close)
        self.in_block = prev_in_block
        return stmts

    def parse_propagate(self) -> ASTNode:
        if not self.in_block:
            raise SyntaxError("Propagate operator `$` only allowed inside functions.")

        prop_tok = self.match("PROP")
        if prop_tok is None:
            raise SyntaxError("Expected '$' propagate operator")

        if self.current().type == "LBRACK":
            expr = self.parse_expression()
            if expr is None:
                raise SyntaxError("Expected valid expression after '$'")
        else:
            tok = self.match("IDENT", "NUMBER", "FLOAT", "STRING", "LITERAL", "THIS")
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

    def parse_try(self) -> ASTNode:
        try_tok = self.match("TRY", "IDENT")
        if try_tok is None:
            raise SyntaxError("Expected 'TRY' keyword")

        try_block = self.parse_block()

        catches = []
        while self.current().type == "CATCH" or (
            self.current().type == "IDENT" and self.current().value == "CATCH"
        ):
            catch_tok = self.match("CATCH", "IDENT")
            if catch_tok is None:
                raise SyntaxError("Expected 'CATCH' keyword")

            if self.current().type == "LPAREN":
                self.match("LPAREN")
                exc_name_tok = self.match("IDENT")
                if exc_name_tok is None:
                    raise SyntaxError("Expected exception name inside CATCH()")
                exc_name = exc_name_tok.value
                self.match("RPAREN")
            elif self.current().type == "IDENT":
                exc_name_tok = self.match("IDENT")
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
            finally_tok = self.match("FINALLY", "IDENT")
            if finally_tok is None:
                raise SyntaxError("Expected 'FINALLY' keyword")
            finally_block = self.parse_block()
            finally_node = ASTNode(
                "finally",
                None,
                finally_block,
                line=finally_tok.line,
                col=finally_tok.col,
            )

        children = try_block
        children.extend(catches)
        if finally_node:
            children.append(finally_node)

        return ASTNode("try", None, children, line=try_tok.line, col=try_tok.col)

    def parse_loop_while(self) -> ASTNode:
        loop_tok = self.match("LOOP_WHILE")
        if loop_tok is None:
            raise SyntaxError("Expected 'WHILE' loop start")

        # Parse condition
        if self.current().type == "LBRACK":
            if self.peek().type == "RBRACK":
                raise SyntaxError("WHILE condition cannot be empty")
            cond_node = self.parse_expression(strict=True)
            if cond_node is None:
                raise SyntaxError("Expected valid condition inside WHILE brackets")
        else:
            tok = self.match("IDENT", "NUMBER", "FLOAT", "STRING", "LITERAL")
            if tok is None:
                raise SyntaxError("Expected condition after WHILE")
            kind = "identifier" if tok.type == "IDENT" else tok.type.lower()
            cond_node = ASTNode(kind, tok.value, line=tok.line, col=tok.col)

        # Require block
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

    def parse(self) -> list[ASTNode]:
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
                    )
            except SyntaxError as e:
                raise e  # fail-fast
        return ast

    def parse_module(self) -> ASTNode:
        """
        Parse a module declaration of the form:
        MODULE Name
        Or, if the lexer emitted it as a generic IDENT “MODULE”:
        IDENT('MODULE') Name
        """
        tok = self.match("MODULE", "IDENT")
        if tok is None:
            raise SyntaxError("Expected 'MODULE' keyword")
        name_tok = self.match("IDENT")
        if name_tok is None:
            raise SyntaxError("Expected module name after 'MODULE'")
        return ASTNode("module", value=name_tok.value, line=tok.line, col=tok.col)

    def parse_expression(self, strict: bool = True) -> ASTNode:
        if self.current().type != "LBRACK":
            if strict:
                raise SyntaxError(
                    f"Expected '[' to start expression, got {self.current()}"
                )
            raise SyntaxError(
                "Non-strict mode no longer supports non-bracketed expression entry"
            )

        open_tok = self.match("LBRACK")
        if open_tok is None:
            raise SyntaxError("Expected '[' to open expression")

        close_type = "RBRACK"

        if self.current().type == close_type:
            self.match(close_type)
            return ASTNode("empty", None, line=open_tok.line, col=open_tok.col)

        op_tok = self.match(*self.expression_ops)
        if op_tok is None:
            raise SyntaxError("Expected operator at start of expression")

        if op_tok.type == "CALL":
            if self.current().type == "IDENT":
                if self.peek().type == "DOT":
                    callee = self.parse_member_access()
                else:
                    ident_tok = self.match("IDENT")
                    if ident_tok is None:
                        raise SyntaxError("Expected identifier after CALL")
                    callee = ASTNode(
                        "identifier",
                        ident_tok.value,
                        line=ident_tok.line,
                        col=ident_tok.col,
                    )

                if self.current().type == "LPAREN":
                    self.match("LPAREN")
                    args = []
                    if self.current().type != "RPAREN":
                        while True:
                            args.append(self.parse_expr_or_literal())
                            if self.current().type == "COMMA":
                                self.match("COMMA")
                            else:
                                break
                    self.match("RPAREN")
                    self.match(close_type)
                    return ASTNode(
                        "call",
                        value=callee,
                        children=args,
                        line=op_tok.line,
                        col=op_tok.col,
                    )

                args = []
                while self.current().type != close_type:
                    args.append(self.parse_expr_or_literal())

                self.match(close_type)
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
                self.match("COMMA")
                continue

            if self.current().type == "LBRACK":
                child = self.parse_expression(strict=strict)
            elif self.current().type == "IDENT" and self.peek().type == "DOT":
                child = self.parse_member_access()
            else:
                tok = self.match(
                    "NUMBER", "FLOAT", "STRING", "LITERAL", "IDENT", "THIS"
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

        self.match(close_type)

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

    def parse_loop_range(self) -> ASTNode:
        loop_tok = self.match("LOOP_FOR")
        if loop_tok is None:
            raise SyntaxError("Expected 'FOR' keyword")

        var_tok = self.match("IDENT")
        if var_tok is None:
            raise SyntaxError("Expected loop variable after 'FOR'")

        def parse_signed_number() -> Token:
            if self.current().type == "SUB":
                self.match("SUB")
                num_tok = self.match("NUMBER", "FLOAT")
                if num_tok is None:
                    raise SyntaxError("Expected number after '-' in signed value")
                num_tok.value = f"-{num_tok.value}"
                return num_tok
            num_tok = self.match("NUMBER", "FLOAT")
            if num_tok is None:
                raise SyntaxError("Expected number")
            return num_tok

        # Optional start value: AT <start>
        if self.current().type == "DELIM_AT":
            self.match("DELIM_AT")
            start_tok = parse_signed_number()
        else:
            start_tok = Token("NUMBER", "0", var_tok.line, var_tok.col)

        # Required end value: TO <end>
        self.match("DELIM_TO")
        end_tok = parse_signed_number()

        # Optional step: BY <step>
        step_tok: Token | None = None
        if self.current().type == "DELIM_BY":
            self.match("DELIM_BY")
            step_tok = parse_signed_number()

        body = self.parse_block()
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
        if step_tok is not None:
            range_children.append(
                ASTNode(
                    "number" if step_tok.type == "NUMBER" else "float",
                    step_tok.value,
                    line=step_tok.line,
                    col=step_tok.col,
                )
            )

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

    def parse_expr_or_literal(self) -> ASTNode:
        if self.current().type == "LBRACK":
            expr = self.parse_expression()
            if expr is None:
                raise SyntaxError("Expected expression inside brackets")
            return expr

        # Allow member access like `a.b.c`
        if self.current().type == "IDENT" and self.peek().type == "DOT":
            return self.parse_member_access()

        tok = self.match(
            "IDENT", "NUMBER", "FLOAT", "STRING", "LITERAL", "THIS", strict=False
        )
        if tok is None:
            raise SyntaxError(f"Expected expression or literal, got {self.current()}")

        kind = "identifier" if tok.type in ("IDENT", "THIS") else tok.type.lower()
        return ASTNode(kind, tok.value, line=tok.line, col=tok.col)

    def parse_class(self) -> ASTNode:
        class_tok = self.match("CLASS")
        if class_tok is None:
            raise SyntaxError("Expected 'CLASS' keyword")

        name_tok = self.match("IDENT")
        if name_tok is None:
            raise SyntaxError("Expected class name after 'CLASS'")

        extends_node: ASTNode | None = None
        if self.current().type == "EXTENDS":
            self.match("EXTENDS")
            base_tok = self.match("IDENT")
            if base_tok is None:
                raise SyntaxError("Expected base class name after 'EXTENDS'")
            extends_node = ASTNode(
                "extends", value=base_tok.value, line=base_tok.line, col=base_tok.col
            )

        self.match("LBRACE")
        body: list[ASTNode] = []
        while self.current().type != "RBRACE":
            stmt = self.parse_statement()
            if stmt is None:
                raise SyntaxError("Invalid statement in class body")
            body.append(stmt)
        self.match("RBRACE")

        children: list[ASTNode] = [extends_node] + body if extends_node else body
        return ASTNode(
            "class",
            value=name_tok.value,
            children=children,
            line=name_tok.line,
            col=name_tok.col,
        )

    def parse_expr_entrypoint(self) -> list[ASTNode]:
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
            expr = self.parse_expression()
            if expr is None:
                raise SyntaxError("Expected expression after operator")
            return [expr]

        elif self.current().type == "IDENT":
            ident_tok = self.match("IDENT")
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
            tok = self.match(self.current().type)
            if tok is None:
                raise SyntaxError("Expected literal value")
            return [ASTNode(tok.type.lower(), tok.value, line=tok.line, col=tok.col)]

        else:
            raise SyntaxError(f"Unrecognized expression: {self.current()}")

    def parse_import(self) -> ASTNode:
        tok = self.match("IMPORT", "IDENT")
        if tok is None:
            raise SyntaxError("Expected 'IMPORT' keyword")

        path_tok = self.match("STRING")
        if path_tok is None:
            raise SyntaxError("Expected string path after IMPORT")

        return ASTNode("import", path_tok.value, line=tok.line, col=tok.col)

    def parse_return(self) -> ASTNode:
        if not self.in_block:
            raise SyntaxError("RETURN is only allowed inside function blocks.")

        tok = self.match("RETURN")
        if tok is None:
            raise SyntaxError("Expected 'RETURN' keyword")

        if self.current().type == "LBRACK":
            expr = self.parse_expression()
            if expr is None:
                raise SyntaxError("Expected expression after RETURN")
            return ASTNode(
                "return",
                children=[expr],
                line=tok.line,
                col=tok.col,
            )

        if self.current().type == "IDENT" and self.peek().type == "DOT":
            member_node = self.parse_member_access()
            return ASTNode(
                "return",
                children=[member_node],
                line=tok.line,
                col=tok.col,
            )

        val = self.match("IDENT", "NUMBER", "FLOAT", "STRING", "LITERAL", strict=False)
        if val is not None:
            kind = "identifier" if val.type == "IDENT" else val.type.lower()
            return ASTNode(
                "return",
                children=[ASTNode(kind, val.value, line=val.line, col=val.col)],
                line=tok.line,
                col=tok.col,
            )

        return ASTNode(
            "return",
            value=None,
            line=tok.line,
            col=tok.col,
        )

    def parse_assignment(self) -> ASTNode:
        assign_tok = self.match("ASSIGN")
        if assign_tok is None:
            raise SyntaxError("Expected '=' assignment operator")

        # LEFT-HAND SIDE
        if self.current().type == "IDENT":
            name_node = self.parse_member_access()
        else:
            raise SyntaxError(f"Invalid assignment target: {self.current()}")

        # RIGHT-HAND SIDE
        # 1) new ClassName(args...)
        if self.current().type == "NEW":
            new_tok = self.match("NEW")
            if new_tok is None:
                raise SyntaxError("Expected 'NEW' keyword")
            class_tok = self.match("IDENT")
            if class_tok is None:
                raise SyntaxError("Expected class name after NEW")
            args = []
            if self.current().type == "LPAREN":
                self.match("LPAREN")
                if self.current().type != "RPAREN":
                    while True:
                        arg_tok = self.match(
                            "IDENT", "NUMBER", "FLOAT", "STRING", "LITERAL", "THIS"
                        )
                        if arg_tok is None:
                            raise SyntaxError("Expected valid argument in NEW call")
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
            value_node = ASTNode(
                "new",
                class_tok.value,
                children=args,
                line=new_tok.line,
                col=new_tok.col,
            )

        # 2) member access
        elif self.current().type == "IDENT" and self.peek().type == "DOT":
            value_node = self.parse_member_access()

        # 3) function call: foo(arg1, arg2)
        elif self.current().type == "IDENT" and self.peek().type == "LPAREN":
            func_tok = self.match("IDENT")
            if func_tok is None:
                raise SyntaxError("Expected function name in call")
            args = []
            self.match("LPAREN")
            if self.current().type != "RPAREN":
                while True:
                    arg_tok = self.match(
                        "IDENT", "NUMBER", "FLOAT", "STRING", "LITERAL", "THIS"
                    )
                    if arg_tok is None:
                        raise SyntaxError("Expected valid argument in function call")
                    kind = (
                        "identifier"
                        if arg_tok.type in ("IDENT", "THIS")
                        else arg_tok.type.lower()
                    )
                    args.append(
                        ASTNode(kind, arg_tok.value, line=arg_tok.line, col=arg_tok.col)
                    )
                    if self.current().type == "COMMA":
                        self.match("COMMA")
                    else:
                        break
            self.match("RPAREN")
            value_node = ASTNode(
                "call",
                func_tok.value,
                children=args,
                line=func_tok.line,
                col=func_tok.col,
            )

        # 4) bracketed expression
        elif self.current().type == "LBRACK":
            value_node = self.parse_expression()
            if value_node is None:
                raise SyntaxError("Expected expression on right-hand side")

        # 5) literal or identifier
        elif self.current().type in (
            "NUMBER",
            "FLOAT",
            "STRING",
            "LITERAL",
            "IDENT",
            "THIS",
        ):
            value_tok = self.match(
                "NUMBER", "FLOAT", "STRING", "LITERAL", "IDENT", "THIS"
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

    def parse_input(self) -> ASTNode:
        tok = self.match("INPUT")
        if tok is None:
            raise SyntaxError("Expected 'INPUT' keyword")

        if self.current().type == "DELIM_FROM":
            self.match("DELIM_FROM")
            f = self.match("IDENT", "STRING")
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

        var_tok = self.match("IDENT")
        if var_tok is None:
            raise SyntaxError("Expected variable name after INPUT")

        node = ASTNode(
            "input", value=var_tok.value, children=[], line=tok.line, col=tok.col
        )

        if self.current().type == "COLON":
            self.match("COLON")
            type_tok = self.match("IDENT")
            if type_tok is None:
                raise SyntaxError("Expected type after colon in input")
            node.type = type_tok.value.lower()

        return node

    def parse_print(self) -> ASTNode:
        """
        Print statement can be:
        ! x
        ! 123
        ! "hello"
        ! [ PLUS x 1 ]
        ! [CALL sum 2 3]
        """
        print_tok = self.match("PRINT")
        if print_tok is None:
            raise SyntaxError("Expected 'PRINT' keyword")

        # Bracketed expression: ! [ ... ]
        if self.current().type == "LBRACK":
            expr = self.parse_expression()
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

    def parse_call(self) -> ASTNode:
        call_tok = self.match("CALL")
        if call_tok is None:
            raise SyntaxError("Expected 'CALL' keyword")

        if self.current().type == "IDENT" and self.peek().type == "LPAREN":
            # CALL sum(2,3)
            ident_tok = self.match("IDENT")
            if ident_tok is None:
                raise SyntaxError("Expected function name after CALL")

            self.match("LPAREN")
            args = []
            if self.current().type != "RPAREN":
                while True:
                    arg = self.parse_expr_or_literal()
                    args.append(arg)
                    if self.current().type == "COMMA":
                        self.match("COMMA")
                    else:
                        break
            self.match("RPAREN")
            return ASTNode(
                "call",
                value=ASTNode(
                    "identifier",
                    ident_tok.value,
                    line=ident_tok.line,
                    col=ident_tok.col,
                ),
                children=args,
                line=call_tok.line,
                col=call_tok.col,
            )

        # CALL sum 2 3
        callee_tok = self.match("IDENT")
        if callee_tok is None:
            raise SyntaxError("Expected callee name after CALL")

        args = []
        while self.current().type in (
            "IDENT",
            "NUMBER",
            "FLOAT",
            "STRING",
            "LITERAL",
            "THIS",
            "LBRACK",
        ):
            args.append(self.parse_expr_or_literal())

        return ASTNode(
            "call",
            value=ASTNode(
                "identifier", callee_tok.value, line=callee_tok.line, col=callee_tok.col
            ),
            children=args,
            line=call_tok.line,
            col=call_tok.col,
        )

    def parse_prefix_operator(self) -> ASTNode:
        op_tok = self.match("PLUS", "SUB", "MULT", "DIV", "MOD", "AND", "NOT")
        if op_tok is None:
            raise SyntaxError("Expected prefix operator")

        def parse_operand() -> ASTNode:
            if self.current().type == "LBRACK":
                expr = self.parse_expression()
                if expr is None:
                    raise SyntaxError("Expected expression in brackets")
                return expr

            tok = self.match("IDENT", "NUMBER")
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

    def parse_export(self) -> ASTNode:
        tok = self.match("EXPORT", "IDENT")
        if tok is None:
            raise SyntaxError("Expected 'EXPORT' keyword")
        assert tok is not None  # for mypy

        name_tok = self.match("IDENT")
        if name_tok is None:
            raise SyntaxError("Expected identifier after EXPORT")
        assert name_tok is not None  # for mypy

        return ASTNode("export", name_tok.value, line=tok.line, col=tok.col)

    def parse_break(self) -> ASTNode:
        if not self.in_block:
            raise SyntaxError("BREAK only allowed inside loop blocks.")

        tok = self.match("BREAK", "IDENT")
        if tok is None:
            raise SyntaxError("Expected 'BREAK' keyword")
        assert tok is not None  # for mypy

        return ASTNode("break", line=tok.line, col=tok.col)

    def parse_continue(self) -> ASTNode:
        if not self.in_block:
            raise SyntaxError("CONTINUE only allowed inside loop blocks.")

        tok = self.match("CONTINUE", "IDENT")
        if tok is None:
            raise SyntaxError("Expected 'CONTINUE' keyword")
        assert tok is not None  # for mypy

        return ASTNode("continue", line=tok.line, col=tok.col)

    def parse_if(self) -> ASTNode:
        if_tok = self.match("IF")
        if if_tok is None:
            raise SyntaxError("Expected 'IF' keyword")
        assert if_tok is not None  # for mypy

        self.match("LPAREN")

        op_tok = self.current()
        if op_tok.type not in self.prefix_ops:
            raise SyntaxError(f"Invalid IF operator: {op_tok}")
        op_tok = self.match_type_only(op_tok.type)
        assert op_tok is not None  # for mypy

        children: list[ASTNode] = []
        while self.current().type != "RPAREN":
            if self.current().type == "EOF":
                raise SyntaxError("Unexpected EOF in IF condition")

            tok = self.match("IDENT", "NUMBER", "FLOAT", "STRING", "LITERAL", "THIS")
            if tok is None:
                raise SyntaxError("Expected token in IF condition")
            kind = "identifier" if tok.type in ("IDENT", "THIS") else tok.type.lower()
            children.append(ASTNode(kind, tok.value, line=tok.line, col=tok.col))
        self.match("RPAREN")

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
            kind = "unknown"

        cond_expr = ASTNode(
            kind, value=op_tok.type, line=op_tok.line, col=op_tok.col, children=children
        )

        if self.current().type != "LBRACE":
            raise SyntaxError("IF body must be enclosed in braces '{ }'")
        then_block = self.parse_block("LBRACE")

        else_block: list[ASTNode] = []
        if self.current().type == "ELSE":
            self.match("ELSE")
            if self.current().type != "LBRACE":
                raise SyntaxError("ELSE body must be enclosed in braces '{ }'")
            else_block = self.parse_block("LBRACE")

        node = ASTNode(
            "if", value=cond_expr, line=if_tok.line, col=if_tok.col, children=then_block
        )
        if else_block:
            node.else_children = else_block
        return node

    def parse_statement(self) -> ASTNode:
        tok = self.current()

        if tok.type == "COMMA":
            self.advance()
            raise SyntaxError("Standalone comma is not a valid statement")

        if tok.type == "PROP":
            return self.parse_propagate()
        if tok.type in ("TRY", "IDENT") and tok.value == "TRY":
            return self.parse_try()
        if tok.type == "ASSIGN":
            return self.parse_assignment()
        if tok.type == "LBRACK":
            return self.parse_expression()
        if tok.type == "IF":
            return self.parse_if()
        if tok.type in ("SKIP", "IDENT") and tok.value == "SKIP":
            return self.parse_skip()
        if tok.type == "FUNC":
            return self.parse_function()
        if tok.type == "CALL":
            return self.parse_call()
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
            if self.peek().type == "IDENT" and self.peek(2).type in (
                "DELIM_TO",
                "DELIM_AT",
            ):
                return self.parse_loop_range()
            return self.parse_loop_block("FOR")
        if tok.type == "LOOP_WHILE":
            return self.parse_loop_while()
        if tok.type == "CLASS":
            return self.parse_class()
        if tok.type in ("PLUS", "SUB", "MULT", "DIV", "MOD", "AND", "NOT"):
            return self.parse_prefix_operator()

        # Fallback: expression statement like `c.getX()`
        try:
            expr = self.parse_expr_or_literal()

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

    def parse_member_access(self) -> ASTNode:
        """
        Parse chained member access expressions of the form:
            obj.y.z
        Consumes IDENT, then any number of (DOT IDENT) pairs.
        """
        ident_tok = self.match("IDENT")
        if ident_tok is None:
            raise SyntaxError("Expected identifier at start of member access")
        assert ident_tok is not None  # for mypy

        node = ASTNode(
            "identifier", ident_tok.value, line=ident_tok.line, col=ident_tok.col
        )

        while self.current().type == "DOT":
            self.match("DOT")
            member_tok = self.match("IDENT")
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

    def parse_skip(self) -> ASTNode:
        tok = self.match("SKIP", "IDENT")
        if tok is None:
            raise SyntaxError("Expected 'SKIP' keyword")
        assert tok is not None  # for mypy

        return ASTNode("skip", line=tok.line, col=tok.col)

    def parse_loop_block(self, kind: str) -> ASTNode:
        """
        Parse FOR { ... } or WHILE { ... } only.
        """
        loop_tok = self.match("LOOP_FOR" if kind == "FOR" else "LOOP_WHILE")
        if loop_tok is None:
            raise SyntaxError(f"Expected '{kind}' loop keyword")
        assert loop_tok is not None  # for mypy

        if self.current().type != "LBRACE":
            raise SyntaxError(f"{kind} loop body must be enclosed in braces '{{}}'")

        body = self.parse_block("LBRACE")
        if not body:
            raise SyntaxError(f"{kind} loop body cannot be empty")

        return ASTNode(
            "loop", value=kind, line=loop_tok.line, col=loop_tok.col, children=body
        )

    def parse_function(self) -> ASTNode:
        func_tok = self.match("FUNC")
        if func_tok is None:
            raise SyntaxError("Expected 'FUNC' keyword")
        assert func_tok is not None  # for mypy

        if self.current().type != "IDENT":
            raise SyntaxError(f"Function name must be IDENT, got {self.current()}")
        name_tok = self.match("IDENT")
        if name_tok is None:
            raise SyntaxError("Expected function name after FUNC")
        assert name_tok is not None  # for mypy

        params: list[str] = []
        if self.current().type == "LPAREN":
            self.match("LPAREN")
            if self.current().type != "RPAREN":
                while True:
                    param_tok = self.match("IDENT")
                    if param_tok is None:
                        raise SyntaxError("Expected parameter name in function")
                    params.append(param_tok.value)
                    if self.current().type == "COMMA":
                        self.match("COMMA")
                    else:
                        break
            self.match("RPAREN")

        return_type: str | None = None
        if self.current().type == "COLON":
            self.match("COLON")
            rt_tok = self.match("IDENT")
            if rt_tok is None:
                raise SyntaxError("Expected return type after ':'")
            return_type = rt_tok.value

        if self.current().type == "LBRACE":
            block = self.parse_block("LBRACE")
        else:
            raise SyntaxError("Function body must start with '{'")

        node = ASTNode(
            "func",
            value=name_tok.value,
            line=func_tok.line,
            col=func_tok.col,
            children=block,
        )
        node.type = ",".join(params) if params else None  # ✅ fix: serialize as str
        if return_type:
            node.return_type = return_type

        return node
