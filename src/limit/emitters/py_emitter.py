# src/limit/emitters/py_emitter.py

from limit.limit_ast import ASTNode


class PythonEmitter:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.indent = 0
        self.exports: list[str] = []

    def indent_str(self) -> str:
        return "    " * self.indent

    def get_output(self) -> str:
        if self.exports:
            self.lines.insert(0, f"__all__ = {self.exports!r}")
        return "\n".join(self.lines)

    def emit_literal(self, node: ASTNode) -> str:
        val = str(node.value)
        if val == "TRUE":
            return "True"
        if val == "FALSE":
            return "False"
        if val == "NULL":
            return "None"
        raise ValueError(f"Unknown literal: {node.value}")

    def emit_identifier(self, node: ASTNode) -> str:
        return str(node.value)

    def emit_number(self, node: ASTNode) -> str:
        return str(node.value)

    def emit_float(self, node: ASTNode) -> str:
        return str(node.value)

    def emit_string(self, node: ASTNode) -> str:
        return repr(str(node.value))

    def emit_file(self, node: ASTNode) -> str:
        return repr(str(node.value))

    def emit_expr_empty(self, node: ASTNode) -> str:
        return "None"

    emit_expr_identifier = emit_identifier
    emit_expr_number = emit_number
    emit_expr_float = emit_float
    emit_expr_string = emit_string
    emit_expr_file = emit_file
    emit_expr_literal = emit_literal
    emit_expr_empty = emit_expr_empty

    def emit_expr_arith(self, node: ASTNode) -> str:
        left = self.emit_expr(node.children[0])
        right = self.emit_expr(node.children[1])
        op = {"PLUS": "+", "SUB": "-", "MULT": "*", "DIV": "/", "MOD": "%"}[
            str(node.value)
        ]
        return f"({left} {op} {right})"

    def emit_expr_bool(self, node: ASTNode) -> str:
        if node.value == "NOT":
            e = self.emit_expr(node.children[0])
            return f"(not {e})"
        left = self.emit_expr(node.children[0])
        right = self.emit_expr(node.children[1])
        op = "and" if node.value == "AND" else "or"
        return f"({left} {op} {right})"

    def emit_expr_member(self, node: ASTNode) -> str:
        base = self.emit_expr(node.children[0])
        return f"{base}.{str(node.value)}"

    def emit_expr_call(self, node: ASTNode) -> str:
        if not isinstance(node.value, ASTNode):
            raise TypeError("Expected ASTNode for call callee")
        callee = self.emit_expr(node.value)
        args = ", ".join(self.emit_expr(c) for c in node.children)
        return f"{callee}({args})"

    def emit_expr_new(self, node: ASTNode) -> str:
        args = ", ".join(self.emit_expr(c) for c in node.children)
        return f"{str(node.value)}({args})" if args else f"{str(node.value)}()"

    def emit_assign(self, node: ASTNode) -> None:
        rhs = self.emit_expr(node.children[1])
        lhs = self.emit_expr(node.children[0])
        self.lines.append(f"{self.indent_str()}{lhs} = {rhs}")

    def emit_print(self, node: ASTNode) -> None:
        if not node.children:
            self.lines.append(f"{self.indent_str()}print()")
        else:
            e = self.emit_expr(node.children[0])
            self.lines.append(f"{self.indent_str()}print({e})")

    def emit_return(self, node: ASTNode) -> None:
        if node.children:
            v = self.emit_expr(node.children[0])
            self.lines.append(f"{self.indent_str()}return {v}")
        else:
            self.lines.append(f"{self.indent_str()}return")

    def emit_propagate(self, node: ASTNode) -> None:
        tmp = "__tmp"
        e = node.children[0]
        if not isinstance(e, ASTNode):
            raise TypeError("Expected ASTNode in propagate")

        expr: str = self.emit_expr(e)
        self.lines.append(f"{self.indent_str()}{tmp} = {expr}")
        self.lines.append(f"{self.indent_str()}if {tmp}: return {tmp}")

    def emit_if(self, node: ASTNode) -> None:
        if not isinstance(node.value, ASTNode):
            raise TypeError("Expected ASTNode in value of IF condition")
        cond_expr = self.emit_expr(
            node.value
        )  # Fix: emit condition from node.value, not children
        self.lines.append(f"{self.indent_str()}if {cond_expr}:")
        self.indent += 1
        for stmt in node.children:
            self._visit(stmt)
        self.indent -= 1

        if getattr(node, "else_children", None):
            self.lines.append(f"{self.indent_str()}else:")
            self.indent += 1
            for stmt in node.else_children:
                self._visit(stmt)
            self.indent -= 1

    def emit_func(self, node: ASTNode) -> None:
        params = ", ".join(str(p) for p in (node.type or []))
        sig = f"def {node.value}({params}):"
        if node.return_type:
            sig = sig[:-1] + f" -> {node.return_type}:"
        self.lines.append(f"{self.indent_str()}{sig}")
        self.indent += 1
        if not node.children:
            self.lines.append(f"{self.indent_str()}pass")
        for stmt in node.children:
            self._visit(stmt)
        self.indent -= 1

    def emit_call(self, node: ASTNode) -> None:
        self.lines.append(f"{self.indent_str()}{self.emit_expr_call(node)}")

    def emit_expr_stmt(self, node: ASTNode) -> None:
        expr_code = self.emit_expr(node.children[0])
        self.lines.append(f"{self.indent_str()}{expr_code}")

    def emit_try(self, node: ASTNode) -> None:
        self.lines.append(f"{self.indent_str()}try:")
        self.indent += 1
        for c in node.children:
            if c.kind in ("catch", "finally"):
                break
            self._visit(c)
        self.indent -= 1

        for c in node.children:
            if c.kind == "catch":
                header = (
                    f"except Exception as {c.value}:"
                    if c.value
                    else "except Exception:"
                )
                self.lines.append(f"{self.indent_str()}{header}")
                self.indent += 1
                for stmt in c.children:
                    self._visit(stmt)
                self.indent -= 1

        for c in node.children:
            if c.kind == "finally":
                self.lines.append(f"{self.indent_str()}finally:")
                self.indent += 1
                for stmt in c.children:
                    self._visit(stmt)
                self.indent -= 1

    def emit_import(self, node: ASTNode) -> None:
        self.lines.append(f"{self.indent_str()}import {node.value}")

    def emit_break(self, node: ASTNode) -> None:
        self.lines.append(f"{self.indent_str()}break")

    def emit_continue(self, node: ASTNode) -> None:
        self.lines.append(f"{self.indent_str()}continue")

    def emit_module(self, node: ASTNode) -> None:
        pass

    def emit_export(self, node: ASTNode) -> None:
        self.exports.append(str(node.value))
        if node.children:
            self._visit(node.children[0])

    def emit_expr_compare(self, node: ASTNode) -> str:
        left = self.emit_expr(node.children[0])
        right = self.emit_expr(node.children[1])
        op = {
            "LT": "<",
            "GT": ">",
            "EQ": "==",
            "NE": "!=",
            "LE": "<=",
            "GE": ">=",
        }.get(str(node.value), str(node.value))
        return f"({left} {op} {right})"

    def emit_input(self, node: ASTNode) -> None:
        line = self.indent_str()
        var = node.value
        if getattr(node, "type", None) == "int":
            self.lines.append(f"{line}{var} = int(input())")
        elif node.type == "float":
            self.lines.append(f"{line}{var} = float(input())")
        else:
            self.lines.append(f"{line}{var} = input()")

    def emit_input_from_file(self, node: ASTNode) -> None:
        child = node.children[0]
        if not isinstance(child, ASTNode):
            raise TypeError("Expected ASTNode in children[0]")

        filename = self.emit_expr(child)
        var = node.value
        self.lines.append(
            f"{self.indent_str()}try:\n"
            f"{self.indent_str()}    {var} = open({filename}).read()\n"
            f"{self.indent_str()}except FileNotFoundError:\n"
            f"{self.indent_str()}    {var} = None"
        )

    def emit_class(self, node: ASTNode) -> None:
        base = "object"
        new_children = []

        for child in node.children:
            if child.kind == "extends":
                if not isinstance(child.value, str):
                    raise TypeError("Expected string in class extends base")
                base = child.value
            else:
                new_children.append(child)

        self.lines.append(f"{self.indent_str()}class {node.value}({base}):")
        self.indent += 1

        if not new_children:
            self.lines.append(f"{self.indent_str()}pass")

        for child in new_children:
            if child.kind == "func" and child.value == "init":
                child.value = "__init__"
                if base != "object":
                    super_call = ASTNode(
                        kind="call",
                        value=ASTNode(
                            kind="member",
                            value="__init__",
                            children=[
                                ASTNode(
                                    kind="call",
                                    value=ASTNode(kind="identifier", value="super"),
                                    children=[],
                                )
                            ],
                        ),
                        children=[],
                    )
                    expr_stmt = ASTNode(kind="expr_stmt", children=[super_call])
                    child.children.insert(0, expr_stmt)
            self._visit(child)

        self.indent -= 1

    def emit_loop(self, node: ASTNode) -> None:
        if node.value == "FOR" and node.children and node.children[0].kind == "range":
            range_node = node.children[0]
            var = str(range_node.children[0].value)

            # Defaults
            start = "0"
            end = None
            step = None

            if len(range_node.children) == 2:
                # FOR i TO N  =>  range(0, N)
                end = str(range_node.children[1].value)
            elif len(range_node.children) == 3:
                # FOR i AT A TO B  =>  range(A, B)
                start = str(range_node.children[1].value)
                end = str(range_node.children[2].value)
            elif len(range_node.children) == 4:
                # FOR i AT A TO B BY C  =>  range(A, B, C)
                start = str(range_node.children[1].value)
                end = str(range_node.children[2].value)
                step = str(range_node.children[3].value)

            range_expr = f"range({start}, {end}" + (f", {step})" if step else ")")
            self.lines.append(f"{self.indent_str()}for {var} in {range_expr}:")
            self.indent += 1
            for stmt in node.children[1:]:
                self._visit(stmt)
            self.indent -= 1

        else:
            # WHILE loop
            cond = self.emit_expr(node.children[0])
            self.lines.append(f"{self.indent_str()}while {cond}:")
            self.indent += 1
            for stmt in node.children[1:]:
                self._visit(stmt)
            self.indent -= 1

        if getattr(node, "else_children", None):
            raise NotImplementedError("ELSE blocks are only valid after IF, not loops.")

    def emit_skip(self, node: ASTNode) -> None:
        self.lines.append(f"{self.indent_str()}pass")

    def emit_expr(self, node: ASTNode) -> str:
        preferred = (
            "arith",
            "bool",
            "compare",
            "member",
            "call",
            "new",
            "identifier",
            "number",
            "float",
            "string",
            "file",
            "empty",
        )
        kind = node.kind

        if kind in preferred:
            method = getattr(self, f"emit_expr_{kind}", None)
            if callable(method):
                return str(method(node))

        fallback = getattr(self, f"emit_{kind}", None)
        if callable(fallback):
            return str(fallback(node))

        raise NotImplementedError(f"No expression emitter for kind '{kind}'")

    def _visit(self, node: ASTNode) -> str | None:
        if node.kind == "skip":
            self.emit_skip(node)
            return None
        if node.kind in (
            "arith",
            "bool",
            "compare",
            "member",
            "call",
            "new",
            "identifier",
            "number",
            "float",
            "string",
            "file",
            "empty",
        ):
            return self.emit_expr(node)

        meth = getattr(self, f"emit_{node.kind}", None)
        if not meth:
            raise NotImplementedError(f"PythonEmitter: no emitter for {node.kind}")
        return str(meth(node))
