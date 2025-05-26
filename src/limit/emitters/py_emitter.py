"""
Translates LIMIT AST nodes into executable Python code.

This module defines the `PythonEmitter` class, responsible for converting LIMIT's abstract syntax trees (ASTs)
into valid and idiomatic Python source code. It is used by the `Transpiler` during the backend phase of
LIMIT's compilation process.

Supported Features:
    - Expressions: arithmetic, boolean, comparison, function calls, object creation, member access
    - Statements: assignment, print, return, propagate, break, continue, import
    - Control flow: IF/ELSE blocks, WHILE loops, FOR loops with optional range/step
    - Classes: inheritance with automatic `super().__init__()` injection
    - Functions: with parameter parsing and optional return type annotations
    - Modules: `EXPORT` support via Python `__all__` list
    - Expression emission and full AST walking

Behavior:
    - Emits structured Python code with proper indentation.
    - Maintains a code buffer (`lines`) which can be retrieved using `get_output()`.

Raises:
    - `TypeError`: If invalid AST structures are encountered during emission.
    - `NotImplementedError`: If an unrecognized AST kind has no corresponding emitter.

This emitter is the backend target for generating runnable Python from LIMIT programs.
"""

from limit.limit_ast import ASTNode


class PythonEmitter:
    """Emits Python code from LIMIT AST nodes.

    This class converts abstract syntax trees (AST) from the LIMIT language into
    valid, runnable Python code. It implements emitters for all major syntax
    constructs including expressions, statements, control flow, functions, classes,
    and module-level features.

    Attributes:
        lines (list[str]): Accumulated lines of emitted Python code.
        indent (int): Current indentation level for emitted code blocks.
        exports (list[str]): List of exported symbols to include in __all__.

    Methods:
        get_output(): Returns the full emitted Python code as a string.
        emit_expr(node): Emits a Python expression from an AST node.
        _visit(node): Dispatches to the appropriate emit_* method for a node.
    """

    def __init__(self) -> None:
        """
        Initializes the emitter with an empty code buffer, zero indentation, and no exports.
        """
        self.lines: list[str] = []
        self.indent = 0
        self.exports: list[str] = []

    def indent_str(self) -> str:
        return "    " * self.indent

    def get_output(self) -> str:
        """
        Returns the full emitted Python code as a single string.

        Returns
        -------
        str
            The joined lines of Python code, with `__all__` inserted if exports are present.
        """
        if self.exports:
            self.lines.insert(0, f"__all__ = {self.exports!r}")
        return "\n".join(self.lines)

    def emit_literal(self, node: ASTNode) -> str:
        """
        Emits a binary arithmetic expression (e.g., +, -, *, /, %) as Python syntax.

        Parameters
        ----------
        node : ASTNode
            An AST node with operator as value and two children.

        Returns
        -------
        str
            The emitted Python expression.
        """
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
        """
        Emits a boolean expression with `AND`, `OR`, `NOT`, or `TRUTHY`.

        Parameters
        ----------
        node : ASTNode
            The boolean AST node.

        Returns
        -------
        str
            A valid Python boolean expression.
        """
        if node.value == "NOT":
            e = self.emit_expr(node.children[0])
            return f"(not {e})"
        if node.value == "TRUTHY":
            e = self.emit_expr(node.children[0])
            return f"bool({e})"
        left = self.emit_expr(node.children[0])
        right = self.emit_expr(node.children[1])
        op = "and" if node.value == "AND" else "or"
        return f"({left} {op} {right})"

    def emit_expr_member(self, node: ASTNode) -> str:
        """
        Emits attribute access like `obj.attr`.

        Parameters
        ----------
        node : ASTNode
            The member access node.

        Returns
        -------
        str
            The emitted Python member access string.
        """
        base = self.emit_expr(node.children[0])
        return f"{base}.{str(node.value)}"

    def emit_expr_call(self, node: ASTNode) -> str:
        """
        Emits a function or method call expression.

        Parameters
        ----------
        node : ASTNode
            A call node with the callee as value and arguments as children.

        Returns
        -------
        str
            A valid Python function call.
        """
        if not isinstance(node.value, ASTNode):
            raise TypeError("Expected ASTNode for call callee")
        callee = self.emit_expr(node.value)
        args = ", ".join(self.emit_expr(c) for c in node.children)
        return f"{callee}({args})"

    def emit_expr_new(self, node: ASTNode) -> str:
        """
        Emits object instantiation using a class name and arguments.

        Parameters
        ----------
        node : ASTNode
            The new object creation node.

        Returns
        -------
        str
            Python code for calling the class constructor.
        """
        args = ", ".join(self.emit_expr(c) for c in node.children)
        return f"{str(node.value)}({args})" if args else f"{str(node.value)}()"

    def emit_assign(self, node: ASTNode) -> None:
        """
        Emits an assignment statement.

        Parameters
        ----------
        node : ASTNode
            An assignment node with lhs and rhs children.
        """
        rhs = self.emit_expr(node.children[1])
        lhs = self.emit_expr(node.children[0])
        self.lines.append(f"{self.indent_str()}{lhs} = {rhs}")

    def emit_print(self, node: ASTNode) -> None:
        """
        Emits a `print` statement.

        If no children are present, emits `print()` with no arguments.
        Otherwise, emits `print(expr)` using the first child as the expression.

        Parameters
        ----------
        node : ASTNode
            A print node with zero or one expression child.
        """
        if not node.children:
            self.lines.append(f"{self.indent_str()}print()")
        else:
            e = self.emit_expr(node.children[0])
            self.lines.append(f"{self.indent_str()}print({e})")

    def emit_return(self, node: ASTNode) -> None:
        """
        Emits a `return` statement.

        If the node has children, emits `return expr` using the first child as the return value.
        If no children are present, emits a bare `return`.

        Parameters
        ----------
        node : ASTNode
            A return node with optional expression child.
        """
        if node.children:
            v = self.emit_expr(node.children[0])
            self.lines.append(f"{self.indent_str()}return {v}")
        else:
            self.lines.append(f"{self.indent_str()}return")

    def emit_propagate(self, node: ASTNode) -> None:
        """
        Emits a `PROPAGATE`-style conditional return.

        Evaluates the expression and stores it in a temporary variable.
        Emits `if __tmp: return __tmp` to return only if the result is truthy.

        Parameters
        ----------
        node : ASTNode
            A propagate node containing a single expression child.
        """
        tmp = "__tmp"
        e = node.children[0]
        if not isinstance(e, ASTNode):
            raise TypeError("Expected ASTNode in propagate")

        expr: str = self.emit_expr(e)
        self.lines.append(f"{self.indent_str()}{tmp} = {expr}")
        self.lines.append(f"{self.indent_str()}if {tmp}: return {tmp}")

    def emit_if(self, node: ASTNode) -> None:
        """
        Emits an `if` statement with optional `else` block.

        Parameters
        ----------
        node : ASTNode
            The if-node with condition in `value`, body in `children`, and optional `else_children`.
        """
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

    def emit_call(self, node: ASTNode) -> None:
        """
        Emits a function or method call as a standalone statement.

        Converts the call expression into a line of Python code.

        Parameters
        ----------
        node : ASTNode
            A call node with the callee in `value` and arguments in `children`.
        """
        self.lines.append(f"{self.indent_str()}{self.emit_expr_call(node)}")

    def emit_expr_stmt(self, node: ASTNode) -> None:
        """
        Emits a standalone expression statement.

        Typically used for expressions with side effects that are not assigned.

        Parameters
        ----------
        node : ASTNode
            A node containing one expression child to be evaluated.
        """
        expr_code = self.emit_expr(node.children[0])
        self.lines.append(f"{self.indent_str()}{expr_code}")

    def emit_try(self, node: ASTNode) -> None:
        """
        Emits a try-catch-finally block.

        Parameters
        ----------
        node : ASTNode
            The try block with children containing `try`, `catch`, and optionally `finally`.
        """
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
        """
        Emits an `import` statement.

        Parameters
        ----------
        node : ASTNode
            An import node with the module name in `value`.
        """
        self.lines.append(f"{self.indent_str()}import {node.value}")

    def emit_break(self, node: ASTNode) -> None:
        """
        Emits a `break` statement to exit a loop.

        Parameters
        ----------
        node : ASTNode
            A break node with no children.
        """
        self.lines.append(f"{self.indent_str()}break")

    def emit_continue(self, node: ASTNode) -> None:
        """
        Emits a `continue` statement to skip to the next iteration of a loop.

        Parameters
        ----------
        node : ASTNode
            A continue node with no children.
        """
        self.lines.append(f"{self.indent_str()}continue")

    def emit_module(self, node: ASTNode) -> None:
        pass

    def emit_export(self, node: ASTNode) -> None:
        """
        Emits an export declaration by adding the symbol to `__all__`.

        Also emits the associated child node, if present.

        Parameters
        ----------
        node : ASTNode
            An export node with the exported symbol in `value` and optional child.
        """
        self.exports.append(str(node.value))
        if node.children:
            self._visit(node.children[0])

    def emit_expr_compare(self, node: ASTNode) -> str:
        """
        Emits a comparison expression (e.g., `<`, `>`, `==`, etc.).

        Parameters
        ----------
        node : ASTNode
            A compare node with two expression children and a comparison operator in `value`.

        Returns
        -------
        str
            The emitted Python comparison expression.
        """
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
        """
        Emits an input statement that assigns user input to a variable.

        Converts input to `int` or `float` if a type is specified, otherwise uses raw `input()`.

        Parameters
        ----------
        node : ASTNode
            An input node with the target variable in `value` and optional type in `type`.
        """
        line = self.indent_str()
        var = node.value
        if getattr(node, "type", None) == "int":
            self.lines.append(f"{line}{var} = int(input())")
        elif node.type == "float":
            self.lines.append(f"{line}{var} = float(input())")
        else:
            self.lines.append(f"{line}{var} = input()")

    def emit_input_from_file(self, node: ASTNode) -> None:
        """
        Emits code to read a file's contents into a variable.

        If the file is not found, assigns `None` to the variable.

        Parameters
        ----------
        node : ASTNode
            A node with the target variable in `value` and a single child representing the filename.
        """
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

    def emit_loop(self, node: ASTNode) -> None:
        """
        Emits either a `for` or `while` loop depending on node type.

        Parameters
        ----------
        node : ASTNode
            A loop node containing loop header and body.
        """
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
            elif len(range_node.children) == 4:  # pragma: no branch
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
        """
        Emits a `pass` statement, typically used for stubbed functions or classes.

        Parameters
        ----------
        node : ASTNode
            A skip node with no operational content.
        """
        self.lines.append(f"{self.indent_str()}pass")

    def emit_expr(self, node: ASTNode) -> str:
        """
        Dispatches expression emission based on node kind.

        Parameters
        ----------
        node : ASTNode
            The expression node to emit.

        Returns
        -------
        str
            The resulting Python code for the expression.

        Raises
        ------
        NotImplementedError
            If no emitter exists for the node kind.
        """
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
                # pylint: disable=not-callable
                return str(method(node))

        fallback = getattr(self, f"emit_{kind}", None)
        if callable(fallback):
            # pylint: disable=not-callable
            return str(fallback(node))

        raise NotImplementedError(f"No expression emitter for kind '{kind}'")

    def _visit(self, node: ASTNode) -> str | None:
        """
        Dispatches an AST node to the appropriate emit method.

        Handles both expressions and statements based on the node's kind.
        Returns the emitted expression string if applicable, otherwise emits in-place.

        Parameters
        ----------
        node : ASTNode
            The AST node to process.

        Returns
        -------
        str | None
            The emitted expression string, or None for statement nodes.

        Raises
        ------
        NotImplementedError
            If no emitter is defined for the node kind.
        """
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
        # pylint: disable=not-callable
        return str(meth(node))

    def emit_class(self, node: ASTNode) -> None:
        """
        Emits a class definition with optional inheritance and methods.

        Parameters
        ----------
        node : ASTNode
            The class node with name in `value` and children for body and `extends`.
        """
        base = "object"
        new_children = []

        # Determine base class and filter out 'extends'
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
            if child.kind == "func":  # pragma: no branch
                child._in_class = True
                if child.value == "init":
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
                        child.children.insert(
                            0, ASTNode(kind="expr_stmt", children=[super_call])
                        )

            self._visit(child)

        self.indent -= 1

    def emit_func(self, node: ASTNode) -> None:
        """
        Emits a function or method definition.

        Parameters
        ----------
        node : ASTNode
            The function node, with name in `value`, params in `type`, and body in `children`.
        """
        if isinstance(node.type, str):
            raw_params = node.type.split(",")
        elif isinstance(node.type, list):
            raw_params = node.type
        else:
            raw_params = []

        is_method = hasattr(node, "_in_class") and node._in_class is True
        cleaned = [p.strip() for p in raw_params if p.strip()]
        if is_method and "self" not in cleaned:
            cleaned.insert(0, "self")

        sig = f"def {node.value}({', '.join(cleaned)}):"
        if node.return_type:
            sig = sig[:-1] + f" -> {node.return_type}:"

        self.lines.append(f"{self.indent_str()}{sig}")
        self.indent += 1
        if not node.children:
            self.lines.append(f"{self.indent_str()}pass")
        for stmt in node.children:
            self._visit(stmt)
        self.indent -= 1
