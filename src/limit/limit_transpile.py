"""
Provides the `Transpiler` class and emitter interfaces for converting LIMIT ASTs into code.

Classes and Features:
    - Emitter (Protocol): Interface for all backend emitters. Requires `__init__` and `get_output`.
    - PythonEmitter: Concrete emitter that translates LIMIT AST nodes to Python code.
    - JavaScriptEmitter, CEmitter: Stubs for future backends.
    - Transpiler: Uses the appropriate emitter based on the selected target (e.g., "py", "js", "c")
      and dispatches AST nodes to corresponding `emit_*` methods.

Usage:
    The Transpiler takes a list of `ASTNode` instances and returns code in the desired output language.

Example:
    >>> transpiler = Transpiler("py")
    >>> output_code = transpiler.transpile(ast_nodes)

Raises:
    ValueError: If the target language is not supported.
    TypeError: If the AST contains invalid node types.
    NotImplementedError: If the emitter lacks an `emit_*` method for a node kind.
"""

from typing import Protocol

from limit.emitters.py_emitter import PythonEmitter
from limit.limit_ast import ASTNode


class Emitter(Protocol):  # pragma: no cover
    """Protocol for all LIMIT language emitters.

    Emitters are responsible for converting AST nodes into target language code.
    Each emitter must implement a constructor and a method to retrieve the final output.

    Methods:
        __init__(): Initializes the emitter.
        get_output(): Returns the complete emitted code as a string.
    """

    def __init__(self) -> None: ...  # pragma: no cover

    def get_output(self) -> str: ...  # pragma: no cover


# Placeholder stubs for future emitters that conform to Emitter
class JavaScriptEmitter(Emitter):
    """Placeholder for JavaScript code emitter.

    Raises:
        NotImplementedError: Always, since this emitter is not yet implemented.
    """

    def __init__(self) -> None:
        raise NotImplementedError("JavaScriptEmitter is not yet implemented.")

    def get_output(self) -> str:
        raise NotImplementedError()


class CEmitter(Emitter):
    """Placeholder for C code emitter.

    Raises:
        NotImplementedError: Always, since this emitter is not yet implemented.
    """

    def __init__(self) -> None:
        raise NotImplementedError("CEmitter is not yet implemented.")

    def get_output(self) -> str:
        raise NotImplementedError()


EmitterType = type[Emitter]
"""Alias for a concrete Emitter class type."""


class Transpiler:
    """Dispatches LIMIT AST nodes to the appropriate target language emitter.

    This class selects a backend emitter (e.g., Python, JavaScript, C) based on
    the specified target language and invokes emitter methods for each AST node.

    Attributes:
        emitter (Emitter): The selected emitter instance for the output target.
    """

    def __init__(self, target: str) -> None:
        """Initializes the transpiler with the desired output target.

        Args:
            target: The desired output language ("py", "js", "c", etc.).

        Raises:
            ValueError: If the target language is not supported.
        """
        emitters: dict[str, EmitterType] = {
            "py": PythonEmitter,
            "python": PythonEmitter,
            "js": JavaScriptEmitter,
            "c": CEmitter,
        }
        target = target.lower()
        if target not in emitters:
            raise ValueError(f"Unknown transpilation target: {target!r}")
        self.emitter: Emitter = emitters[target]()

    def transpile(self, ast: list[ASTNode]) -> str:
        """Transpiles a list of AST nodes into source code for the selected target.

        Args:
            ast: A list of ASTNode objects representing the LIMIT program.

        Returns:
            The emitted source code as a string.

        Raises:
            TypeError: If any element in the AST list is not an ASTNode.
        """
        if not all(isinstance(node, ASTNode) for node in ast):
            raise TypeError("All items in AST must be ASTNode instances.")
        for node in ast:
            self._visit(node)
        return self.emitter.get_output()

    def _visit(self, node: ASTNode) -> None:
        """Invokes the appropriate emit method on the emitter for a given AST node.

        Args:
            node: The ASTNode to emit code for.

        Raises:
            NotImplementedError: If the emitter does not support the node kind.
        """
        method_name = f"emit_{node.kind}"
        if hasattr(self.emitter, method_name):
            emit_method = getattr(self.emitter, method_name)
            emit_method(node)
        else:
            raise NotImplementedError(
                f"No emitter method for node kind '{node.kind}' "
                f"(line {node.line}, col {node.col})"
            )
