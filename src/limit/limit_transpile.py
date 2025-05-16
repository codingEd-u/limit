from typing import Protocol

from limit.emitters.py_emitter import PythonEmitter
from limit.limit_ast import ASTNode


class Emitter(Protocol):
    def __init__(self) -> None:
        ...

    def get_output(self) -> str:
        ...


# Placeholder stubs for future emitters that conform to Emitter
class JavaScriptEmitter(Emitter):
    def __init__(self) -> None:
        raise NotImplementedError("JavaScriptEmitter is not yet implemented.")

    def get_output(self) -> str:
        raise NotImplementedError()


class CEmitter(Emitter):
    def __init__(self) -> None:
        raise NotImplementedError("CEmitter is not yet implemented.")

    def get_output(self) -> str:
        raise NotImplementedError()


EmitterType = type[Emitter]


class Transpiler:
    def __init__(self, target: str):
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
        if not all(isinstance(node, ASTNode) for node in ast):
            raise TypeError("All items in AST must be ASTNode instances.")
        for node in ast:
            self._visit(node)
        return self.emitter.get_output()

    def _visit(self, node: ASTNode) -> None:
        method_name = f"emit_{node.kind}"
        if hasattr(self.emitter, method_name):
            emit_method = getattr(self.emitter, method_name)
            emit_method(node)
        else:
            raise NotImplementedError(
                f"No emitter method for node kind '{node.kind}' "
                f"(line {node.line}, col {node.col})"
            )
