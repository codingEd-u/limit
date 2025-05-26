"""
Defines the abstract syntax tree (AST) node structure for the LIMIT programming language.

Classes:
    ASTNode:
        Represents a node in the syntax tree, used by the parser, emitter, and transpiler.
        Supports recursion, type annotations, and metadata for error reporting and transpilation.

    ASTDict:
        TypedDict representation for serializing ASTNode instances to plain Python dictionaries,
        suitable for JSON output or debugging.

Each ASTNode tracks:
    kind (str): The syntactic construct type (e.g., "func", "if", "call").
    value (Union[str, ASTNode], optional): A raw string or another ASTNode.
    children (list[ASTNode]): Primary child nodes.
    else_children (list[ASTNode]): Used for else/catch/finally blocks.
    type (str, optional): Type annotation for declarations or inputs.
    return_type (str, optional): Return type for functions.
    line (int): Source line number for error messages.
    col (int): Source column number for error messages.

Usage:
    This module serves as the core data structure for LIMIT’s parser output.
    It is also used by the emitter for code generation and by test suites for asserting structure.

Example:
    node = ASTNode("func", value="main", children=[...], type_="int", return_type="str")
"""

from typing import Any, TypedDict, Union


class ASTDict(TypedDict, total=False):
    """
    TypedDict representation of an ASTNode used for serialization.

    This structure defines the shape of a serialized abstract syntax tree (AST) node,
    typically used when converting an ASTNode to a dictionary (e.g., for JSON output,
    debugging, or inspection).

    Fields:
        kind (str): The type of AST node (e.g., "func", "call", "if").
        value (Any): The node's value, which may be a string, nested ASTDict, or literal.
        line (int): Line number in the source code where the node originates.
        col (int): Column number in the source code where the node originates.
        type (Optional[str]): Optional type annotation (e.g., for variables or inputs).
        return_type (Optional[str]): Optional return type annotation (e.g., for functions).
        children (List[ASTDict]): Primary child nodes in the AST hierarchy.
        else_children (List[ASTDict]): Alternate branch nodes (e.g., for 'else', 'catch').
    """

    kind: str
    value: Any
    line: int
    col: int
    type: str | None
    return_type: str | None
    children: list["ASTDict"]
    else_children: list["ASTDict"]


class ASTNode:
    """
    Represents a node in the abstract syntax tree (AST) for the LIMIT language.

    Each node captures a syntactic construct such as a function, expression, loop, or class.
    Nodes are recursively structured and support conversion to dictionary form for serialization.

    Args:
        kind (str): The type of node (e.g., "func", "call", "assign", "if").
        value (Union[str, ASTNode], optional): A literal value or another AST node (e.g., function name or callee).
        children (list[ASTNode], optional): Primary child nodes in the syntax tree.
        line (int): Source line number (default is 0).
        col (int): Source column number (default is 0).
        type_ (str, optional): Optional type annotation (e.g., input type or variable type).
        return_type (str, optional): Optional return type (e.g., for functions or methods).

    Attributes:
        kind (str): Type of the AST node.
        value (Union[str, ASTNode] | None): Value or nested node.
        children (list[ASTNode]): Main child nodes.
        else_children (list[ASTNode]): Alternate path nodes (e.g., else or catch blocks).
        type (str | None): Type annotation.
        return_type (str | None): Return type annotation.
        line (int): Line number in the source file.
        col (int): Column number in the source file.
        _in_class (bool | None): Internal flag indicating if the node was declared inside a class.

    Methods:
        __repr__(): Returns a structured string representation for debugging.
        __eq__(other): Checks structural equality with another ASTNode.
        to_dict(): Converts the node (and all descendants) into a nested dictionary format.
    """

    def __init__(
        self,
        kind: str,
        value: Union[str, "ASTNode"] | None = None,
        children: list["ASTNode"] | None = None,
        line: int = 0,
        col: int = 0,
        type_: str | None = None,
        return_type: str | None = None,
    ):
        self.kind = kind
        self.value = value
        self.children: list["ASTNode"] = children or []
        self.line = line
        self.col = col
        self.type = type_
        self.return_type = return_type
        self.else_children: list["ASTNode"] = []
        self._in_class: bool | None = None

    def __repr__(self) -> str:
        parts = [f"{self.kind}"]
        if self.value is not None:
            parts.append(f"value={repr(self.value)}")
        if self.type is not None:
            parts.append(f"type_={self.type}")
        if self.return_type is not None:
            parts.append(f"return_type={self.return_type}")
        if self.children:
            preview = ", ".join(repr(c) for c in self.children[:3])
            if len(self.children) > 3:
                preview += ", ..."
            parts.append(f"children=[{preview}]")
        if self.else_children:
            preview = ", ".join(repr(c) for c in self.else_children[:3])
            if len(self.else_children) > 3:
                preview += ", ..."
            parts.append(f"else_children=[{preview}]")
        return f"ASTNode({', '.join(parts)})"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ASTNode):
            return False
        values_equal = (
            self.value == other.value
            if not isinstance(self.value, ASTNode)
            else isinstance(other.value, ASTNode) and self.value == other.value
        )
        return (
            self.kind == other.kind
            and values_equal
            and self.line == other.line
            and self.col == other.col
            and self.type == other.type
            and self.return_type == other.return_type
            and self.children == other.children
            and self.else_children == other.else_children
        )

    def to_dict(self) -> ASTDict:
        val: Any = self.value  # use separate var, don't reuse self.value
        if isinstance(val, ASTNode):
            val = val.to_dict()
        elif isinstance(val, list):
            val = [v.to_dict() if isinstance(v, ASTNode) else v for v in val]

        return {
            "kind": self.kind,
            "value": val,
            "line": self.line,
            "col": self.col,
            "type": self.type,
            "return_type": self.return_type,
            "children": [c.to_dict() for c in self.children],
            "else_children": [c.to_dict() for c in self.else_children],
        }
