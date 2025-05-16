from typing import Any, TypedDict, Union


class ASTDict(TypedDict, total=False):
    kind: str
    value: Any
    line: int
    col: int
    type: str | None
    return_type: str | None
    children: list["ASTDict"]
    else_children: list["ASTDict"]


class ASTNode:
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
