"""
Provides the `UserInterfaceMapper` class for managing user-defined aliases ("sugar")
in the LIMIT programming language.

This module enables customization of symbolic tokens via alias mappings, allowing
developers to configure more human-readable or domain-specific keywords.

Classes:
    - UserInterfaceMapper: Maps user-defined aliases to canonical token types.
    - MappingError: Raised when configuration or alias conflicts occur.

Features:
    - Maps alias strings to canonical token types using `CANONICAL_TOKENS`
    - Supports both dict-mode (explicit alias-to-token mapping) and list-mode
      (positional mapping against `CANONICAL_TOKENS`)
    - Detects and reports alias conflicts
    - Loads mappings from JSON configuration files
    - Generates alias reports and session diffs
    - Stores optional descriptions for aliases (used by REPL)

Usage:
    >>> mapper = UserInterfaceMapper.from_canonical()
    >>> mapper.configure({"add": "PLUS"})
    >>> tok = mapper.get_token("add")
    >>> print(tok.type)  # "PLUS"

Note:
    This component is primarily used by the REPL for sugar aliasing, but it can also
    be used in batch transpilation workflows or UI frontends.
"""

import json
from typing import Any

from limit.limit_constants import CANONICAL_TOKEN_MAP, CANONICAL_TOKENS
from limit.limit_lexer import Token


class MappingError(Exception):
    """Custom exception for user alias mapping conflicts in LIMIT.

    Raised when a user-defined alias configuration is invalid or
    contains conflicts with existing mappings.

    Attributes:
        conflicts (list[str]): A list of conflicting alias descriptions,
            typically showing alias name and conflicting token types.

    Example:
        raise MappingError("Alias conflict", ["'add' → PLUS vs MULT"])
    """

    def __init__(self, message: str, conflicts: list[str] | None = None):
        super().__init__(message)
        self.conflicts = conflicts or []


class UserInterfaceMapper:
    """Manages user-defined alias-to-token mappings for the LIMIT language.

    This class provides functionality for configuring and maintaining symbolic token
    aliases (also known as "sugar") used during lexing and parsing. It supports both
    dictionary-based and list-based alias configuration modes and provides conflict
    detection, JSON loading, and reporting utilities.

    Attributes:
        token_map (dict[str, str]): Maps user-defined alias strings to canonical token types.
        alias_report (dict[str, str]): A copy of token_map for reporting and summary output.
        descriptions (dict[str, str]): Optional descriptions for aliases (used in REPL reporting).
    """

    def __init__(self) -> None:
        self.token_map: dict[str, str] = {}  # alias → symbolic token
        self.alias_report: dict[str, str] = {}  # same, used for reporting
        self.descriptions: dict[str, str] = {}

    def get_token(self, alias: str, line: int = 0, col: int = 0) -> Token | None:
        """Resolves an alias to a Token if it exists in the current alias map.

        Args:
            alias: The user-defined alias string.
            line: Line number to attach to the token (optional).
            col: Column number to attach to the token (optional).

        Returns:
            A `Token` with the resolved symbolic type if the alias exists, else `None`.
        """
        sym = self.token_map.get(alias)
        return Token(sym, alias, line, col) if sym else None

    def report(self, verbose: bool = False) -> str:
        """Generates a formatted string report of the current alias mappings.

        Args:
            verbose: If True, includes token slot index information for each mapping.

        Returns:
            A newline-separated string summarizing alias → token relationships.
        """
        lines: list[str] = []
        for alias, sym in sorted(self.alias_report.items()):
            if verbose:
                idx = CANONICAL_TOKENS.index(sym)
                lines.append(f"{alias:>12} → {sym:<16} (slot {idx})")
            else:
                lines.append(f"{alias:>12} → {sym}")
        return "\n".join(lines)

    def summary(self) -> dict[str, str]:
        """Returns a copy of the current alias mapping.

        Returns:
            A dictionary mapping alias strings to symbolic token types.
        """
        return dict(self.alias_report)

    def _extract_aliases(self, entry: Any) -> list[str]:
        """Recursively extracts alias strings from a flexible configuration entry.

        Supports strings, numbers, iterables, and dicts for broad compatibility.

        Args:
            entry: An input item from an alias config (e.g., str, list, dict).

        Returns:
            A flat list of alias strings extracted from the entry.
        """
        if entry is None:
            return []
        if isinstance(entry, str):
            return [entry]
        if isinstance(entry, (int, float)):
            return [str(entry)]
        if isinstance(entry, (list, tuple, set)):
            aliases: list[str] = []
            for item in entry:
                aliases.extend(self._extract_aliases(item))
            return aliases
        if isinstance(entry, dict):
            return [str(k) for k in entry.keys()]
        return []

    @classmethod
    def from_canonical(cls) -> "UserInterfaceMapper":
        """
        Constructs a `UserInterfaceMapper` preloaded with the default canonical aliases.

        Uses `CANONICAL_TOKEN_MAP` to build a lowercase mapping of aliases to symbolic tokens.

        Returns:
            A configured `UserInterfaceMapper` instance with canonical mappings applied.
        """
        instance = cls()
        canonical_aliases: dict[str, str] = {
            key.lower(): value
            for key, value in CANONICAL_TOKEN_MAP.items()
            if isinstance(key, str) and isinstance(value, str)
        }
        instance.configure(canonical_aliases)
        return instance

    def load_from_json(self, path: str) -> None:
        """
        Loads alias-to-token mappings from a JSON file and applies them via `configure`.

        The JSON file should contain a dictionary where each key is a string of comma-separated
        aliases and each value is a canonical token type. Keys are automatically split into
        multiple aliases.

        Example JSON structure:
            {
                "plus,add,+": "PLUS",
                "sub,-": "SUB"
            }

        Args:
            path: Path to the JSON file containing alias mappings.

        Raises:
            MappingError: If the file cannot be loaded or the configuration is invalid.
        """
        try:
            with open(path, encoding="utf-8") as f:
                raw_cfg = json.load(f)

            parsed_cfg: dict[tuple[Any, ...], str] = {}
            for key, value in raw_cfg.items():
                aliases = [alias.strip() for alias in key.split(",")]
                parsed_cfg[tuple(aliases)] = value

            self.configure(parsed_cfg)

        except Exception as e:
            raise MappingError(f"Failed to load sugar file: {e}") from e

    def configure(self, cfg: list[Any] | dict[Any, Any]) -> None:
        """
        Applies a new alias-to-token configuration to the mapper.

        Supports two modes:
        - Dict mode: maps alias groups (str, list, tuple, set) to a canonical token.
        - List mode: positional alias groups aligned to `CANONICAL_TOKENS` by index.

        Handles duplicate detection and symbolic token validation.

        Args:
            cfg: A configuration object containing alias definitions.
                - If a dict: maps aliases (or groups of aliases) to symbolic tokens.
                - If a list: maps positional entries to the token at that index in `CANONICAL_TOKENS`.

        Raises:
            MappingError: If any of the following occur:
                - A symbol is not found in `CANONICAL_TOKENS`
                - An alias maps to multiple conflicting symbols
                - The list-mode config exceeds the number of available canonical tokens
        """
        new_token_map: dict[str, str] = {}
        new_alias_report: dict[str, str] = {}
        conflicts: list[str] = []

        valid_symbols = set(CANONICAL_TOKENS)

        if isinstance(cfg, dict):
            for alias_group, sym in cfg.items():
                if sym not in valid_symbols:
                    raise MappingError(f"Unknown symbolic token name: {sym}")
                for alias in self._extract_aliases(alias_group):
                    if (alias in self.token_map and self.token_map[alias] != sym) or (
                        alias in new_token_map and new_token_map[alias] != sym
                    ):
                        conflicts.append(
                            f"'{alias}' → conflict between {self.token_map.get(alias)} and {sym}"
                        )
                    else:
                        new_token_map[alias] = sym
                        new_alias_report[alias] = sym

        elif isinstance(cfg, list):
            if len(cfg) > len(CANONICAL_TOKENS):
                raise MappingError("Too many entries in list-mode config")
            for idx, entry in enumerate(cfg):
                sym = CANONICAL_TOKENS[idx]
                for alias in self._extract_aliases(entry):
                    if (alias in self.token_map and self.token_map[alias] != sym) or (
                        alias in new_token_map and new_token_map[alias] != sym
                    ):
                        conflicts.append(
                            f"'{alias}' → conflict between {self.token_map.get(alias)} and {sym}"
                        )
                    else:
                        new_token_map[alias] = sym
                        new_alias_report[alias] = sym
        else:
            raise MappingError("Configuration must be either a list or a dict")

        if conflicts:
            raise MappingError("Alias collision(s) detected", conflicts)

        self.token_map.update(new_token_map)
        self.alias_report.update(new_alias_report)

    def session_diff(self) -> dict[str, str]:
        # internal method used in REPL v2 for tracking alias delta  # noqa: ERASEDOC
        return {
            alias: token
            for alias, token in self.token_map.items()
            if alias not in CANONICAL_TOKEN_MAP or CANONICAL_TOKEN_MAP[alias] != token
        }
