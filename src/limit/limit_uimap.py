import json
from typing import Any

from limit.limit_constants import CANONICAL_TOKEN_MAP, CANONICAL_TOKENS
from limit.limit_lexer import Token


class MappingError(Exception):
    def __init__(self, message: str, conflicts: list[str] | None = None):
        super().__init__(message)
        self.conflicts = conflicts or []


class UserInterfaceMapper:
    def __init__(self) -> None:
        self.token_map: dict[str, str] = {}  # alias → symbolic token
        self.alias_report: dict[str, str] = {}  # same, used for reporting
        self.descriptions: dict[str, str] = {}

    def get_token(self, alias: str, line: int = 0, col: int = 0) -> Token | None:
        sym = self.token_map.get(alias)
        return Token(sym, alias, line, col) if sym else None

    def report(self, verbose: bool = False) -> str:
        lines: list[str] = []
        for alias, sym in sorted(self.alias_report.items()):
            if verbose:
                idx = CANONICAL_TOKENS.index(sym)
                lines.append(f"{alias:>12} → {sym:<16} (slot {idx})")
            else:
                lines.append(f"{alias:>12} → {sym}")
        return "\n".join(lines)

    def summary(self) -> dict[str, str]:
        return dict(self.alias_report)

    def _extract_aliases(self, entry: Any) -> list[str]:
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
        instance = cls()
        canonical_aliases: dict[str, str] = {
            key.lower(): value
            for key, value in CANONICAL_TOKEN_MAP.items()
            if isinstance(key, str) and isinstance(value, str)
        }
        instance.configure(canonical_aliases)
        return instance

    def load_from_json(self, path: str) -> None:
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
        return {
            alias: token
            for alias, token in self.token_map.items()
            if alias not in CANONICAL_TOKEN_MAP or CANONICAL_TOKEN_MAP[alias] != token
        }
