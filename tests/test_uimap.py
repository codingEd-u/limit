import json
import os
import tempfile

import pytest

from limit.limit_constants import CANONICAL_TOKEN_MAP, CANONICAL_TOKENS, token_hashmap
from limit.limit_lexer import Token
from limit.limit_uimap import MappingError, UserInterfaceMapper


def test_dict_mode_basic() -> None:
    uimap = UserInterfaceMapper()
    cfg = {"eq": "ASSIGN", "plus": "PLUS"}
    uimap.configure(cfg)
    assert uimap.token_map["eq"] == "ASSIGN"
    assert uimap.token_map["plus"] == "PLUS"


def test_dict_mode_alias_conflict_raises() -> None:
    uimap = UserInterfaceMapper()
    cfg1 = {"eq": "ASSIGN"}
    cfg2 = {"eq": "PLUS"}  # conflict with previous
    uimap.configure(cfg1)
    with pytest.raises(MappingError) as e:
        uimap.configure(cfg2)
    assert "Alias collision" in str(e.value)


def test_dict_mode_invalid_symbol_raises() -> None:
    uimap = UserInterfaceMapper()
    with pytest.raises(MappingError, match="Unknown symbolic token name"):
        uimap.configure({"foo": "NOT_A_SYMBOL"})


def test_list_mode_basic_flat() -> None:
    uimap = UserInterfaceMapper()
    sample = [["="], ["+"], ["-"]]
    uimap.configure(sample)

    # FIXED: Match symbol based on CANONICAL_TOKENS order, not token_hashmap.values()
    from limit.limit_constants import CANONICAL_TOKENS

    assert uimap.token_map["="] == CANONICAL_TOKENS[0]
    assert uimap.token_map["+"] == CANONICAL_TOKENS[1]
    assert uimap.token_map["-"] == CANONICAL_TOKENS[2]


def test_list_mode_too_many_entries_raises() -> None:
    uimap = UserInterfaceMapper()
    long_list = [["x"]] * (len(token_hashmap) + 1)
    with pytest.raises(MappingError) as e:
        uimap.configure(long_list)
    assert "Too many entries" in str(e.value) or "Alias collision" in str(e.value)


def test_list_mode_nested_aliases() -> None:
    uimap = UserInterfaceMapper()
    nested = [{"=": None}, ["+"], ("/",), {"*": "x"}]
    uimap.configure(nested)
    assert "=" in uimap.token_map
    assert "+" in uimap.token_map
    assert "/" in uimap.token_map
    assert "*" in uimap.token_map


def test_list_mode_conflicting_aliases() -> None:
    uimap = UserInterfaceMapper()
    entries = [["="], ["="]]  # same alias, different symbol
    with pytest.raises(MappingError) as e:
        uimap.configure(entries)
    assert "Alias collision" in str(e.value)


def test_extract_aliases_all_types() -> None:
    uimap = UserInterfaceMapper()
    assert uimap._extract_aliases(None) == []
    assert uimap._extract_aliases("abc") == ["abc"]
    assert uimap._extract_aliases(123) == ["123"]
    assert uimap._extract_aliases(["a", "b"]) == ["a", "b"]
    assert uimap._extract_aliases(("c",)) == ["c"]
    assert uimap._extract_aliases({1: 2}) == ["1"]


def test_get_token_returns_token() -> None:
    uimap = UserInterfaceMapper()
    uimap.configure({"eq": "ASSIGN"})
    tok = uimap.get_token("eq", line=2, col=3)
    assert isinstance(tok, Token)
    assert tok.type == "ASSIGN"
    assert tok.line == 2
    assert tok.col == 3


def test_get_token_returns_none() -> None:
    uimap = UserInterfaceMapper()
    assert uimap.get_token("missing") is None


def test_report_verbose_and_summary_consistency() -> None:
    uimap = UserInterfaceMapper()
    cfg = {"plus": "PLUS", "minus": "SUB"}
    uimap.configure(cfg)
    summary = uimap.summary()
    report = uimap.report()
    assert "plus → PLUS" in report
    assert "minus → SUB" in report
    assert summary == cfg


def test_configure_invalid_type_raises() -> None:
    uimap = UserInterfaceMapper()
    with pytest.raises(MappingError, match="must be either a list or a dict"):
        uimap.configure("not a list or dict")  # type: ignore


def test_extract_aliases_unknown_type() -> None:
    uimap = UserInterfaceMapper()

    class Dummy:
        pass

    assert uimap._extract_aliases(Dummy()) == []


def test_summary_reflects_config() -> None:
    uimap = UserInterfaceMapper()
    uimap.configure({"a": "ASSIGN", "p": "PLUS"})
    assert uimap.summary() == {"a": "ASSIGN", "p": "PLUS"}


def test_get_token_fallback_returns_none() -> None:
    uimap = UserInterfaceMapper()
    assert uimap.get_token("xyz") is None


def test_verbose_report_format() -> None:
    uimap = UserInterfaceMapper()
    uimap.configure({"x": "ASSIGN", "y": "PLUS"})
    report = uimap.report(verbose=True)
    assert "x" in report
    assert "slot" in report
    assert "ASSIGN" in report


def test_load_from_json_success() -> None:
    uimap = UserInterfaceMapper()
    cfg = {"eq , equals": "ASSIGN", "plus , +": "PLUS"}

    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json") as tmp:
        json.dump(cfg, tmp)
        tmp_path = tmp.name

    try:
        uimap.load_from_json(tmp_path)
        assert uimap.token_map["eq"] == "ASSIGN"
        assert uimap.token_map["equals"] == "ASSIGN"
        assert uimap.token_map["plus"] == "PLUS"
        assert uimap.token_map["+"] == "PLUS"
    finally:
        os.remove(tmp_path)


def test_load_from_json_invalid_symbol_raises() -> None:
    uimap = UserInterfaceMapper()
    cfg = {"bad": "NOT_A_SYMBOL"}

    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json") as tmp:
        json.dump(cfg, tmp)
        tmp_path = tmp.name

    try:
        with pytest.raises(MappingError, match="Unknown symbolic token name"):
            uimap.load_from_json(tmp_path)
    finally:
        os.remove(tmp_path)


def test_load_from_json_corrupt_file_raises() -> None:
    uimap = UserInterfaceMapper()
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json") as tmp:
        tmp.write("{{{ this is not json }}}")
        tmp_path = tmp.name

    try:
        with pytest.raises(MappingError, match="Failed to load sugar file"):
            uimap.load_from_json(tmp_path)
    finally:
        os.remove(tmp_path)


def test_session_diff_detects_custom_aliases() -> None:
    uimap = UserInterfaceMapper()

    # Use real canonical aliases from the canonical map
    canonical_cfg = {
        k: v
        for k, v in CANONICAL_TOKEN_MAP.items()
        if isinstance(k, str) and isinstance(v, str)
    }

    # Only load the first two to keep it simple
    subset = dict(list(canonical_cfg.items())[:2])
    uimap.configure(subset)

    assert uimap.session_diff() == {}

    # Add a non-canonical alias for an existing symbol
    uimap.configure({"custom": list(subset.values())[0]})

    # Expect only the custom alias to appear
    diff = uimap.session_diff()
    assert diff == {"custom": list(subset.values())[0]}


def test_list_mode_too_many_entries_hits_limit_branch() -> None:
    uimap = UserInterfaceMapper()
    long_list = [["x"]] * (len(CANONICAL_TOKENS) + 1)
    with pytest.raises(MappingError, match="Too many entries"):
        uimap.configure(long_list)


def test_from_canonical_populates_expected_aliases() -> None:
    uimap = UserInterfaceMapper.from_canonical()

    # Should contain every lowercase string key from CANONICAL_TOKEN_MAP
    expected = {
        key.lower(): value
        for key, value in CANONICAL_TOKEN_MAP.items()
        if isinstance(key, str) and isinstance(value, str)
    }

    assert all(alias in uimap.token_map for alias in expected)
    assert all(uimap.token_map[alias] == expected[alias] for alias in expected)
