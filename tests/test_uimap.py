import pytest

from limit.limit_lexer import Token, token_hashmap
from limit.limit_uimap import MappingError, UserInterfaceMapper


def test_dict_mode_basic():
    uimap = UserInterfaceMapper()
    cfg = {"eq": "ASSIGN", "plus": "PLUS"}
    uimap.configure(cfg)
    assert uimap.token_map["eq"] == "ASSIGN"
    assert uimap.token_map["plus"] == "PLUS"


def test_dict_mode_alias_conflict_raises():
    uimap = UserInterfaceMapper()
    cfg1 = {"eq": "ASSIGN"}
    cfg2 = {"eq": "PLUS"}  # conflict with previous
    uimap.configure(cfg1)
    with pytest.raises(MappingError) as e:
        uimap.configure(cfg2)
    assert "Alias collision" in str(e.value)


def test_dict_mode_invalid_symbol_raises():
    uimap = UserInterfaceMapper()
    with pytest.raises(MappingError, match="Unknown symbolic token name"):
        uimap.configure({"foo": "NOT_A_SYMBOL"})


def test_list_mode_basic_flat():
    uimap = UserInterfaceMapper()
    sample = [["="], ["+"], ["-"]]
    uimap.configure(sample)

    # FIXED: Match symbol based on CANONICAL_TOKENS order, not token_hashmap.values()
    from limit.limit_constants import CANONICAL_TOKENS

    assert uimap.token_map["="] == CANONICAL_TOKENS[0]
    assert uimap.token_map["+"] == CANONICAL_TOKENS[1]
    assert uimap.token_map["-"] == CANONICAL_TOKENS[2]


def test_list_mode_too_many_entries_raises():
    uimap = UserInterfaceMapper()
    long_list = [["x"]] * (len(token_hashmap) + 1)
    with pytest.raises(MappingError) as e:
        uimap.configure(long_list)
    assert "Too many entries" in str(e.value) or "Alias collision" in str(e.value)


def test_list_mode_nested_aliases():
    uimap = UserInterfaceMapper()
    nested = [{"=": None}, ["+"], ("/",), {"*": "x"}]
    uimap.configure(nested)
    assert "=" in uimap.token_map
    assert "+" in uimap.token_map
    assert "/" in uimap.token_map
    assert "*" in uimap.token_map


def test_list_mode_conflicting_aliases():
    uimap = UserInterfaceMapper()
    entries = [["="], ["="]]  # same alias, different symbol
    with pytest.raises(MappingError) as e:
        uimap.configure(entries)
    assert "Alias collision" in str(e.value)


def test_extract_aliases_all_types():
    uimap = UserInterfaceMapper()
    assert uimap._extract_aliases(None) == []
    assert uimap._extract_aliases("abc") == ["abc"]
    assert uimap._extract_aliases(123) == ["123"]
    assert uimap._extract_aliases(["a", "b"]) == ["a", "b"]
    assert uimap._extract_aliases(("c",)) == ["c"]
    assert uimap._extract_aliases({1: 2}) == ["1"]


def test_get_token_returns_token():
    uimap = UserInterfaceMapper()
    uimap.configure({"eq": "ASSIGN"})
    tok = uimap.get_token("eq", line=2, col=3)
    assert isinstance(tok, Token)
    assert tok.type == "ASSIGN"
    assert tok.line == 2
    assert tok.col == 3


def test_get_token_returns_none():
    uimap = UserInterfaceMapper()
    assert uimap.get_token("missing") is None


def test_report_verbose_and_summary_consistency():
    uimap = UserInterfaceMapper()
    cfg = {"plus": "PLUS", "minus": "SUB"}
    uimap.configure(cfg)
    summary = uimap.summary()
    report = uimap.report()
    assert "plus → PLUS" in report
    assert "minus → SUB" in report
    assert summary == cfg


def test_configure_invalid_type_raises():
    uimap = UserInterfaceMapper()
    with pytest.raises(MappingError, match="must be either a list or a dict"):
        uimap.configure("not a list or dict")


def test_extract_aliases_unknown_type():
    uimap = UserInterfaceMapper()

    class Dummy:
        pass

    assert uimap._extract_aliases(Dummy()) == []


def test_summary_reflects_config():
    uimap = UserInterfaceMapper()
    uimap.configure({"a": "ASSIGN", "p": "PLUS"})
    assert uimap.summary() == {"a": "ASSIGN", "p": "PLUS"}


def test_get_token_fallback_returns_none():
    uimap = UserInterfaceMapper()
    assert uimap.get_token("xyz") is None


def test_verbose_report_format():
    uimap = UserInterfaceMapper()
    uimap.configure({"x": "ASSIGN", "y": "PLUS"})
    report = uimap.report(verbose=True)
    assert "x" in report
    assert "slot" in report
    assert "ASSIGN" in report
