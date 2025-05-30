"""
Defines the canonical token vocabulary and token mappings for the LIMIT language.

Contents:
    1. CANONICAL_TOKENS:
        - Ordered list of all valid tokens in LIMIT.
        - Includes operators, keywords, literals, delimiters, and primitives.
        - Used by the lexer, parser, emitter, and UI alias mapper.

    2. CANONICAL_TOKEN_MAP:
        - Maps human-readable symbolic names (e.g., 'ADDITION', 'WHILE_LOOP') to canonical token strings.
        - Used for alias resolution, introspection, and external sugar configuration.

    3. operator_tokens:
        - Indexed operator map used by the parser to normalize arithmetic, logical, and comparison tokens.

    4. token_hashmap:
        - Maps raw input strings (like '+', '==', 'RETURN') to canonical tokens.
        - Used by the lexer for longest-match recognition and token normalization.

    5. CONTROL_SYMBOLS:
        - Subset of CANONICAL_TOKENS representing all valid control structures.
        - Used by UI mappers and token validators.

    6. Token Categories and Constants:
        - Includes EOF, IDENT, ERROR, NUMBER, STRING, LITERAL.
        - Also defines character groups (e.g., WHITESPACE_CHARS, QUOTE_CHARS) and delimiter tokens.

Usage:
    These definitions are the single source of truth for all token-based logic in LIMIT.
    Any addition to language syntax must first be reflected in this module to ensure consistency.

Used by:
    - Lexer (`token_hashmap`, `CANONICAL_TOKENS`)
    - Parser (`operator_tokens`, `CONTROL_SYMBOLS`)
    - REPL alias engine (`CANONICAL_TOKEN_MAP`)
    - Transpilers and emitters (via token normalization)

"""

CANONICAL_TOKENS = [
    # Arithmetic and logic ops
    "PLUS",
    "SUB",
    "MULT",
    "DIV",
    "MOD",
    "EQ",
    "NE",
    "LT",
    "LE",
    "GT",
    "GE",
    "AND",
    "OR",
    "NOT",
    "TRUTHY",
    # Expression constructs
    "CALL",
    "PROP",
    "ASSIGN",
    # Control structures
    "IF",
    "ELSE",
    "FUNC",
    "PRINT",
    "RETURN",
    "SKIP",
    "LOOP_WHILE",
    "LOOP_FOR",
    "DELIM_AT",
    "DELIM_BY",
    "DELIM_TO",
    "BREAK",
    "CONTINUE",
    # Class / object
    "CLASS",
    "EXTENDS",
    "NEW",
    "THIS",
    # Modules and exceptions
    "TRY",
    "CATCH",
    "FINALLY",
    "EXPORT",
    "IMPORT",
    "MODULE",
    # IO and literals
    "INPUT",
    "DELIM_FROM",
    "TRUE",
    "FALSE",
    "NULL",
    # Brackets / delimiters
    "LBRACK",
    "RBRACK",
    "LBRACE",
    "RBRACE",
    "LPAREN",
    "RPAREN",
    "COMMA",
    "COLON",
    "DOT",
    # Primitive token types
    "EOF",
    "IDENT",
    "ERROR",
    "NUMBER",
    "FLOAT",
    "STRING",
    "LITERAL",
    " ",  # SPACE_CHAR
    "\t",  # TAB_CHAR
    "\r",  # CARRIAGE_CHAR
    "\n",  # NEWLINE_CHAR
    "_",  # IDENT_PREFIX
    ".",  # NUMBER_DELIMITER
    "#",  # COMMENT_PREFIX
    '"',  # QUOTE_CHARS[0]
    "'",  # QUOTE_CHARS[1]
]

CANONICAL_TOKEN_MAP = {
    "ADDITION": CANONICAL_TOKENS[0],
    "SUBTRACTION": CANONICAL_TOKENS[1],
    "MULTIPLICATION": CANONICAL_TOKENS[2],
    "DIVISION": CANONICAL_TOKENS[3],
    "MODULUS": CANONICAL_TOKENS[4],
    "EQUALS": CANONICAL_TOKENS[5],
    "NOT_EQUALS": CANONICAL_TOKENS[6],
    "LESS_THAN": CANONICAL_TOKENS[7],
    "LESS_THAN_EQUAL": CANONICAL_TOKENS[8],
    "GREATER_THAN": CANONICAL_TOKENS[9],
    "GREATER_THAN_EQUAL": CANONICAL_TOKENS[10],
    "LOGICAL_AND": CANONICAL_TOKENS[11],
    "LOGICAL_OR": CANONICAL_TOKENS[12],
    "LOGICAL_NOT": CANONICAL_TOKENS[13],
    "LOGICAL_TRUTHY": CANONICAL_TOKENS[14],  # ✅ NEW
    "FUNCTION_CALL": CANONICAL_TOKENS[15],
    "PROPAGATE_RESULT": CANONICAL_TOKENS[16],
    "ASSIGNMENT": CANONICAL_TOKENS[17],
    "IF_STATEMENT": CANONICAL_TOKENS[18],
    "ELSE_CLAUSE": CANONICAL_TOKENS[19],
    "FUNCTION_DEFINITION": CANONICAL_TOKENS[20],
    "PRINT_STATEMENT": CANONICAL_TOKENS[21],
    "RETURN_STATEMENT": CANONICAL_TOKENS[22],
    "SKIP_BLOCK": CANONICAL_TOKENS[23],
    "WHILE_LOOP": CANONICAL_TOKENS[24],
    "FOR_LOOP": CANONICAL_TOKENS[25],
    "FOR_RANGE_START": CANONICAL_TOKENS[26],
    "FOR_RANGE_STEP": CANONICAL_TOKENS[27],
    "FOR_RANGE_END": CANONICAL_TOKENS[28],
    "BREAK_LOOP": CANONICAL_TOKENS[29],
    "CONTINUE_LOOP": CANONICAL_TOKENS[30],
    "CLASS_DEFINITION": CANONICAL_TOKENS[31],
    "INHERITANCE": CANONICAL_TOKENS[32],
    "OBJECT_CREATION": CANONICAL_TOKENS[33],
    "THIS_REFERENCE": CANONICAL_TOKENS[34],
    "TRY_BLOCK": CANONICAL_TOKENS[35],
    "CATCH_BLOCK": CANONICAL_TOKENS[36],
    "FINALLY_BLOCK": CANONICAL_TOKENS[37],
    "EXPORT_SYMBOL": CANONICAL_TOKENS[38],
    "IMPORT_MODULE": CANONICAL_TOKENS[39],
    "MODULE_DECLARATION": CANONICAL_TOKENS[40],
    "INPUT_STATEMENT": CANONICAL_TOKENS[41],
    "INPUT_FROM_FILE": CANONICAL_TOKENS[42],
    "BOOLEAN_TRUE": CANONICAL_TOKENS[43],
    "BOOLEAN_FALSE": CANONICAL_TOKENS[44],
    "NULL_LITERAL": CANONICAL_TOKENS[45],
    "OPEN_BRACKET": CANONICAL_TOKENS[46],
    "CLOSE_BRACKET": CANONICAL_TOKENS[47],
    "OPEN_BRACE": CANONICAL_TOKENS[48],
    "CLOSE_BRACE": CANONICAL_TOKENS[49],
    "OPEN_PAREN": CANONICAL_TOKENS[50],
    "CLOSE_PAREN": CANONICAL_TOKENS[51],
    "COMMA_SEPARATOR": CANONICAL_TOKENS[52],
    "COLON_SEPARATOR": CANONICAL_TOKENS[53],
    "DOT_ACCESSOR": CANONICAL_TOKENS[54],
    "END_OF_FILE": CANONICAL_TOKENS[55],
    "IDENTIFIER": CANONICAL_TOKENS[56],
    "LEX_ERROR": CANONICAL_TOKENS[57],
    "INTEGER_LITERAL": CANONICAL_TOKENS[58],
    "FLOAT_LITERAL": CANONICAL_TOKENS[59],
    "STRING_LITERAL": CANONICAL_TOKENS[60],
    "GENERIC_LITERAL": CANONICAL_TOKENS[61],
    "SPACE_CHAR": CANONICAL_TOKENS[62],
    "TAB_CHAR": CANONICAL_TOKENS[63],
    "CARRIAGE_RETURN": CANONICAL_TOKENS[64],
    "NEWLINE_CHAR": CANONICAL_TOKENS[65],
    "IDENTIFIER_PREFIX": CANONICAL_TOKENS[66],
    "DECIMAL_POINT": CANONICAL_TOKENS[67],
    "COMMENT_PREFIX": CANONICAL_TOKENS[68],
    "DOUBLE_QUOTE": CANONICAL_TOKENS[69],
    "SINGLE_QUOTE": CANONICAL_TOKENS[70],
}


# Operator index mapping
operator_tokens = {
    0: CANONICAL_TOKEN_MAP["ADDITION"],
    1: CANONICAL_TOKEN_MAP["SUBTRACTION"],
    2: CANONICAL_TOKEN_MAP["MULTIPLICATION"],
    3: CANONICAL_TOKEN_MAP["DIVISION"],
    4: CANONICAL_TOKEN_MAP["MODULUS"],
    5: CANONICAL_TOKEN_MAP["LOGICAL_AND"],
    6: CANONICAL_TOKEN_MAP["LOGICAL_OR"],
    7: CANONICAL_TOKEN_MAP["LOGICAL_NOT"],
    8: CANONICAL_TOKEN_MAP["LOGICAL_TRUTHY"],  # ✅ NEW
    9: CANONICAL_TOKEN_MAP["LESS_THAN"],
    10: CANONICAL_TOKEN_MAP["LESS_THAN_EQUAL"],
    11: CANONICAL_TOKEN_MAP["GREATER_THAN"],
    12: CANONICAL_TOKEN_MAP["GREATER_THAN_EQUAL"],
    13: CANONICAL_TOKEN_MAP["EQUALS"],
    14: CANONICAL_TOKEN_MAP["NOT_EQUALS"],
    15: CANONICAL_TOKEN_MAP["FUNCTION_CALL"],  # shifted
    16: CANONICAL_TOKEN_MAP["PROPAGATE_RESULT"],
}


# Token type mapping
token_hashmap = {
    "=": CANONICAL_TOKEN_MAP["ASSIGNMENT"],
    "+": operator_tokens[0],
    "-": operator_tokens[1],
    "*": operator_tokens[2],
    "/": operator_tokens[3],
    "%": operator_tokens[4],
    "==": operator_tokens[13],
    "!=": operator_tokens[14],
    "<": operator_tokens[9],
    "<=": operator_tokens[10],
    ">": operator_tokens[11],
    ">=": operator_tokens[12],
    "EQ": operator_tokens[13],
    "NE": operator_tokens[14],
    "LT": operator_tokens[9],
    "LE": operator_tokens[10],
    "GT": operator_tokens[11],
    "GE": operator_tokens[12],
    "AND": operator_tokens[5],
    "OR": operator_tokens[6],
    "NOT": operator_tokens[7],
    "TRUTHY": operator_tokens[8],  # ✅ NEW
    "CALL": operator_tokens[15],
    "$": operator_tokens[16],
    "?": CANONICAL_TOKEN_MAP["IF_STATEMENT"],
    "@": CANONICAL_TOKEN_MAP["FUNCTION_DEFINITION"],
    "!": CANONICAL_TOKEN_MAP["PRINT_STATEMENT"],
    "[": CANONICAL_TOKEN_MAP["OPEN_BRACKET"],
    "]": CANONICAL_TOKEN_MAP["CLOSE_BRACKET"],
    "{": CANONICAL_TOKEN_MAP["OPEN_BRACE"],
    "}": CANONICAL_TOKEN_MAP["CLOSE_BRACE"],
    "(": CANONICAL_TOKEN_MAP["OPEN_PAREN"],
    ")": CANONICAL_TOKEN_MAP["CLOSE_PAREN"],
    ",": CANONICAL_TOKEN_MAP["COMMA_SEPARATOR"],
    ":": CANONICAL_TOKEN_MAP["COLON_SEPARATOR"],
    ".": CANONICAL_TOKEN_MAP["DOT_ACCESSOR"],
    "TRUE": CANONICAL_TOKEN_MAP["GENERIC_LITERAL"],
    "FALSE": CANONICAL_TOKEN_MAP["GENERIC_LITERAL"],
    "NULL": CANONICAL_TOKEN_MAP["GENERIC_LITERAL"],
    "INPUT": CANONICAL_TOKEN_MAP["INPUT_STATEMENT"],
    "FROM": CANONICAL_TOKEN_MAP["INPUT_FROM_FILE"],
    "ELSE": CANONICAL_TOKEN_MAP["ELSE_CLAUSE"],
    "TO": CANONICAL_TOKEN_MAP["FOR_RANGE_END"],
    "AT": CANONICAL_TOKEN_MAP["FOR_RANGE_START"],
    "BY": CANONICAL_TOKEN_MAP["FOR_RANGE_STEP"],
    "RETURN": CANONICAL_TOKEN_MAP["RETURN_STATEMENT"],
    "WHILE": CANONICAL_TOKEN_MAP["WHILE_LOOP"],
    "FOR": CANONICAL_TOKEN_MAP["FOR_LOOP"],
    "SKIP": CANONICAL_TOKEN_MAP["SKIP_BLOCK"],
    "CLASS": CANONICAL_TOKEN_MAP["CLASS_DEFINITION"],
    "NEW": CANONICAL_TOKEN_MAP["OBJECT_CREATION"],
    "THIS": CANONICAL_TOKEN_MAP["THIS_REFERENCE"],
    "EXTENDS": CANONICAL_TOKEN_MAP["INHERITANCE"],
    "TRY": CANONICAL_TOKEN_MAP["TRY_BLOCK"],
    "CATCH": CANONICAL_TOKEN_MAP["CATCH_BLOCK"],
    "FINALLY": CANONICAL_TOKEN_MAP["FINALLY_BLOCK"],
    "EXPORT": CANONICAL_TOKEN_MAP["EXPORT_SYMBOL"],
    "IMPORT": CANONICAL_TOKEN_MAP["IMPORT_MODULE"],
    "MODULE": CANONICAL_TOKEN_MAP["MODULE_DECLARATION"],
    "BREAK": CANONICAL_TOKEN_MAP["BREAK_LOOP"],
    "CONTINUE": CANONICAL_TOKEN_MAP["CONTINUE_LOOP"],
}

CONTROL_SYMBOLS = [
    "ASSIGN",
    "PLUS",
    "SUB",
    "MULT",
    "DIV",
    "MOD",
    "EQ",
    "NE",
    "LT",
    "LE",
    "GT",
    "GE",
    "AND",
    "OR",
    "NOT",
    "CALL",
    "PROP",
    "INPUT",
    "DELIM_FROM",
    "IF",
    "ELSE",
    "FUNC",
    "PRINT",
    "RETURN",
    "SKIP",
    "LOOP_WHILE",
    "LOOP_FOR",
    "TO",
    "AT",
    "BY",
    "BREAK",
    "CONTINUE",
    "CLASS",
    "EXTENDS",
    "NEW",
    "THIS",
    "TRY",
    "CATCH",
    "FINALLY",
    "EXPORT",
    "IMPORT",
    "MODULE",
    "TRUE",
    "FALSE",
    "NULL",
    "LBRACK",
    "RBRACK",
    "LBRACE",
    "RBRACE",
    "LPAREN",
    "RPAREN",
    "COMMA",
    "COLON",
    "DOT",
]


# Constants for non-symbolic types
TOKEN_EOF = "EOF"
TOKEN_IDENT = "IDENT"
TOKEN_ERROR = "ERROR"
TOKEN_NUMBER = "NUMBER"
TOKEN_FLOAT = "FLOAT"
TOKEN_STRING = "STRING"
TOKEN_LITERAL = "LITERAL"

## SINGLE CHARS
# Character categories
SPACE_CHAR = " "
TAB_CHAR = "\t"
CARRIAGE_CHAR = "\r"
NEWLINE_CHAR = "\n"
IDENT_PREFIX = "_"
NUMBER_DELIMITER = "."
COMMENT_PREFIX = "#"  # If you ever want to make comments start with another symbol
MAX_OPERATOR_LENGTH = 64


# Groupings
WHITESPACE_CHARS = (SPACE_CHAR, TAB_CHAR, CARRIAGE_CHAR, NEWLINE_CHAR)
QUOTE_CHARS = ('"', "'")
