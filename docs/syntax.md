# Syntax Cheatsheet

This is the **canonical token mapping** for the LIMIT programming language, with examples after every group. Each token maps to a normalized internal parser type.

---

## Control Structures

| Symbol | Token |
|--------|--------|
| `?` | `IF` |
| `else` | `ELSE` |
| `=` | `ASSIGN` |
| `!` | `PRINT` |
| `WHILE` | `LOOP_WHILE` |
| `FOR` | `LOOP_FOR` |
| `BREAK` | `BREAK` |
| `CONTINUE` | `CONTINUE` |

**Examples:**

```limit
? (> x 10) { ! "Too big" } else { ! "OK" }

= count 0

WHILE [< count 3] {
  ! count
  = count [+ count 1]
}

FOR i TO 3 {
  ! i
}

BREAK
CONTINUE
```

---

## Function & Invocation

| Symbol   | Token    |
| -------- | -------- |
| `@`      | `FUNC`   |
| `CALL`   | `CALL`   |
| `RETURN` | `RETURN` |
| `SKIP`   | `SKIP`   |

**Examples:**

```limit
@ add(x, y) {
  RETURN [+ x y]
}

! [CALL add 1 2]

@ doNothing() {
  SKIP
}
```

---

## Input & Output

| Symbol  | Token        |
| ------- | ------------ |
| `INPUT` | `INPUT`      |
| `FROM`  | `DELIM_FROM` |

**Examples:**

```limit
INPUT name
INPUT age: int
INPUT score: float
INPUT FROM "data.txt"
```

---

## Error Handling

| Symbol    | Token     |
| --------- | --------- |
| `TRY`     | `TRY`     |
| `CATCH`   | `CATCH`   |
| `FINALLY` | `FINALLY` |

**Examples:**

```limit
TRY {
  = x [/ 1 0]
} CATCH {
  ! "Error"
} FINALLY {
  ! "Cleanup"
}
```

---

## Classes & Objects

| Symbol    | Token     |
| --------- | --------- |
| `CLASS`   | `CLASS`   |
| `EXTENDS` | `EXTENDS` |
| `NEW`     | `NEW`     |
| `THIS`    | `THIS`    |

**Examples:**

```limit
CLASS Point {
  @ init(self, x, y) {
    = self.x x
    = self.y y
  }
}

= p NEW Point(1, 2)
! p.x
```

---

## Modules

| Symbol   | Token    |
| -------- | -------- |
| `MODULE` | `MODULE` |
| `IMPORT` | `IMPORT` |
| `EXPORT` | `EXPORT` |

**Examples:**

```limit
MODULE mymod
IMPORT "file.limit"

@ foo() {
  RETURN 1
}
EXPORT foo
```

---

## Propagation

| Symbol | Token  |
| ------ | ------ |
| `$`    | `PROP` |

**Examples:**

```limit
@ maybe(x) {
  $ x
  RETURN 0
}
CALL maybe(42)
```

---

## Literals

| Symbol  | Token     |
| ------- | --------- |
| `TRUE`  | `LITERAL` |
| `FALSE` | `LITERAL` |
| `NULL`  | `LITERAL` |

**Examples:**

```limit
= flag TRUE
= nope FALSE
= x NULL
```

---

## Prefix Operators

| Symbol(s)                        | Token  |
| -------------------------------- | ------ |
| `+`                              | `PLUS` |
| `-`                              | `SUB`  |
| `*`                              | `MULT` |
| `/`                              | `DIV`  |
| `%`                              | `MOD`  |
| `==`, `EQ`                       | `EQ`   |
| `!=`, `NE`                       | `NE`   |
| `<`                              | `LT`   |
| `>`                              | `GT`   |
| `<=`                             | `LE`   |
| `>=`                             | `GE`   |
| `AND`                            | `AND`  |
| `OR`                             | `OR`   |
| `NOT`                            | `NOT`  |

**Examples:**

```limit
[+ 1 2]
[* 3 4]
[== a b]
[AND TRUE FALSE]
[NOT TRUE]
```

---

## Symbols & Delimiters

| Symbol  | Token              |
| ------- | ------------------ |
| `(` `)` | `LPAREN`, `RPAREN` |
| `[` `]` | `LBRACK`, `RBRACK` |
| `{` `}` | `LBRACE`, `RBRACE` |
| `.`     | `DOT`              |
| `:`     | `COLON`            |
| `,`     | `COMMA`            |

**Examples:**

```limit
! [CALL sum(1, 2)]
= obj.field
@ f(x: int) { ! x }
```

---

## FOR Loop Enhancements

| Symbol | Token      |
| ------ | ---------- |
| `TO`   | `DELIM_TO` |
| `AT`   | `DELIM_AT` |
| `BY`   | `DELIM_BY` |

**Examples:**

```limit
FOR i TO 3 {
  ! i
}

FOR j AT 1 TO 5 {
  ! j
}

FOR k AT 10 TO 0 BY -2 {
  ! k
}
```

---

## Types & Identifiers

| Value            | Token                       |
| ---------------- | --------------------------- |
| `"text"`         | `STRING`                    |
| `42`             | `NUMBER`                    |
| `3.14`           | `FLOAT`                     |
| `my_var`, `café` | `IDENT` (Unicode supported) |

**Examples:**

```limit
= name "Alice"
= age 42
= pi 3.14
= café "crème brûlée"
```

---

## Sugar Mapping (Aliases)

LIMIT allows alternate names for tokens using `SUGAR` blocks.

**Examples:**

```limit
SUGAR {
  ["add", "sum", "plus"]: "PLUS"
}
SUGAR {
  ["increase", "inc"]: "PLUS"
}
```

Use these to customize the language surface for a domain-specific use case or natural-language style.

---

Want to try these live? Jump to the [REPL Usage Guide](repl.md).

```

---
