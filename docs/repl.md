# REPL Usage Guide

The LIMIT REPL lets you write and test LIMIT code interactively with instant feedback. It's ideal for experimentation, debugging, and learning the language incrementally.

---

## Launch the REPL

From your project directory, run:

```bash
pdm run limit
```

or:

```bash
pdm run python -m limit.limit_cli
```

You’ll see:

```
LIMIT > 
```

This is your interactive prompt.

---

## Writing Code in the REPL

You can enter full LIMIT statements or expressions directly.

**Examples:**

```limit
= x 5
! x
```

```limit
@ square(n) {
  RETURN [* n n]
}
! [CALL square 3]
```

---

## Supported Constructs

Anything you can write in a `.limit` file works in the REPL:

* Assignments: `= x 10`
* Function calls: `CALL foo`
* Expressions: `[+ 1 2]`
* Control flow: `? (== x 5) { ! "ok" } else { ! "fail" }`
* Loops, classes, modules, try/catch, etc.

---

## Expression Results

If you enter a bare expression or function call inside brackets, the REPL will **evaluate and print the result**:

```limit
! [CALL sum 1 2]
```

If you enter an expression **without** `!`, it won’t print unless returned from a function.

---

## Errors and Feedback

The REPL provides clear errors for:

* Syntax problems (`SyntaxError`)
* Runtime failures (`ZeroDivisionError`, `AttributeError`, etc.)
* Invalid token or unknown identifier

**Example:**

```limit
= x [/ 1 0]
```

Outputs:

```
RuntimeError: division by zero
```

---

## Loading `.limit` Files

You can run a script file instead of typing manually:

```bash
pdm run python -m limit.limit_cli myscript.limit
```

This lets you build reusable programs and modules.

---

## Advanced Testing

You can test syntax, edge cases, and transpilation interactively:

```limit
CLASS A {
  @ get(self) {
    RETURN 42
  }
}
= a NEW A()
! CALL a.get()
```

---

## Exiting

To exit the REPL, press:

```
Ctrl + C   # interrupt current line
Ctrl + D   # exit REPL completely
```

---

## Tips

* Use `SKIP` to safely define placeholder methods or functions.
* Use `PRINT` (`!`) for debugging or observing expressions.
* You can alias symbols using `SUGAR` inside the REPL too.

---

Explore the [Syntax Cheatsheet](syntax.md) for more constructs.

```

---
