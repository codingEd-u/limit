# LIMIT

**LIMIT** is a minimal, extensible programming language with a clean REPL, strict syntax, and support for multi-target transpilation. Designed for **clarity**, **fast iteration**, and **education**, it provides a lightweight platform for experimenting with core language design, interpreters, and compilation.

---

## 🚀 Features

* ⚙️ **Custom Handwritten Parser** — No parser generators; full control over syntax and error handling.
* ↻ **Transpiles to Python** — With pluggable emitters for future targets like C, JS, or WASM.
* 🧠 **User Interface Mapper** — Bind custom aliases to tokens for building DSLs or visual languages.
* 🧪 **100% Unit Test Discipline** — Over 240 passing tests, including fuzz, branch, and error coverage.
* 💿 **REPL Interface** — Execute LIMIT code interactively with verbose feedback and expression evaluation.
* 📦 **CI/CD Ready** — GitHub Actions, pre-commit hooks, and strict code style enforcement.
* 🧠 **Educationally Valuable** — Great for exploring language construction and compiler internals.

---

## 📦 Quickstart

### 1. Clone & Install

```bash
git clone https://github.com/your-user/limit.git
cd limit
pdm install
```

To launch the REPL:

```bash
pdm run python -m limit.limit_cli
```

---

## 📦 Installation (Clean, Bulletproof)

### 1. Install Python 3.12

Download from: [https://www.python.org/downloads/](https://www.python.org/downloads/)

* ☑ Add to PATH
* ☑ Enable pip

---

### 2. Install PDM (Python Dev Manager)

```bash
pip install -U pdm
```

Docs: [https://pdm.fming.dev](https://pdm.fming.dev)

---

### 3. Clone and Configure

```bash
git clone https://github.com/your-user/limit.git
cd limit
pdm use -f 3.12
```

Ensure this in `pyproject.toml`:

```toml
[tool.pdm.build]
package-dir = "src"
```

---

### 4. Install Project Dependencies

```bash
pdm install
```

---

### 5. (Optional) Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
```

Ensure `.pre-commit-config.yaml` includes:

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.9.1
    hooks:
      - id: black

  - repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: v0.4.4
    hooks:
      - id: ruff
        args: ["--fix"]

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ["--profile=black"]

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: mixed-line-ending
```

---

## 🔧 Usage

### Launch the REPL

```bash
pdm run limit
```

Or:

```bash
pdm run python -m limit.limit_cli
```

---

### 📚 Example: hello.limit

```limit
@ f()
{
  = msg "Hello, LIMIT!"
  PRINT msg
}

CALL f
```

---

### 🧰 Example: Counter

```limit
@ counter()
{
  = x 0
  WHILE [< x 5]
  {
    PRINT x
    = x [+ x 1]
  }
}

CALL counter
```

---

## 🖾 Packaging (Optional)

Build the package:

```bash
pdm build
```

Install system-wide:

```bash
pip install dist/limit-0.1.0-py3-none-any.whl
```

Now you can run:

```bash
limit hello.limit
```
