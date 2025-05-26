# Quickstart

Get LIMIT up and running in minutes.

---

## Prerequisites

- Python **3.12**
- [PDM](https://pdm.fming.dev/) (Python Development Manager)

---

## 1. Install Python 3.12

Download: [https://www.python.org/downloads/](https://www.python.org/downloads/)

When installing:
- ☑ Add to PATH
- ☑ Enable pip

---

## 2. Install PDM

```bash
pip install -U pdm
````

---

## 3. Clone the Repo

```bash
git clone https://github.com/codingEd-u/limit.git
cd limit
```

(Optional but recommended):

```bash
pdm use -f 3.12
```

---

## 4. Install Dependencies

```bash
pdm install
```

This installs all dependencies defined in `pyproject.toml`.

---

## 5. Run the REPL

```bash
pdm run limit
```

Or explicitly:

```bash
pdm run python -m limit.limit_cli
```

---

## 6. Run a `.limit` File

Save your code as `hello.limit`:

```limit
@ main() {
  = msg "Hello, LIMIT!"
  ! msg
}
CALL main
```

Then run:

```bash
pdm run python -m limit.limit_cli hello.limit
```

---

## 7. Run the Tests (Optional)

```bash
pdm run test
```

---

## 8. View the Docs Locally (Optional)

```bash
pdm run docs
```

Visit `http://127.0.0.1:8000/` to browse the docs.

---

LIMIT is now ready to explore.

```

---
