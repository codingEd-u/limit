# Contributing to LIMIT

First off, thank you for your interest in contributing to **LIMIT**, a minimalist and expressive DSL!

LIMIT is developed with transparency and precision, heavily accelerated using AI tools (e.g., ChatGPT). Contributors are encouraged to leverage such tools responsibly while retaining critical oversight.

---

## Project Structure

```
limit/
├── .github/                  # CI/CD workflows
│   └── workflows/
│       └── ci.yml
├── src/
│   └── limit/                 # Core LIMIT source code
│       ├── __init__.py
│       ├── emitters/         # Output emitters (e.g., Python)
│       │   ├── __init__.py
│       │   └── py_emitter.py
│       ├── limit_ast.py       # AST node system
│       ├── limit_cli.py       # CLI interface
│       ├── limit_lexer.py     # Tokenizer
│       ├── limit_parser.py    # Syntax tree builder
│       ├── limit_repl.py      # Interactive REPL
│       ├── limit_transpile.py # Transpilation backend
│       └── limit_uimap.py     # User alias/token mapper
├── tests/                    # Unit and property tests
├── pyproject.toml            # Tool and build config
├── pdm.lock                  # Lock file
├── .pre-commit-config.yaml   # Pre-commit rules
├── .gitignore
├── LICENSE
├── README.md
└── print_tree.py             # Project structure printer
```

---

## Development Setup

1. Install [PDM](https://pdm.fming.dev) and dependencies:

```bash
pdm install
```

2. Launch REPL:

```bash
pdm run python -m limit.limit_repl
```

3. Run tests:

```bash
pdm run pytest --cache-clear --cov=src/limit --cov-report=html
```

---

## Testing Guidelines

We use:

* `pytest` for unit testing
* `hypothesis` for property-based fuzzing
* `pytest-benchmark` for performance

### Full test suite with benchmark support:

```bash
pdm run pytest --benchmark-enable
```

### Specific test module:

```bash
pdm run pytest tests/test_parser.py
```

### Coverage target:

We aim for >90% total coverage, with **100%** on small/easy modules where practical.

---

## Code Style and Tooling

All code must pass:

* [`black`](https://github.com/psf/black)
* [`isort`](https://pycqa.github.io/isort/)
* [`mypy`](https://mypy-lang.org/)
* [`ruff`](https://github.com/astral-sh/ruff)

Run manually:

```bash
pdm run black .
pdm run isort .
pdm run ruff check .
pdm run mypy src/
```

---

## Pre-Commit Hooks

We use [pre-commit](https://pre-commit.com/) to enforce standards automatically.

### Install hooks:

```bash
pdm run pre-commit install
```

### Run all hooks manually:

```bash
pdm run pre-commit run --all-files
```

### Auto-fix issues:

```bash
pdm run pre-commit run --all-files --hook-stage manual
```

Pre-commit checks include:

* Black formatting
* Import sorting (isort)
* Ruff linting
* Mypy type enforcement
* Line endings, trailing whitespace, and YAML sanity checks

---

## Creating a Feature

1. Create a feature branch:

```bash
git checkout -b feat/your-feature-name
```

2. Add your code and matching tests
3. Format, type-check, and lint
4. Push and open a pull request to `main`

---

Here’s the updated version of your `CONTRIBUTING.md` with a new section titled **"Local CI Testing with `act`"** added at the end, just before the license section:

---


## ✅ Local CI Testing with `act`

We use [`act`](https://github.com/nektos/act) to run GitHub Actions locally, allowing fast feedback on CI pipelines without pushing to GitHub.

### 💡 Why Use `act`?

Running `act` locally helps you:

* Validate GitHub Actions workflows before pushing
* Detect pre-commit and coverage errors early
* Simulate tagged releases and `workflow_dispatch` inputs
* Shorten iteration cycles dramatically

---

### 🚀 How to Run the Release Workflow Locally

1. **Install `act`:**
   [https://github.com/nektos/act#installation](https://github.com/nektos/act#installation)

2. **Create a `.secrets` file in your project root:**

```env
GITHUB_TOKEN=ghp_yourRealPersonalAccessToken
```

> Your token must have `repo`, `workflow`, and `write:packages` scopes if you use private actions.

3. **Run the release workflow:**

```bash
act workflow_dispatch --input version=0.1.0-alpha -W .github/workflows/release.yml -P ubuntu-latest=ghcr.io/catthehacker/ubuntu:act-latest --env ACT=true
```

This:

* Injects `version=0.1.0-alpha`
* Uses your `.secrets` token for GitHub access
* Mocks `ACT=true` to skip real git push steps
* Tests the full `release.yml` workflow safely inside Docker

---

### 🧪 Other Examples

* **Default test suite:**

```bash
act -j test
```

* **Force clean run:**

```bash
act -j test --rm --container-architecture linux/amd64
```

* **Simulate a push event:**

```bash
act push
```

* **Debug volume paths:**

```bash
act -j test --bind
```

---

> ⚠️ `coverage.py` may raise teardown errors inside Docker containers. This project safely monkeypatches `coverage.collector` to avoid false crashes. Coverage will still export correctly.

---

## License

All contributions are accepted under the terms of the [MIT License](./LICENSE).

---

## Need Help?

Open an issue or start a discussion if you:

* Encounter a bug
* Want to propose a new syntax or emitter
* Need help with the codebase

Thanks again for contributing to **LIMIT**!
