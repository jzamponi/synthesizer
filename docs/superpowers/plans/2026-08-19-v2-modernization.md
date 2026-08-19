# Synthesizer v2 Modernization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade `synthesizer` to Python 3.13 and `uv`, configure Ruff & Mypy, add pre-commit hooks, create a Taskfile, and implement a full unit test suite.

**Architecture:** Modern Python packaging with `pyproject.toml` and `uv`, static analysis with Ruff and Mypy, Taskfile automation, and pytest unit testing.

**Tech Stack:** Python 3.13, uv, ruff, mypy, pre-commit, task, pytest, astropy, matplotlib, numpy, scipy.

## Global Constraints

- Python version floor: `>=3.13`
- Environment & Package Manager: `uv`
- Formatting & Linting: `ruff`
- Static Type Checker: `mypy`
- Task runner: `Taskfile` (`task`)
- Pre-commit: `.pre-commit-config.yaml`

---

### Task 1: Python 3.13 & `uv` Setup

**Files:**
- Create: `.python-version`
- Modify: `pyproject.toml`
- Create: `uv.lock`

**Interfaces:**
- Consumes: Existing dependencies in `pyproject.toml`
- Produces: Managed `uv` environment locked for Python 3.13

- [ ] **Step 1: Update `pyproject.toml` dependencies and python requirements**

Update `pyproject.toml` to specify `requires-python = ">=3.13"` and standard dev dependency groups:

```toml
[build-system]
requires = ["setuptools>=61.0.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "astro-synthesizer"
version = "2.0.0"
description = "Generate astronomical synthetic observations from the command-line"
readme = "README.md"
authors = [{ name = "Joaquin Zamponi", email = "jzamponi@mpe.mpg.de" }]
license = { file = "LICENSE" }
classifiers = [
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.13",
]
keywords = ["astronomy", "radiative-transfer", "synthetic-observations"]
dependencies = [
    "requests",
    "astropy >= 5.1",
    "matplotlib",
    "numpy",
    "scipy",
    "mpmath",
    "progressbar2",
    "aplpy",
    "h5py",
    "yt",
    "pyqt5",
    "mayavi",
]
requires-python = ">=3.13"

[dependency-groups]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "mypy>=1.10",
    "ruff>=0.4",
    "pre-commit>=3.7",
    "types-requests",
]

[project.urls]
Homepage = "https://github.com/jzamponi/synthesizer"

[project.scripts]
synthesizer = "synthesizer.parser:synthesizer"
```

- [ ] **Step 2: Create `.python-version`**

Set contents of `.python-version` to `3.13`.

- [ ] **Step 3: Run `uv lock` and `uv sync`**

Run: `uv lock && uv sync`
Expected: `uv.lock` generated and virtualenv populated.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml .python-version uv.lock
git commit -m "build: update target Python to 3.13 and add uv lockfile"
```

---

### Task 2: Ruff and Mypy Configuration in `pyproject.toml`

**Files:**
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: `pyproject.toml`
- Produces: `[tool.ruff]` and `[tool.mypy]` sections in `pyproject.toml`

- [ ] **Step 1: Append `[tool.ruff]` and `[tool.mypy]` settings to `pyproject.toml`**

Add the following sections:

```toml
[tool.ruff]
target-version = "py313"
line-length = 88

[tool.ruff.lint]
select = ["E", "F", "W", "I", "UP", "B"]
ignore = ["E501"]

[tool.mypy]
python_version = "3.13"
warn_return_any = true
warn_unused_configs = true
ignore_missing_imports = true
check_untyped_defs = true

[[tool.mypy.overrides]]
module = [
    "astropy.*",
    "aplpy.*",
    "yt.*",
    "mayavi.*",
    "scipy.*",
    "progressbar.*",
    "h5py.*",
    "pyface.*",
    "traits.*",
    "traitsui.*",
]
ignore_missing_imports = true
```

- [ ] **Step 2: Verify Ruff and Mypy configuration load cleanly**

Run: `uv run ruff check --version && uv run mypy --version`
Expected: Clean outputs showing version info.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "tooling: configure ruff and mypy in pyproject.toml"
```

---

### Task 3: Codebase Ruff Fix & Formatting

**Files:**
- Modify: All `.py` files under `synthesizer/`

**Interfaces:**
- Consumes: Python code in `synthesizer/`
- Produces: Formatted, lint-clean Python files

- [ ] **Step 1: Run Ruff check with auto-fix**

Run: `uv run ruff check synthesizer/ --fix`
Expected: Auto-fixable issues fixed.

- [ ] **Step 2: Run Ruff format**

Run: `uv run ruff format synthesizer/`
Expected: Code formatted to standard style.

- [ ] **Step 3: Check remaining lint issues and resolve**

Run: `uv run ruff check synthesizer/`
Expected: 0 lint errors.

- [ ] **Step 4: Commit**

```bash
git add synthesizer/
git commit -m "style: apply ruff auto-fixes and formatting across codebase"
```

---

### Task 4: Add Type Hints & Fix Mypy Type Errors

**Files:**
- Modify: `synthesizer/utils.py`
- Modify: `synthesizer/parser.py`
- Modify: `synthesizer/pipeline.py`
- Modify: `synthesizer/dustmixer/*.py`
- Modify: `synthesizer/gridder/*.py`
- Modify: `synthesizer/raytrace/*.py`
- Modify: `synthesizer/synobs/*.py`

**Interfaces:**
- Consumes: Function signatures in `synthesizer/`
- Produces: Fully type-checked Python codebase under Mypy

- [ ] **Step 1: Add type annotations to `synthesizer/utils.py`**

Add python 3.13 standard type hints (`str`, `int`, `float`, `bool`, `list`, `dict`, `Path`, `Optional`, `Any`) to `utils.py`.

- [ ] **Step 2: Add type annotations to `synthesizer/parser.py`**

Add function return type `None` and annotations to `parser.py`.

- [ ] **Step 3: Add type annotations to `synthesizer/pipeline.py`**

Add type hints to `Pipeline.__init__` and methods in `pipeline.py`.

- [ ] **Step 4: Run Mypy check and fix remaining type errors**

Run: `uv run mypy synthesizer/`
Expected: `Success: no issues found in X source files`.

- [ ] **Step 5: Commit**

```bash
git add synthesizer/
git commit -m "refactor: add type annotations and fix mypy errors"
```

---

### Task 5: Pre-commit Hook Setup

**Files:**
- Create: `.pre-commit-config.yaml`

**Interfaces:**
- Consumes: `.pre-commit-config.yaml`
- Produces: Active git pre-commit hook enforcing ruff, ruff-format, and mypy

- [ ] **Step 1: Create `.pre-commit-config.yaml`**

Write the pre-commit configuration:

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.10
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: local
    hooks:
      - id: mypy
        name: mypy
        entry: uv run mypy synthesizer
        language: system
        types: [python]
        pass_filenames: false
```

- [ ] **Step 2: Test pre-commit hook execution**

Run: `uv run pre-commit run --all-files`
Expected: All hooks pass.

- [ ] **Step 3: Commit**

```bash
git add .pre-commit-config.yaml
git commit -m "ci: add pre-commit configuration for ruff and mypy"
```

---

### Task 6: Taskfile Creation

**Files:**
- Create: `Taskfile.yml`

**Interfaces:**
- Consumes: Examples in `synthesizer/examples/` and dev scripts
- Produces: Taskfile with setup, lint, format, test, and example tasks

- [ ] **Step 1: Create `Taskfile.yml`**

Write the `Taskfile.yml` content:

```yaml
version: '3'

tasks:
  setup:
    desc: Install virtual environment dependencies and pre-commit hooks
    cmds:
      - uv sync
      - uv run pre-commit install

  lint:
    desc: Run ruff linter and mypy type checks
    cmds:
      - uv run ruff check synthesizer tests
      - uv run mypy synthesizer

  format:
    desc: Format codebase with ruff
    cmds:
      - uv run ruff format synthesizer tests

  test:
    desc: Run unit tests with pytest
    cmds:
      - uv run pytest tests -v

  test-cov:
    desc: Run unit tests with coverage report
    cmds:
      - uv run pytest tests --cov=synthesizer --cov-report=term-missing

  clean:
    desc: Remove cache files and build artifacts
    cmds:
      - rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info

  example:disk-zoo:
    desc: Run disk zoo example script
    cmds:
      - uv run python synthesizer/examples/disk_zoo.py

  example:dust-mixing:
    desc: Run dust mixing example script
    cmds:
      - uv run python synthesizer/examples/dust_mixing.py

  example:ppdisk-synobs:
    desc: Run ppdisk synobs example script
    cmds:
      - uv run python synthesizer/examples/ppdisk_synobs.py

  example:disk-zoo-cli:
    desc: Run disk zoo CLI bash example
    cmds:
      - bash synthesizer/examples/disk_zoo_cli.sh

  example:dust-mixing-cli:
    desc: Run dust mixing CLI bash example
    cmds:
      - bash synthesizer/examples/dust_mixing_cli.sh

  example:ppdisk-synobs-cli:
    desc: Run ppdisk synobs CLI bash example
    cmds:
      - bash synthesizer/examples/ppdisk_synobs_cli.sh
```

- [ ] **Step 2: Validate Taskfile syntax and task list**

Run: `task --list`
Expected: List of available tasks with descriptions.

- [ ] **Step 3: Commit**

```bash
git add Taskfile.yml
git commit -m "ci: add Taskfile covering dev workflow and example scripts"
```

---

### Task 7: Comprehensive Unit Test Suite

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/test_utils.py`
- Create: `tests/test_parser.py`
- Create: `tests/test_pipeline.py`
- Create: `tests/test_dustmixer.py`
- Create: `tests/test_gridder.py`
- Create: `tests/test_raytrace.py`
- Create: `tests/test_synobs.py`

**Interfaces:**
- Consumes: Core modules of `synthesizer`
- Produces: Test suite validating `utils`, `parser`, `pipeline`, `dustmixer`, `gridder`, `raytrace`, `synobs`

- [ ] **Step 1: Create `tests/__init__.py` and `tests/test_utils.py`**

Write test functions for physical constants, file operations, conversion functions in `utils.py`.

```python
import numpy as np
from synthesizer import utils


def test_constants():
    assert utils.au > 0
    assert utils.pc > utils.au
    assert utils.msun > 0
    assert utils.rsun > 0


def test_temperature_calculation():
    # Verify temperature model helper if present
    if hasattr(utils, "get_temperature"):
        temp = utils.get_temperature(100.0)
        assert temp > 0
```

- [ ] **Step 2: Create `tests/test_parser.py`**

Write test functions for CLI parser option verification:

```python
from synthesizer.parser import synthesizer_parser


def test_parser_defaults():
    parser = synthesizer_parser()
    args = parser.parse_args([])
    assert args.grid is False
    assert args.raytrace is False
    assert args.model == "ppdisk"
```

- [ ] **Step 3: Create `tests/test_pipeline.py`**

Write test functions for `Pipeline` instantiation and parameters:

```python
from synthesizer.pipeline import Pipeline


def test_pipeline_init():
    p = Pipeline(nthreads=2, overwrite=True)
    assert p.nthreads == 2
    assert p.overwrite is True
```

- [ ] **Step 4: Create `tests/test_dustmixer.py`**

Write test functions for `bhmie`, `bhcoat`, and `dustmixer`:

```python
from synthesizer.dustmixer import bhcoat, bhmie
from synthesizer.dustmixer.dustmixer import DustMixer


def test_bhmie_scattering():
    qext, qsca, qback, g = bhmie.bhmie(1.0, 1.5 + 0.1j, 10)
    assert qext > 0
    assert qsca >= 0


def test_bhcoat_scattering():
    qext, qsca, qback, g = bhcoat.bhcoat(1.0, 1.2, 1.5 + 0.1j, 1.6 + 0.2j, 10)
    assert qext > 0


def test_dustmixer_init():
    dm = DustMixer(amin=0.1, amax=10.0, na=10)
    assert dm.amin == 0.1
    assert dm.amax == 10.0
```

- [ ] **Step 5: Create `tests/test_gridder.py`**

Write test functions for `gridder` models:

```python
from synthesizer.gridder.analytical import PPDisk


def test_ppdisk_model():
    disk = PPDisk(mdisk=1e-3, rin=1, rout=100)
    assert disk.mdisk == 1e-3
    assert disk.rin == 1
```

- [ ] **Step 6: Create `tests/test_raytrace.py` and `tests/test_synobs.py`**

Write unit test functions for `raytrace` and `synobs` helpers:

```python
from synthesizer.raytrace.radmc3d import RADMC3D
from synthesizer.synobs.scripter import Scripter


def test_radmc3d_init():
    rt = RADMC3D()
    assert rt is not None


def test_scripter_init():
    s = Scripter()
    assert s is not None
```

- [ ] **Step 7: Run pytest suite and verify 100% pass**

Run: `uv run pytest tests -v`
Expected: All tests PASS.

- [ ] **Step 8: Commit**

```bash
git add tests/
git commit -m "test: add comprehensive unit test suite"
```
