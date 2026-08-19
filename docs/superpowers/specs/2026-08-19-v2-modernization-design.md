# Synthesizer v2 Modernization & Quality Upgrade Design

## Overview
This document specifies the design for modernizing the `synthesizer` codebase (astro-synthesizer v2.0.0). The upgrade transforms the legacy Python codebase into a modern, robust, type-checked, linted, and fully tested Python 3.13+ package managed by `uv`.

---

## 1. Project Management & Python 3.13 Upgrade (`uv`)

### 1.1 Python Version
- Target Python version: `3.13` (minimum constraint `requires-python = ">=3.13"`).
- `.python-version` set to `3.13`.

### 1.2 Dependency & Environment Management
- Primary environment and package manager: `uv`.
- `pyproject.toml` updated to include:
  - Runtime dependencies (`astropy`, `matplotlib`, `numpy`, `scipy`, `mpmath`, `progressbar2`, `aplpy`, `h5py`, `yt`, `pyqt5`, `mayavi`, `requests`).
  - Development dependency group `[dependency-groups]` containing `pytest`, `pytest-cov`, `mypy`, `ruff`, `pre-commit`, `types-requests`.
- Generator of reproducible lockfile: `uv lock`.

---

## 2. Code Quality, Linting, & Type Checking

### 2.1 Ruff Configuration
- Managed in `pyproject.toml` under `[tool.ruff]`:
  - Target Python version: `py313`.
  - Selected linter rules: `E` (Pycodestyle errors), `F` (Pyflakes), `W` (Warnings), `I` (isort), `UP` (pyupgrade), `B` (flake8-bugbear).
  - Target line length: 88.
  - Auto-fixes applied across all codebase files.

### 2.2 Mypy Configuration & Type Hints
- Managed in `pyproject.toml` under `[tool.mypy]`:
  - `python_version = "3.13"`
  - `warn_return_any = true`, `warn_unused_configs = true`, `disallow_untyped_defs = false` (enabled per module progressively).
  - Overrides for non-typed scientific libraries (`astropy.*`, `yt.*`, `mayavi.*`, `aplpy.*`, `h5py.*`, `scipy.*`, `progressbar.*`).
- Concrete type annotations added across core modules (`synthesizer/utils.py`, `synthesizer/parser.py`, `synthesizer/pipeline.py`, `synthesizer/dustmixer/`, `synthesizer/gridder/`, `synthesizer/raytrace/`, `synthesizer/synobs/`).

---

## 3. Git Hooks (`pre-commit`)

### 3.1 `.pre-commit-config.yaml`
Hooks configured:
- `ruff` (linter)
- `ruff-format` (formatter)
- `mypy` (type checking)
- `check-yaml`, `end-of-file-fixer`, `trailing-whitespace`, `check-added-large-files`

---

## 4. Taskfile (`Taskfile.yml`)

`Taskfile.yml` will expose clean commands using `uv`:
- **Development / QA**:
  - `task setup`: Install virtual environment and pre-commit hooks (`uv sync`, `uv run pre-commit install`).
  - `task lint`: Run `uv run ruff check .` and `uv run mypy synthesizer`.
  - `task format`: Run `uv run ruff format .`.
  - `task test`: Run `uv run pytest`.
  - `task test-cov`: Run `uv run pytest --cov=synthesizer --cov-report=term-missing`.
  - `task clean`: Remove temporary files, build artifacts, cache files (`.pytest_cache`, `.mypy_cache`, `.ruff_cache`, build output).
- **Examples Execution**:
  - `task example:disk-zoo`: Run `uv run python synthesizer/examples/disk_zoo.py`
  - `task example:dust-mixing`: Run `uv run python synthesizer/examples/dust_mixing.py`
  - `task example:ppdisk-synobs`: Run `uv run python synthesizer/examples/ppdisk_synobs.py`
  - `task example:disk-zoo-cli`: Execute `synthesizer/examples/disk_zoo_cli.sh`
  - `task example:dust-mixing-cli`: Execute `synthesizer/examples/dust_mixing_cli.sh`
  - `task example:ppdisk-synobs-cli`: Execute `synthesizer/examples/ppdisk_synobs_cli.sh`

---

## 5. Full Codebase Analysis & Unit Test Suite

### 5.1 Analysis Summary of Existing Codebase
- **`synthesizer/parser.py`**: CLI parser for `synthesizer` executable using `argparse`. Sets up parameters for grid generation, radiative transfer, raytracing, and synthetic observations.
- **`synthesizer/pipeline.py`**: Core `Pipeline` class linking gridder, dustmixer, raytrace, and synobs.
- **`synthesizer/utils.py`**: Helper utilities (file IO, unit conversions, physical constants, plotting helpers).
- **`synthesizer/dustmixer/`**: Mie scattering calculations (`bhmie.py`, `bhcoat.py`) and dust composition mixer (`dustmixer.py`).
- **`synthesizer/gridder/`**: Grid generators for analytical models (e.g. protoplanetary disks `ppdisk`, envelope models, hydro readers).
- **`synthesizer/raytrace/`**: RADMC-3D interface (`radmc3d.py`).
- **`synthesizer/synobs/`**: CASA synthetic observations scripting (`scripter.py`) and image analysis (`syn_imager.py`).

### 5.2 Test Architecture
Pytest test suite under `tests/`:
- `tests/test_utils.py`: Unit tests for physical constants, unit conversions, helper functions in `utils.py`.
- `tests/test_parser.py`: Unit tests for command line parsing options in `parser.py`.
- `tests/test_pipeline.py`: Unit tests for `Pipeline` initialization, default settings, and component invocation logic.
- `tests/test_dustmixer.py`: Unit tests for Mie calculations (`bhmie`, `bhcoat`) and `DustMixer`.
- `tests/test_gridder.py`: Unit tests for analytical models, grid construction, and custom models in `gridder/`.
- `tests/test_raytrace.py`: Unit tests for `RADMC3D` parameters generation and control files output.
- `tests/test_synobs.py`: Unit tests for `CASA` script generation templates in `scripter.py`.

---

## 6. Verification & Validation Plan
1. `uv sync` installs clean Python 3.13 venv without conflicts.
2. `uv run ruff check .` passes with 0 errors.
3. `uv run mypy synthesizer` passes with 0 type errors.
4. `uv run pre-commit run --all-files` passes.
5. `task --list` shows all tasks, and `task test` passes all unit tests with full coverage of core modules.
