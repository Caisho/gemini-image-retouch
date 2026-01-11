# Project Overview

This is a Python project template that provides a standardized structure for Python projects. It uses modern Python development tools and follows best practices for project organization, testing, and code quality.

## Folder Structure

- `/src`: Contains the main source code for the project.
- `/tests`: Contains all test files organized into unit and integration test directories.
  - `/tests/unit`: Unit tests for individual components.
  - `/tests/integration`: Integration tests for component interactions.
- `/notebooks`: Contains Jupyter notebooks for data analysis and experimentation.
- `/.github`: Contains GitHub-specific files including this copilot instructions file.

## Libraries and Frameworks

- **UV**: Modern Python package and project manager for dependency management and virtual environments.
- **pytest**: Testing framework with coverage support via pytest-cov.
- **Ruff**: Fast Python linter and formatter for code quality and style enforcement.
- **Vulture**: Dead code detection tool.
- **Jupyter**: Interactive notebooks for data analysis (ipykernel, ipython).

## Coding Standards

- **Line length**: Maximum 79 characters (configured in pyproject.toml).
- **Indentation**: 4 spaces (no tabs).
- **Quote style**: Double quotes for strings.
- **Python version**: Requires Python >=3.10, <3.13 (target version 3.13).
- **Import organization**: Follow PEP 8 import guidelines.
- **Docstrings**: Use for all public functions, classes, and modules.

## Development Workflow

### Python Script Execution
Always use `uv run` instead of `python` directly. This ensures scripts are executed within the proper virtual environment with correct dependencies.

Examples:
- Instead of `python script.py`, use `uv run python script.py`
- Instead of `python -m module`, use `uv run python -m module`
- For direct script execution: `uv run script.py`

### Code Quality
- **Format code**: `ruff format`
- **Lint code**: `ruff check`
- **Run tests**: `pytest`
- **Check for dead code**: `vulture src/`
- **Test with coverage**: `pytest --cov=src`

### Project Setup
1. Create virtual environment: `uv venv --python 3.13`
2. Activate environment: `source .venv/bin/activate`
3. Install dependencies: `uv sync --extra dev`
