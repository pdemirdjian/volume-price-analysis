# Contributing to Volume-Price Analysis

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Getting Started

### Prerequisites

- Python 3.14+
- [UV](https://github.com/astral-sh/uv) (recommended) or pip

### Development Setup

1. Fork and clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/volume-price-analysis.git
   cd volume-price-analysis
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   # Using UV (recommended)
   uv venv
   source .venv/bin/activate
   uv pip install -e ".[dev]"

   # Or using pip
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```

3. Run the tests to verify your setup:
   ```bash
   pytest
   ```

## Development Workflow

### Code Style

This project uses:
- **Ruff** for formatting and linting
- **mypy** for type checking

Before submitting a PR, ensure your code passes all checks:

```bash
# Format code
ruff format src/ tests/

# Lint and auto-fix issues
ruff check --fix src/ tests/

# Type checking
mypy src/
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/volume_price_analysis --cov-report=term-missing

# Run specific test file
pytest tests/test_indicators.py
```

### Adding New Indicators

When adding new technical indicators:

1. Add the calculation function to `src/volume_price_analysis/indicators.py`
2. Add corresponding tests in `tests/test_indicators.py`
3. If exposing as an MCP tool, add the tool definition in `src/volume_price_analysis/server.py`
4. Update the README.md with documentation

## Pull Request Process

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit with clear, descriptive messages

3. Ensure all tests pass and code quality checks are clean

4. Push your branch and open a Pull Request

5. Provide a clear description of the changes and their purpose

## Reporting Issues

When reporting bugs, please include:
- Python version
- Operating system
- Steps to reproduce the issue
- Expected vs actual behavior
- Any relevant error messages

## Feature Requests

Feature requests are welcome! Please open an issue describing:
- The problem you're trying to solve
- Your proposed solution
- Any alternatives you've considered

## Questions?

Feel free to open an issue for any questions about contributing.
