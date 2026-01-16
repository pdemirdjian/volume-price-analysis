# UV Setup Guide

This project uses [UV](https://github.com/astral-sh/uv) for fast, reliable Python package and environment management. UV is a modern replacement for pip and virtual environments, written in Rust.

## Why UV?

- ⚡ **10-100x faster** than pip
- 🔒 **Reliable** dependency resolution
- 🎯 **Simple** virtual environment management
- 📦 **Drop-in replacement** for pip and venv

## Installation

### Install UV

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**With pip (any platform):**
```bash
pip install uv
```

**With Homebrew (macOS):**
```bash
brew install uv
```

Verify installation:
```bash
uv --version
```

## Quick Start

### 1. Create Virtual Environment

UV automatically creates and manages virtual environments:

```bash
# Navigate to project directory
cd volume-price-analysis

# Create a virtual environment with Python 3.12
uv venv

# UV will create a .venv directory automatically
```

### 2. Activate Virtual Environment

**macOS/Linux:**
```bash
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
.venv\Scripts\activate
```

**Windows (CMD):**
```cmd
.venv\Scripts\activate.bat
```

### 3. Install Dependencies

```bash
# Install production dependencies
uv pip install -e .

# Install with development dependencies
uv pip install -e ".[dev]"
```

That's it! UV will install everything blazingly fast.

## Common UV Commands

### Package Installation

```bash
# Install a package
uv pip install pandas

# Install specific version
uv pip install "pandas>=2.2.0"

# Install from requirements.txt
uv pip install -r requirements.txt

# Install project in editable mode
uv pip install -e .

# Install with optional dependencies
uv pip install -e ".[dev]"
```

### Package Management

```bash
# List installed packages
uv pip list

# Show package details
uv pip show mcp

# Uninstall package
uv pip uninstall yfinance

# Freeze dependencies
uv pip freeze > requirements.txt
```

### Python Version Management

```bash
# Use specific Python version
uv venv --python 3.12

# UV automatically uses .python-version file
# (Already configured in this project as 3.12)
```

## Development Workflow

### First Time Setup

```bash
# 1. Clone/navigate to project
cd volume-price-analysis

# 2. Create virtual environment (UV detects .python-version)
uv venv

# 3. Activate virtual environment
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate  # Windows

# 4. Install project with dev dependencies
uv pip install -e ".[dev]"

# 5. Verify installation
python example_usage.py
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/volume_price_analysis --cov-report=term-missing

# Run specific test file
pytest tests/test_indicators.py

# Run specific test
pytest tests/test_indicators.py::TestOBV::test_obv_uptrend

# Run with verbose output
pytest -v
```

### Code Quality

```bash
# Format code with ruff
ruff format src/ tests/

# Lint code
ruff check src/ tests/

# Auto-fix linting issues
ruff check --fix src/ tests/

# Type checking
mypy src/
```

## UV vs Traditional Tools

| Task | Traditional | UV |
|------|-------------|-----|
| Create venv | `python -m venv .venv` | `uv venv` |
| Install package | `pip install pandas` | `uv pip install pandas` |
| Install from file | `pip install -r requirements.txt` | `uv pip install -r requirements.txt` |
| Install editable | `pip install -e .` | `uv pip install -e .` |
| Speed | 🐌 Slow | ⚡ 10-100x faster |

## Upgrading Dependencies

```bash
# Upgrade a specific package
uv pip install --upgrade pandas

# Upgrade all packages
uv pip install --upgrade -r requirements.txt

# Upgrade to latest compatible versions
uv pip install -e ".[dev]" --upgrade
```

## Troubleshooting

### UV command not found

Add UV to your PATH:
```bash
export PATH="$HOME/.cargo/bin:$PATH"
```

Add to your shell profile (`~/.bashrc`, `~/.zshrc`, etc.) to make permanent.

### Wrong Python version

Specify Python version explicitly:
```bash
uv venv --python 3.12
```

Or ensure `.python-version` file exists (already configured in this project).

### Cache issues

Clear UV cache:
```bash
uv cache clean
```

### Permission errors

On Unix systems, ensure UV install script is executable:
```bash
chmod +x ~/.cargo/bin/uv
```

## Advanced Usage

### Sync Dependencies

Create a `requirements.lock` file for reproducible installs:

```bash
# Generate lock file
uv pip compile pyproject.toml -o requirements.lock

# Install from lock file
uv pip install -r requirements.lock
```

### Multiple Python Versions

```bash
# Create venv with specific Python
uv venv --python 3.11 .venv-py311
uv venv --python 3.12 .venv-py312

# Activate the one you want
source .venv-py312/bin/activate
```

### Global Package Installation

```bash
# Install package globally (not in venv)
uv pip install --system ruff
```

## Integration with Claude Code

When configuring the MCP server with Claude Code, use the UV-managed Python:

```json
{
  "mcpServers": {
    "volume-price-analysis": {
      "command": "/path/to/volume-price-analysis/.venv/bin/python",
      "args": ["-m", "volume_price_analysis.server"]
    }
  }
}
```

Find the path:
```bash
which python  # When venv is activated
# or
echo "$(pwd)/.venv/bin/python"
```

## Resources

- [UV Documentation](https://github.com/astral-sh/uv)
- [UV Installation Guide](https://github.com/astral-sh/uv#installation)
- [Astral (UV creators)](https://astral.sh/)

## Comparison with Other Tools

UV replaces or improves upon:
- `pip` - Package installer
- `pip-tools` - Dependency management
- `virtualenv` - Virtual environment
- `poetry` - Dependency management (simpler alternative)
- `pipenv` - Environment + dependency management

## Next Steps

1. ✅ Install UV
2. ✅ Create virtual environment: `uv venv`
3. ✅ Activate it: `source .venv/bin/activate`
4. ✅ Install project: `uv pip install -e ".[dev]"`
5. ✅ Run tests: `pytest`
6. ✅ Start coding! 🚀
