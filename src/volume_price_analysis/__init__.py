"""Volume-Price Analysis MCP Server."""

from importlib.metadata import PackageNotFoundError, version

try:
    # Single source of truth: the dist version release-please writes to pyproject.toml.
    __version__ = version("volume-price-analysis-mcp")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0+unknown"
