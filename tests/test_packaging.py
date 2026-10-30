"""Packaging invariants: version single-sourcing and the PEP 561 marker."""

from importlib.metadata import version
from pathlib import Path

import volume_price_analysis
from volume_price_analysis import server

DIST_NAME = "volume-price-analysis-mcp"


def test_dunder_version_matches_installed_dist():
    assert volume_price_analysis.__version__ == version(DIST_NAME)


def test_server_reports_same_version_as_package():
    assert server._SERVER_VERSION == volume_price_analysis.__version__


def test_py_typed_marker_ships_in_package():
    package_dir = Path(volume_price_analysis.__file__).parent
    assert (package_dir / "py.typed").is_file()
