"""Every environment variable the app reads is documented for users."""

from pathlib import Path

import pytest

from tests.conftest import _APP_ENV_VARS

_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("doc", [".env.example", "README.md"])
@pytest.mark.parametrize("var", _APP_ENV_VARS)
def test_env_var_is_documented(doc, var):
    assert var in (_ROOT / doc).read_text(encoding="utf-8")
