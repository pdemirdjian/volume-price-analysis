"""Every environment variable the app reads is documented for users."""

import re
from pathlib import Path

import pytest

from tests.conftest import _APP_ENV_VARS

_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("doc", [".env.example", "README.md"])
@pytest.mark.parametrize("var", _APP_ENV_VARS)
def test_env_var_is_documented(doc, var):
    # Word boundaries: AI_PROVIDER must not pass on the strength of AI_PROVIDER_API_KEY.
    assert re.search(rf"\b{re.escape(var)}\b", (_ROOT / doc).read_text(encoding="utf-8"))
