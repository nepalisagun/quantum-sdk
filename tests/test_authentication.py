"""Tests for shared SDK authentication helpers."""

from collections.abc import Generator

import pytest

from qpiai_quantum.authentication.user import (
    SDKUser,
    clear_user,
    get_user,
    get_active_auth_header,
    get_active_api_key,
    resolve_api_key,
    set_user,
)
from qpiai_quantum.authentication.exceptions import APIKeyInvalidError
from qpiai_quantum.jobmanager.execution_engine import ExecutionEngine


@pytest.fixture(autouse=True)
def clear_authenticated_user() -> Generator[None, None, None]:
    """Keep process-global SDK authentication isolated between tests."""
    previous_user = get_user()
    clear_user()
    try:
        yield
    finally:
        if previous_user is None:
            clear_user()
        else:
            set_user(previous_user)


def test_resolve_api_key_uses_explicit_key_before_other_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit credentials are normalized and have the highest precedence."""
    monkeypatch.setenv("API_KEY", "environment-key")
    set_user(SDKUser(name="Test User", email="test@qpiai.tech", api_key="active-key"))

    assert resolve_api_key(" explicit-key ") == "explicit-key"


def test_resolve_api_key_uses_active_user_before_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Logged-in SDK users do not need to repeat their API key."""
    monkeypatch.setenv("API_KEY", "environment-key")
    set_user(SDKUser(name="Test User", email="test@qpiai.tech", api_key="active-key"))

    assert resolve_api_key() == "active-key"


def test_active_auth_helpers_normalize_credentials() -> None:
    """Active-user clients receive a valid, consistent auth header."""
    set_user(SDKUser(name="Test User", email="test@qpiai.tech", api_key=" active-key "))

    assert get_active_api_key() == "active-key"
    assert get_active_auth_header() == {"X-Secret-Token": "active-key"}


def test_resolve_api_key_ignores_blank_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whitespace-only credentials are not valid HTTP header values."""
    monkeypatch.setenv("API_KEY", "\n")

    assert resolve_api_key() is None


def test_set_user_rejects_blank_credentials() -> None:
    """Whitespace-only API keys cannot enter the active SDK context."""
    with pytest.raises(APIKeyInvalidError):
        set_user("\n")


def test_execution_engine_uses_shared_credential_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cloud execution follows the same precedence as feature clients."""
    monkeypatch.setenv("API_KEY", "environment-key")
    set_user(SDKUser(name="Test User", email="test@qpiai.tech", api_key="active-key"))

    assert ExecutionEngine._get_api_key() == "active-key"
    assert ExecutionEngine._get_api_key(" explicit-key ") == "explicit-key"


def test_execution_engine_rejects_blank_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Execution fails locally instead of passing blank values to requests."""
    monkeypatch.setenv("API_KEY", "\n")

    with pytest.raises(ValueError, match="API_KEY not found"):
        ExecutionEngine._get_api_key()
