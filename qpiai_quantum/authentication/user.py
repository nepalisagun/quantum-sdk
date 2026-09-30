from .exceptions import APIKeyInvalidError
from dataclasses import dataclass
import contextvars
import contextlib
import os


@dataclass
class SDKUser:
    """
    Represents a user in the SDK.
    """

    name: str
    email: str
    api_key: str


_user_context: contextvars.ContextVar[SDKUser | None] = contextvars.ContextVar(
    "user", default=None
)

# Global fallback for Jupyter notebooks where contextvars don't persist across cells
_global_user: SDKUser | None = None


def set_user(user_data: SDKUser | str) -> None:
    """
    Set the current user in the context.

    Args:
        user_data (SDKUser | str): The user object or API key to set in the context.
    """
    global _global_user
    if isinstance(user_data, str):
        api_key = normalize_api_key(user_data)
        if api_key is None:
            raise APIKeyInvalidError()
        user = SDKUser(name="", email="", api_key=api_key)
    elif isinstance(user_data, SDKUser):
        api_key = normalize_api_key(user_data.api_key)
        if api_key is None:
            raise APIKeyInvalidError()
        user = SDKUser(
            name=user_data.name,
            email=user_data.email,
            api_key=api_key,
        )
    else:
        raise APIKeyInvalidError()

    _user_context.set(user)
    # Also store globally for Jupyter notebook persistence across cells
    _global_user = user


def get_user() -> SDKUser | None:
    """
    Get the current user from the context.

    Returns:
        SDKUser | None: The current user, or None if no user is set.
    """
    # Try context var first (for multi-threading/async safety)
    user = _user_context.get()
    # Fall back to global if context is empty (Jupyter notebook persistence)
    if user is None:
        user = _global_user
    return user


def normalize_api_key(api_key: str | None) -> str | None:
    """Return a trimmed API key, or ``None`` when the value is blank or invalid."""
    if not isinstance(api_key, str):
        return None
    normalized_key = api_key.strip()
    return normalized_key or None


def get_active_api_key() -> str | None:
    """Return the normalized API key stored by ``QpiAIQuantumAuth.login``."""
    user = get_user()
    return normalize_api_key(user.api_key) if user is not None else None


def get_active_auth_header() -> dict[str, str] | None:
    """Return the authentication header for the active SDK user, if present."""
    api_key = get_active_api_key()
    return {"X-Secret-Token": api_key} if api_key is not None else None


def resolve_api_key(api_key: str | None = None) -> str | None:
    """Resolve a usable API key from explicit, active-user, and environment values.

    Explicit credentials take precedence, followed by the API key stored by
    :meth:`QpiAIQuantumAuth.login`, then ``API_KEY`` from the environment.
    Empty and whitespace-only values are ignored so they cannot be sent as
    invalid HTTP headers.
    """
    candidates = (
        api_key,
        get_active_api_key(),
        os.getenv("API_KEY"),
    )
    for candidate in candidates:
        normalized_key = normalize_api_key(candidate)
        if normalized_key is not None:
            return normalized_key
    return None


def clear_user() -> None:
    """
    Clear the current user from the context.
    """
    global _global_user
    _user_context.set(None)
    _global_user = None


@contextlib.contextmanager
def user_context(api_key: str):
    """
    Context manager to temporarily set the active user context for this thread/task.
    Ensures that the context is reset when done. Safe for multi-user, async, and notebook/server use.

    Usage:
        with user_context(api_key):
            ...  # cloud calls with this user
        # Context reset automatically
    """
    old_user = get_user()
    set_user(api_key)
    try:
        yield
    finally:
        if old_user is not None:
            # Restore previous user context
            set_user(old_user)
        else:
            clear_user()
