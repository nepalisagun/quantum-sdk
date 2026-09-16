from .exceptions import APIKeyInvalidError
from dataclasses import dataclass
import contextvars
import contextlib


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
        if not user_data:
            raise APIKeyInvalidError()
        user = SDKUser(name="", email="", api_key=user_data)
    elif isinstance(user_data, SDKUser):
        user = user_data
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
