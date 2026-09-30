from qpiai_quantum.authentication.user import (
    SDKUser,
    clear_user,
    get_user,
    set_user,
    user_context,
)


def test_user_context_restores_complete_user() -> None:
    previous_user = get_user()
    original_user = SDKUser(
        name="Test User",
        email="test@example.com",
        api_key="original-key",
    )
    set_user(original_user)

    try:
        with user_context("temporary-key"):
            assert get_user() == SDKUser(
                name="",
                email="",
                api_key="temporary-key",
            )

        assert get_user() == original_user
    finally:
        if previous_user is None:
            clear_user()
        else:
            set_user(previous_user)


def test_user_context_restores_identity_after_error() -> None:
    """An execution failure must not erase the outer login profile."""
    import pytest

    previous_user = get_user()
    original_user = SDKUser(
        name="Auth Test", email="auth@example.test", api_key="outer-key"
    )
    set_user(original_user)
    try:
        with pytest.raises(RuntimeError, match="execution failed"):
            with user_context("temporary-key"):
                raise RuntimeError("execution failed")
        assert get_user() == original_user
    finally:
        if previous_user is None:
            clear_user()
        else:
            set_user(previous_user)
