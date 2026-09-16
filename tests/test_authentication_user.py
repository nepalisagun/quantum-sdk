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
