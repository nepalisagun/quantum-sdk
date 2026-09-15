import os

import pytest
from dotenv import load_dotenv

from qpiai_quantum import QpiAIQuantumAuth

load_dotenv("qcloud.env")
load_dotenv()


@pytest.fixture(scope="session")
def authenticated_sdk_user():
    """Authenticate tests that exercise the access-controlled local backend."""
    api_key = os.getenv("API_KEY")
    if not api_key:
        pytest.skip("API_KEY is required for authenticated QSV-Local tests")

    try:
        return QpiAIQuantumAuth.login(api_key)
    except Exception:
        pytest.fail(
            "QSV-Local test authentication failed; credentials were not displayed",
            pytrace=False,
        )
