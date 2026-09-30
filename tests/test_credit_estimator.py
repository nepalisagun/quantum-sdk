"""Tests for the QpiAI Credit Estimator client."""

from collections.abc import Generator
from unittest.mock import MagicMock

import pytest
import requests

from qpiai_quantum.authentication.user import SDKUser, clear_user, get_user, set_user
from qpiai_quantum.config import HTTP_TIMEOUT
from qpiai_quantum.creditestimator import CreditEstimator, CreditEstimatorError


COMPUTE_RESOURCE_ID = "d6e9e5d2-48c8-421a-8a3d-d0719a95568c"
QASM = (
    "OPENQASM 2.0;\n"
    'include "qelib1.inc";\n'
    "qreg q[3];\n"
    "creg c[1];\n"
    "h q[0];\n"
    "h q[1];\n"
    "s q[0];\n"
    "s q[1];\n"
    "h q[0];\n"
    "cx q[0], q[1];\n"
    "cx q[1], q[0];\n"
    "h q[1];\n"
    "h q[0];\n"
    "cx q[1], q[2];\n"
    "measure q[0] -> c[0];\n"
)


@pytest.fixture(autouse=True)
def clear_authenticated_user() -> Generator[None, None, None]:
    """Keep active SDK authentication isolated between tests."""
    previous_user = get_user()
    clear_user()
    try:
        yield
    finally:
        if previous_user is None:
            clear_user()
        else:
            set_user(previous_user)


def test_estimate_for_resource_sends_supplied_payload() -> None:
    """The supplied resource ID and OpenQASM are posted unchanged."""
    session = MagicMock()
    response = MagicMock(ok=True)
    response.json.return_value = {"estimated_credits": 1.5}
    session.request.return_value = response
    base_url = "https://credit.example.test/api/credit-estimator/"
    client = CreditEstimator(api_key="test-key", base_url=base_url, session=session)

    result = client.estimate_for_resource(QASM, COMPUTE_RESOURCE_ID)

    assert result == {"estimated_credits": 1.5}
    session.request.assert_called_once_with(
        "POST",
        "https://credit.example.test/api/credit-estimator/estimate",
        headers={"X-Secret-Token": "test-key"},
        json={"compute_resource_id": COMPUTE_RESOURCE_ID, "qasm": QASM},
        timeout=HTTP_TIMEOUT,
    )


def test_estimate_for_device_resolves_an_active_resource_before_estimating() -> None:
    """Device names are mapped to the ID required by the estimator API."""
    session = MagicMock()
    resources_response = MagicMock(ok=True)
    resources_response.json.return_value = [
        {
            "ID": COMPUTE_RESOURCE_ID,
            "backend_name": "QpiAI-QSV-Simulator",
            "device_name": "QpiAI-QSV-Simulator",
            "status": "active",
        }
    ]
    estimate_response = MagicMock(ok=True)
    estimate_response.json.return_value = {"credits": 1}
    session.request.side_effect = [resources_response, estimate_response]
    client = CreditEstimator(
        api_key="test-key",
        base_url="https://credit.example.test/api/credit-estimator",
        session=session,
    )

    result = client.estimate_for_device(QASM, "QpiAI-QSV-Simulator")

    assert result == {"credits": 1}
    assert session.request.call_args_list[0].args == (
        "GET",
        "https://qcloud-server.qpiai.tech/api/compute-resources",
    )
    assert session.request.call_args_list[0].kwargs["json"] is None
    assert session.request.call_args_list[1].args == (
        "POST",
        "https://credit.example.test/api/credit-estimator/estimate",
    )
    assert session.request.call_args_list[1].kwargs["json"] == {
        "compute_resource_id": COMPUTE_RESOURCE_ID,
        "qasm": QASM,
    }


def test_estimate_for_device_accepts_backend_name() -> None:
    """Backend names are accepted when they differ from the device name."""
    session = MagicMock()
    resources_response = MagicMock(ok=True)
    resources_response.json.return_value = [
        {
            "ID": COMPUTE_RESOURCE_ID,
            "backend_name": "backend-name",
            "device_name": "device-name",
            "status": "active",
        }
    ]
    estimate_response = MagicMock(ok=True)
    estimate_response.json.return_value = {"credits": 1}
    session.request.side_effect = [resources_response, estimate_response]
    client = CreditEstimator(api_key="test-key", session=session)

    client.estimate_for_device(QASM, "backend-name")

    assert session.request.call_count == 2


def test_estimate_for_device_rejects_unknown_or_inactive_resources() -> None:
    """Only one active resource may match the requested device."""
    session = MagicMock()
    resources_response = MagicMock(ok=True)
    resources_response.json.return_value = [
        {
            "ID": COMPUTE_RESOURCE_ID,
            "device_name": "QpiAI-QSV-Simulator",
            "status": "inactive",
        }
    ]
    session.request.return_value = resources_response
    client = CreditEstimator(api_key="test-key", session=session)

    with pytest.raises(CreditEstimatorError, match="No active compute resource"):
        client.estimate_for_device(QASM, "QpiAI-QSV-Simulator")


def test_estimate_for_device_rejects_ambiguous_resources() -> None:
    """Ambiguous device-name matches cannot silently select a resource."""
    session = MagicMock()
    resources_response = MagicMock(ok=True)
    resources_response.json.return_value = [
        {"ID": "resource-1", "device_name": "shared-device", "status": "active"},
        {"ID": "resource-2", "device_name": "shared-device", "status": "active"},
    ]
    session.request.return_value = resources_response
    client = CreditEstimator(api_key="test-key", session=session)

    with pytest.raises(CreditEstimatorError, match="Multiple active compute resources"):
        client.estimate_for_device(QASM, "shared-device")


def test_uses_configured_server_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default endpoint follows the SDK server configuration."""
    monkeypatch.setenv("QPIAI_SERVER_URL", "https://staging.example.test/")

    client = CreditEstimator(api_key="test-key", session=MagicMock())

    assert client.base_url == "https://staging.example.test/api/credit-estimator"


def test_uses_active_sdk_login_when_no_api_key_is_supplied() -> None:
    """Authenticated SDK users can use credit estimation without duplicating a key."""
    session = MagicMock()
    response = MagicMock(ok=True)
    response.json.return_value = {"estimated_credits": 1.5}
    session.request.return_value = response
    set_user(SDKUser(name="Test User", email="test@qpiai.tech", api_key="active-key"))

    CreditEstimator(session=session).estimate_for_resource(QASM, COMPUTE_RESOURCE_ID)

    assert session.request.call_args.kwargs["headers"] == {
        "X-Secret-Token": "active-key"
    }


def test_explicit_api_key_takes_precedence_over_active_sdk_login() -> None:
    """A per-client key can override the key stored by SDK authentication."""
    set_user(SDKUser(name="Test User", email="test@qpiai.tech", api_key="active-key"))

    client = CreditEstimator(api_key="explicit-key", session=MagicMock())

    assert client.api_key == "explicit-key"


def test_requires_an_api_key_before_sending_a_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing credentials fail locally without contacting the service."""
    monkeypatch.delenv("API_KEY", raising=False)
    session = MagicMock()
    client = CreditEstimator(session=session)

    with pytest.raises(CreditEstimatorError, match="API key is required"):
        client.estimate_for_resource(QASM, COMPUTE_RESOURCE_ID)

    session.request.assert_not_called()


@pytest.mark.parametrize(
    "qasm, compute_resource_id", [("", COMPUTE_RESOURCE_ID), (QASM, "")]
)
def test_estimate_for_resource_requires_non_empty_inputs(
    qasm: str, compute_resource_id: str
) -> None:
    client = CreditEstimator(api_key="test-key", session=MagicMock())

    with pytest.raises(ValueError):
        client.estimate_for_resource(qasm, compute_resource_id)


def test_request_raises_a_typed_error_for_http_failures() -> None:
    """HTTP failures retain status without exposing backend details."""
    session = MagicMock()
    response = MagicMock(ok=False, status_code=403)
    response.json.return_value = {"detail": "private-backend-detail"}
    session.request.return_value = response
    client = CreditEstimator(api_key="test-key", session=session)

    with pytest.raises(CreditEstimatorError, match="403") as exc_info:
        client.estimate_for_resource(QASM, COMPUTE_RESOURCE_ID)

    assert exc_info.value.status_code == 403
    assert "private-backend-detail" not in str(exc_info.value)
    response.json.assert_not_called()


def test_request_raises_a_typed_error_for_invalid_success_json() -> None:
    """A successful but non-JSON response is reported clearly."""
    session = MagicMock()
    response = MagicMock(ok=True)
    response.json.side_effect = ValueError("not JSON")
    session.request.return_value = response
    client = CreditEstimator(api_key="test-key", session=session)

    with pytest.raises(CreditEstimatorError, match="invalid JSON"):
        client.estimate_for_resource(QASM, COMPUTE_RESOURCE_ID)


def test_request_raises_a_typed_error_for_network_failures() -> None:
    """Request failures are wrapped in the feature-specific exception."""
    session = MagicMock()
    session.request.side_effect = requests.RequestException("connection refused")
    client = CreditEstimator(api_key="test-key", session=session)

    with pytest.raises(CreditEstimatorError, match="Could not reach"):
        client.estimate_for_resource(QASM, COMPUTE_RESOURCE_ID)


@pytest.mark.parametrize("base_url", ["", "   "])
def test_base_url_must_be_non_empty(base_url: str) -> None:
    """Explicit endpoint overrides must be usable URLs."""
    with pytest.raises(ValueError, match="base_url must be a non-empty string"):
        CreditEstimator(api_key="test-key", base_url=base_url)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
