"""Client for QpiAI's hosted credit-estimation API."""

import requests
from requests import Session
from typing import Any

from ..authentication.user import resolve_api_key
from ..config import HTTP_TIMEOUT, get_api_url


class CreditEstimatorError(RuntimeError):
    """Raised when the credit-estimator service cannot fulfil a request."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class CreditEstimator:
    """Estimate execution credits for OpenQASM circuits on QpiAI resources."""

    API_PATH = "/api/credit-estimator"
    COMPUTE_RESOURCES_PATH = "/api/compute-resources"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | tuple[float, float] = HTTP_TIMEOUT,
        session: Session | None = None,
    ) -> None:
        """Create a client.

        Args:
            api_key: API key to use. If omitted, the active SDK login is used,
                followed by the ``API_KEY`` environment variable.
            base_url: Credit-estimator API base URL. Defaults to the configured
                QpiAI server URL plus ``/api/credit-estimator``.
            timeout: Connect and read timeout forwarded to ``requests``.
            session: Optional requests session, primarily for custom transport
                configuration and testing.
        """
        if base_url is not None:
            self._require_text(base_url, "base_url")
        self.api_key = resolve_api_key(api_key)
        self.base_url = (base_url or get_api_url(self.API_PATH)).rstrip("/")
        self.timeout = timeout
        self.session = session or Session()

    def estimate_for_resource(
        self, qasm: str, compute_resource_id: str
    ) -> dict[str, Any]:
        """Estimate credits for an OpenQASM circuit on a compute resource."""
        self._require_text(qasm, "qasm")
        self._require_text(compute_resource_id, "compute_resource_id")
        return self._request(
            "POST",
            "/estimate",
            {"compute_resource_id": compute_resource_id, "qasm": qasm},
        )

    def estimate_for_device(self, qasm: str, device_name: str) -> dict[str, Any]:
        """Estimate credits for an OpenQASM circuit by device name.

        The credit-estimator API requires a compute-resource ID. This method
        resolves an active resource matching ``device_name`` through the SDK's
        compute-resources API before requesting the estimate.
        """
        self._require_text(qasm, "qasm")
        self._require_text(device_name, "device_name")
        compute_resource_id = self._resolve_compute_resource_id(device_name)
        return self.estimate_for_resource(qasm, compute_resource_id)

    def _request(
        self, method: str, path: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        return self._request_url(method, f"{self.base_url}{path}", payload)

    def _resolve_compute_resource_id(self, device_name: str) -> str:
        """Return the ID of the active resource matching a device name."""
        resources = self._request_url(
            "GET",
            get_api_url(self.COMPUTE_RESOURCES_PATH),
            None,
        )
        if not isinstance(resources, list):
            raise CreditEstimatorError(
                "Compute resources API returned an invalid response"
            )

        matching_resources = [
            resource
            for resource in resources
            if isinstance(resource, dict)
            and resource.get("status") == "active"
            and device_name
            in {resource.get("device_name"), resource.get("backend_name")}
        ]
        if not matching_resources:
            raise CreditEstimatorError(
                f"No active compute resource found for device '{device_name}'"
            )
        if len(matching_resources) > 1:
            raise CreditEstimatorError(
                f"Multiple active compute resources found for device '{device_name}'"
            )

        compute_resource_id = matching_resources[0].get("ID")
        if not isinstance(compute_resource_id, str) or not compute_resource_id.strip():
            raise CreditEstimatorError(
                f"Compute resource for device '{device_name}' has no valid ID"
            )
        return compute_resource_id

    def _request_url(
        self, method: str, url: str, payload: dict[str, Any] | None
    ) -> Any:
        if not self.api_key:
            raise CreditEstimatorError(
                "An API key is required. Pass api_key= or set API_KEY."
            )
        try:
            response = self.session.request(
                method,
                url,
                headers={"X-Secret-Token": self.api_key},
                json=payload,
                timeout=self.timeout,
            )
        except requests.RequestException:
            raise CreditEstimatorError(
                "Could not reach the Credit Estimator API. Check connectivity and retry."
            ) from None
        if not response.ok:
            raise CreditEstimatorError(
                f"Credit Estimator API returned {response.status_code}. Check credentials and request inputs.",
                status_code=response.status_code,
            )
        try:
            return response.json()
        except ValueError:
            raise CreditEstimatorError(
                "Credit Estimator API returned invalid JSON"
            ) from None

    @staticmethod
    def _require_text(value: str, name: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
