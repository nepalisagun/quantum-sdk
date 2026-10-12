"""Regression coverage for hardware dispatch and resource selection."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

from qpiai_quantum.circuit import Circuit
from qpiai_quantum.jobmanager import Backend, JobManager


def test_indus_circuit_dispatch_matches_backend_enum() -> None:
    with patch(
        "qpiai_quantum.jobmanager.ExecutionEngine.execute_circuit",
        return_value=SimpleNamespace(counts=None),
    ) as execute:
        Circuit(1).run(device_name="QpiAI-Indus-1")

    submitted = execute.call_args.kwargs
    assert (submitted["method"], submitted["device_name"]) == (
        Backend.INDUS_QPU.to_method_and_device()
    )
    assert submitted["need_statevector"] is False
    assert submitted["need_density_matrix"] is False


def test_indus_qpu_method_resolves_backend_resource() -> None:
    response = Mock(status_code=200)
    response.json.return_value = {
        "compute_resources": [
            {"ID": "indus-resource", "backend_name": "QpiAI-Indus-1"},
            {"ID": "simulator-resource", "backend_name": "QpiAI-QSV-Simulator"},
        ]
    }
    manager = JobManager()
    with (
        patch.object(JobManager, "_resource_cache", {}),
        patch.object(manager, "_get_auth_header", return_value={}),
        patch("requests.get", return_value=response) as get,
    ):
        assert manager._get_compute_resource_id("qpu", "QpiAI-Indus-1") == (
            "indus-resource"
        )
        assert manager._get_compute_resource_id("qpu", "QpiAI-Indus-1") == (
            "indus-resource"
        )
        get.assert_called_once()
