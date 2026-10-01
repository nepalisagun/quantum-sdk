import json
from pathlib import Path

import numpy as np
import pytest

from qpiai_quantum.jobmanager.exceptions import InvalidResponseError
from qpiai_quantum.jobmanager.jobmanager import JobManager
from qpiai_quantum.jobmanager.result_contract import (
    ContractValidationError,
    validate_binary_response,
    validate_json_response,
)


FIXTURES = Path(__file__).parents[1] / "contracts" / "simulator-results" / "v1"


def _fixture(name: str) -> dict[str, object]:
    return json.loads((FIXTURES / "responses" / name).read_text(encoding="utf-8"))


def test_bell_json_fixture_passes_contract_and_jobmanager_decoder() -> None:
    response = _fixture("bell-json.json")

    contracted = validate_json_response(response, {"statevector", "density_matrix"})
    parsed = JobManager()._parse_simulator_response(
        {"results": {"data": response["data"]}}
    )

    contracted_statevector = contracted["statevector"]
    contracted_density_matrix = contracted["density_matrix"]
    assert isinstance(contracted_statevector, list)
    assert isinstance(contracted_density_matrix, list)
    np.testing.assert_allclose(
        np.asarray(contracted_statevector), parsed["statevector"]
    )
    np.testing.assert_allclose(
        np.asarray(contracted_density_matrix), parsed["density_matrix"]
    )


def test_density_json_fixture_passes_contract_and_jobmanager_decoder() -> None:
    response = _fixture("density-json.json")

    contracted = validate_json_response(response, {"density_matrix"})
    parsed = JobManager()._parse_simulator_response(
        {"results": {"data": response["data"]}}
    )

    contracted_density_matrix = contracted["density_matrix"]
    assert isinstance(contracted_density_matrix, list)
    np.testing.assert_allclose(
        np.asarray(contracted_density_matrix), parsed["density_matrix"]
    )


def test_binary_contract_and_jobmanager_use_same_manifest_rules() -> None:
    statevector = np.asarray([1.0, 0.0], dtype="<c16")
    density_matrix = np.asarray([[1.0, 0.0], [0.0, 0.0]], dtype="<c16")
    body = statevector.tobytes() + density_matrix.tobytes()
    manifest = {
        "format": "qpiai_binary_artifacts",
        "schema_version": 1,
        "counts": {"0": 64},
        "execution_time": 0.125,
        "artifacts": [
            {
                "result_type": "statevector",
                "dtype": "complex128",
                "byte_order": "little",
                "shape": [2],
                "offset_bytes": 0,
                "size_bytes": statevector.nbytes,
            },
            {
                "result_type": "density_matrix",
                "dtype": "complex128",
                "byte_order": "little",
                "shape": [2, 2],
                "offset_bytes": statevector.nbytes,
                "size_bytes": density_matrix.nbytes,
            },
        ],
    }

    validate_binary_response(body, manifest, {"statevector", "density_matrix"})
    parsed = JobManager._decode_binary_artifact_payload(body, manifest)

    np.testing.assert_allclose(parsed["statevector"], statevector.reshape(-1, 1))
    np.testing.assert_allclose(parsed["density_matrix"], density_matrix)


def test_contract_and_jobmanager_reject_the_same_broken_binary_manifest() -> None:
    body = np.asarray([1.0, 0.0], dtype="<c16").tobytes()
    manifest = {
        "format": "qpiai_binary_artifacts",
        "schema_version": 1,
        "counts": None,
        "execution_time": 0.0,
        "artifacts": [
            {
                "result_type": "statevector",
                "dtype": "complex128",
                "byte_order": "little",
                "shape": [2],
                "offset_bytes": 16,
                "size_bytes": 32,
            }
        ],
    }

    with pytest.raises(ContractValidationError, match="contiguous"):
        validate_binary_response(body, manifest, {"statevector"})
    with pytest.raises(
        InvalidResponseError,
    ) as exc_info:
        JobManager._decode_binary_artifact_payload(body, manifest)

    assert isinstance(exc_info.value.__cause__, ContractValidationError)
