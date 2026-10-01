"""Dependency-free validation for simulator result transport contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypeAlias, TypeGuard, TypedDict


class ContractValidationError(ValueError):
    """Raised when a simulator response violates the SDK wire contract."""


class JSONComplex(TypedDict):
    """JSON representation of one complex number."""

    real: float
    imag: float


StatevectorData: TypeAlias = list[list[complex]]
DensityMatrixData: TypeAlias = list[list[complex]]


@dataclass(frozen=True)
class BinaryArtifactDescriptor:
    """Validated location and shape of one binary result artifact."""

    result_type: str
    shape: tuple[int, ...]
    offset_bytes: int
    size_bytes: int


def _is_json_complex(value: object) -> TypeGuard[JSONComplex]:
    if not isinstance(value, dict):
        return False
    real = value.get("real")
    imag = value.get("imag")
    return (
        isinstance(real, int | float)
        and not isinstance(real, bool)
        and math.isfinite(float(real))
        and isinstance(imag, int | float)
        and not isinstance(imag, bool)
        and math.isfinite(float(imag))
    )


def _decode_complex_scalar(value: object, artifact_name: str) -> complex:
    if _is_json_complex(value):
        return complex(value["real"], value["imag"])
    if isinstance(value, bool) or not isinstance(value, int | float | complex):
        raise ContractValidationError(
            f"{artifact_name} contains an invalid complex amplitude"
        )
    decoded = complex(value)
    if not math.isfinite(decoded.real) or not math.isfinite(decoded.imag):
        raise ContractValidationError(
            f"{artifact_name} contains a non-finite complex amplitude"
        )
    return decoded


def decode_statevector_contract(statevector: object) -> StatevectorData | None:
    """Decode a statevector into the SDK's canonical column-vector shape."""
    if statevector is None:
        return None
    if not isinstance(statevector, list):
        raise ContractValidationError("Statevector must be a list of amplitudes")

    decoded: StatevectorData = []
    for amplitude in statevector:
        if isinstance(amplitude, list):
            if len(amplitude) != 1:
                raise ContractValidationError(
                    "Statevector column entries must contain one amplitude"
                )
            amplitude = amplitude[0]
        decoded.append([_decode_complex_scalar(amplitude, "Statevector")])
    return decoded


def decode_density_matrix_contract(
    density_matrix: object,
) -> DensityMatrixData | None:
    """Decode and validate a square density matrix."""
    if density_matrix is None:
        return None
    if not isinstance(density_matrix, list):
        raise ContractValidationError("Density matrix must be a list of rows")

    decoded: DensityMatrixData = []
    expected_width: int | None = None
    for row in density_matrix:
        if not isinstance(row, list):
            raise ContractValidationError("Density matrix rows must be lists")
        if expected_width is None:
            expected_width = len(row)
        elif len(row) != expected_width:
            raise ContractValidationError("Density matrix rows must have equal lengths")
        decoded.append(
            [_decode_complex_scalar(value, "Density matrix") for value in row]
        )

    if decoded and len(decoded) != expected_width:
        raise ContractValidationError("Density matrix must be square")
    return decoded


def _validate_counts(counts: object, label: str) -> dict[str, int] | None:
    if counts is None:
        return None
    if not isinstance(counts, dict) or any(not isinstance(key, str) for key in counts):
        raise ContractValidationError(f"{label} counts must be an object or null")
    validated: dict[str, int] = {}
    for key, count in counts.items():
        if (
            isinstance(count, bool)
            or not isinstance(count, int | float)
            or not math.isfinite(float(count))
            or count < 0
            or not float(count).is_integer()
        ):
            raise ContractValidationError(
                f"{label} counts must contain nonnegative integers"
            )
        validated[key] = int(count)
    return validated


def extract_json_result_data(response: object) -> dict[str, object]:
    """Extract and validate the data object from a direct or worker response."""
    if not isinstance(response, dict):
        raise ContractValidationError("JSON simulator response must be an object")

    data = response.get("data")
    if not isinstance(data, dict):
        results = response.get("results")
        if isinstance(results, dict):
            data = results.get("data")
    if not isinstance(data, dict):
        raise ContractValidationError(
            "JSON simulator response must contain a data object"
        )
    return data


def validate_json_response(
    response: object, required_artifacts: set[str]
) -> dict[str, object]:
    """Validate JSON artifacts and return their canonical SDK values."""
    unsupported = required_artifacts - {"statevector", "density_matrix"}
    if unsupported:
        raise ContractValidationError(
            f"Unsupported required JSON artifacts: {sorted(unsupported)}"
        )

    data = extract_json_result_data(response)
    decoded: dict[str, object] = {
        "counts": _validate_counts(data.get("counts"), "JSON")
    }
    if "statevector" in data:
        decoded["statevector"] = decode_statevector_contract(data["statevector"])
    if "density_matrix" in data:
        decoded["density_matrix"] = decode_density_matrix_contract(
            data["density_matrix"]
        )
    missing = required_artifacts - {
        name
        for name in ("statevector", "density_matrix")
        if decoded.get(name) is not None
    }
    if missing:
        raise ContractValidationError(
            f"JSON simulator response is missing required artifacts: {sorted(missing)}"
        )
    return decoded


def validate_binary_manifest(
    raw_size: int, manifest: object
) -> tuple[dict[str, int] | None, float, tuple[BinaryArtifactDescriptor, ...]]:
    """Validate the versioned binary manifest against its payload size."""
    if not isinstance(manifest, dict):
        raise ContractValidationError("Binary result manifest must be an object")
    if manifest.get("format") != "qpiai_binary_artifacts":
        raise ContractValidationError("Unsupported binary result format")
    schema_version = manifest.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != 1
    ):
        raise ContractValidationError("Unsupported binary result schema version")
    if "counts" not in manifest or "execution_time" not in manifest:
        raise ContractValidationError(
            "Binary result manifest is missing counts or execution_time"
        )

    counts = _validate_counts(manifest["counts"], "Binary result")
    execution_time = manifest["execution_time"]
    if isinstance(execution_time, bool) or not isinstance(execution_time, int | float):
        raise ContractValidationError("Binary result execution_time must be numeric")
    execution_time = float(execution_time)
    if not math.isfinite(execution_time) or execution_time < 0:
        raise ContractValidationError(
            "Binary result execution_time must be finite and nonnegative"
        )

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ContractValidationError(
            "Binary result manifest must contain at least one artifact"
        )

    descriptors: list[BinaryArtifactDescriptor] = []
    seen_types: set[str] = set()
    expected_offset = 0
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict):
            raise ContractValidationError(
                f"Binary artifact {index} descriptor must be an object"
            )
        result_type = artifact.get("result_type")
        if result_type not in {"statevector", "density_matrix"}:
            raise ContractValidationError(
                f"Binary artifact {index} has an unsupported result_type"
            )
        if result_type in seen_types:
            raise ContractValidationError(
                f"Binary result contains duplicate {result_type} artifacts"
            )
        seen_types.add(result_type)
        if artifact.get("dtype") != "complex128":
            raise ContractValidationError(
                f"Binary artifact {index} must use complex128 dtype"
            )
        if artifact.get("byte_order") != "little":
            raise ContractValidationError(
                f"Binary artifact {index} must use little byte order"
            )

        shape = artifact.get("shape")
        if (
            not isinstance(shape, list)
            or not shape
            or any(
                isinstance(dimension, bool)
                or not isinstance(dimension, int)
                or dimension <= 0
                for dimension in shape
            )
        ):
            raise ContractValidationError(
                f"Binary artifact {index} has an invalid shape"
            )
        if result_type == "statevector" and len(shape) != 1:
            raise ContractValidationError("Statevector artifact shape must be [N]")
        if result_type == "density_matrix" and (
            len(shape) != 2 or shape[0] != shape[1]
        ):
            raise ContractValidationError(
                "Density-matrix artifact shape must be square [N, N]"
            )

        offset_bytes = artifact.get("offset_bytes")
        size_bytes = artifact.get("size_bytes")
        if (
            isinstance(offset_bytes, bool)
            or not isinstance(offset_bytes, int)
            or offset_bytes < 0
        ):
            raise ContractValidationError(
                f"Binary artifact {index} has an invalid offset_bytes"
            )
        if (
            isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes <= 0
        ):
            raise ContractValidationError(
                f"Binary artifact {index} has an invalid size_bytes"
            )
        if offset_bytes != expected_offset:
            raise ContractValidationError(
                "Binary artifact ranges must be contiguous and start at byte zero"
            )

        element_count = math.prod(shape)
        if size_bytes != element_count * 16:
            raise ContractValidationError(
                f"Binary artifact {index} size does not match its shape"
            )
        end_offset = offset_bytes + size_bytes
        if end_offset > raw_size:
            raise ContractValidationError(
                f"Binary artifact {index} exceeds the downloaded payload"
            )
        descriptors.append(
            BinaryArtifactDescriptor(
                result_type=result_type,
                shape=tuple(shape),
                offset_bytes=offset_bytes,
                size_bytes=size_bytes,
            )
        )
        expected_offset = end_offset

    if expected_offset != raw_size:
        raise ContractValidationError(
            "Binary artifact ranges do not cover the downloaded payload"
        )
    return counts, execution_time, tuple(descriptors)


def validate_binary_response(
    raw_bytes: bytes, manifest: object, required_artifacts: set[str]
) -> tuple[BinaryArtifactDescriptor, ...]:
    """Validate a complete binary response and its required artifacts."""
    _, _, descriptors = validate_binary_manifest(len(raw_bytes), manifest)
    available = {descriptor.result_type for descriptor in descriptors}
    missing = required_artifacts - available
    if missing:
        raise ContractValidationError(
            f"Binary simulator response is missing required artifacts: {sorted(missing)}"
        )
    return descriptors
