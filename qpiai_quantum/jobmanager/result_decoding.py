"""Decode JSON-safe quantum result artifacts into typed numeric values."""

from .exceptions import InvalidResponseError
from .result_contract import (
    ContractValidationError,
    DensityMatrixData,
    StatevectorData,
    decode_density_matrix_contract,
    decode_statevector_contract,
)


def decode_statevector(statevector: object) -> StatevectorData | None:
    """Decode a statevector and return the SDK's canonical column-vector shape."""
    try:
        return decode_statevector_contract(statevector)
    except ContractValidationError as exc:
        raise InvalidResponseError(str(exc)) from exc


def decode_density_matrix(density_matrix: object) -> DensityMatrixData | None:
    """Decode JSON complex entries in a density matrix."""
    try:
        return decode_density_matrix_contract(density_matrix)
    except ContractValidationError as exc:
        raise InvalidResponseError(str(exc)) from exc
