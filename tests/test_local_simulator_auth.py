from unittest.mock import patch

import numpy as np
import pytest

from qpiai_quantum import Circuit, DensityMatrix, Statevector


def test_local_backend_rejects_unauthenticated_execution() -> None:
    circuit = Circuit(1)

    with patch(
        "qpiai_quantum.authentication.user.get_user",
        return_value=None,
    ):
        with pytest.raises(ValueError, match="Authentication required"):
            circuit.run(device_name="QpiAI-QSV-Local")


def test_statevector_from_circuit_uses_active_login(
    authenticated_sdk_user: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("API_KEY", raising=False)
    circuit = Circuit(1)
    circuit.h(0)

    state = Statevector(circuit)

    np.testing.assert_allclose(state.data, [1 / np.sqrt(2), 1 / np.sqrt(2)])


def test_density_matrix_from_circuit_uses_active_login(
    authenticated_sdk_user: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("API_KEY", raising=False)
    circuit = Circuit(1)
    circuit.x(0)

    state = DensityMatrix(circuit)

    np.testing.assert_allclose(state.data, [[0, 0], [0, 1]])
