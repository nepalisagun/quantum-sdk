"""Check shared gate bodies against independent complex-unitary references."""

import numpy as np
import pytest

from qpiai_quantum.circuit import Circuit
from qpiai_quantum.icr.circuitoperation import (
    CircuitOperation,
    OperationType,
    CXGate,
    DCXGate,
    RCCXGate,
    ResetOperation,
)
from qpiai_quantum.icr.operationmodifiers import CustomControlOperation
from qpiai_quantum.icr import decomposition
from qpiai_quantum.iem.qasm.primitives import lower_operation
from qpiai_quantum.simulator.statevector import StatevectorSimulator
from qpiai_quantum.simulator.gates import decompose


def unitary(circuit: Circuit) -> np.ndarray:
    """Run every basis vector below the authentication boundary."""
    basis = np.eye(2**circuit.num_qubits, dtype=complex)
    simulator = StatevectorSimulator()
    columns: list[np.ndarray] = []
    for state in basis:
        vector = simulator.run(circuit, initial_state=state).statevector
        assert vector is not None
        columns.append(np.asarray(vector, dtype=complex))
    return np.column_stack(columns)


def reference(name: str, wires: list[int], control: int | None) -> np.ndarray:
    """Independent basis mapping with the required RCCX phases."""
    n = len(wires) + (control is not None)
    result = np.zeros((2**n, 2**n), dtype=complex)
    for source in range(2**n):
        target, phase = source, 1.0 + 0j
        if control is None or (source >> control) & 1:
            a, b = wires[:2]
            if name == "dcx":
                if (target >> a) & 1:
                    target ^= 1 << b
                if (target >> b) & 1:
                    target ^= 1 << a
            else:
                c = wires[2]
                local = sum(((source >> q) & 1) << i for i, q in enumerate(wires))
                if local in (3, 7):
                    target ^= 1 << c
                    phase = 1j if local == 3 else -1j
                elif local == 5:
                    phase = -1
        result[target, source] = phase
    return result


@pytest.mark.parametrize("name", ["dcx", "rccx"])
@pytest.mark.parametrize("controlled", [False, True])
@pytest.mark.parametrize("permuted", [False, True])
def test_shared_body_simulation_inverse_and_export(
    name: str, controlled: bool, permuted: bool
) -> None:
    size = 2 if name == "dcx" else 3
    n = size + controlled
    wires = list(reversed(range(n))) if permuted else list(range(n))
    control = wires.pop() if controlled else None
    operation = DCXGate(*wires) if name == "dcx" else RCCXGate(*wires)
    if control is not None:
        operation = CustomControlOperation(control, operation=operation)
    circuit = Circuit(n)
    circuit.add_operation(operation)
    expected = reference(name, wires, control)
    np.testing.assert_allclose(unitary(circuit), expected, atol=1e-12)
    np.testing.assert_allclose(
        unitary(circuit.inverse()), expected.conj().T, atol=1e-12
    )
    exported = Circuit(n)
    for gate, params, qubits in lower_operation(operation):
        exported.add_operation(
            CircuitOperation(OperationType.N_QUBIT_PARAMETRIC, gate, qubits, params)
        )
    np.testing.assert_allclose(unitary(exported), expected, atol=1e-12)


def test_new_definition_needs_no_consumer_special_cases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        decomposition._DEFINITIONS,
        "test_pair",
        lambda q: [CXGate(q[0], q[1]), CXGate(q[1], q[0])],
    )
    circuit = Circuit(2)
    operation = CircuitOperation(OperationType.OPERATION, "test_pair", [0, 1])
    circuit.add_operation(operation)
    expected = reference("dcx", [0, 1], None)
    np.testing.assert_allclose(unitary(circuit), expected, atol=1e-12)
    np.testing.assert_allclose(
        unitary(circuit.inverse()), expected.conj().T, atol=1e-12
    )
    assert lower_operation(operation) == [("cx", [], [0, 1]), ("cx", [], [1, 0])]


def test_bodies_are_independent_and_opaque_operations_remain_opaque() -> None:
    operation = DCXGate(2, 0)
    body = operation.decompose()
    assert body is not None
    body[0].qubits[0] = 99
    fresh_body = operation.decompose()
    assert fresh_body is not None
    assert fresh_body[0].qubits == [2, 0]
    assert ResetOperation(0).decompose() is None
    assert CXGate(0, 1).decompose() is None
    assert decompose("DCX", [2, 0]) == [("cx", [], [2, 0]), ("cx", [], [0, 2])]
    with pytest.raises(ValueError, match="does not accept parameters"):
        decomposition.gate_decomposition("dcx", [0, 1], [0.2])
