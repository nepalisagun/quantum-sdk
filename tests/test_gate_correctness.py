"""Regression checks for gate semantics, controls and non-unitary execution."""

import numpy as np
import pytest
import re
import json
from types import SimpleNamespace
from typing import Any

from qpiai_quantum.circuit import Circuit
from qpiai_quantum.circuit.exceptions import CircuitError
from qpiai_quantum.icr.circuitoperation import (
    CircuitOperation,
    OperationType,
    MeasureOperation,
    HGate,
    RXGate,
    XGate,
)
from qpiai_quantum.icr.operationmodifiers import CustomControlOperation
from qpiai_quantum.simulator.statevector import StatevectorSimulator


def simulate(
    circuit: Circuit, state: np.ndarray | None = None, shots: int = 1024
) -> Any:
    return StatevectorSimulator().run(circuit, initial_state=state, shots=shots, seed=7)


@pytest.mark.parametrize("basis", range(8))
def test_rccx_basis_mapping_and_phase(basis: int) -> None:
    circuit = Circuit(3)
    circuit.rccx(0, 1, 2)
    state = np.eye(8, dtype=complex)[:, basis]
    expected = state.copy()
    if basis == 3:
        expected = 1j * np.eye(8)[:, 7]
    elif basis == 7:
        expected = -1j * np.eye(8)[:, 3]
    elif basis == 5:
        expected = -state
    np.testing.assert_allclose(
        simulate(circuit, state).statevector, expected, atol=1e-12
    )


@pytest.mark.parametrize("basis", range(4))
def test_dcx_inverse_roundtrip(basis: int) -> None:
    circuit = Circuit(2)
    circuit.dcx(0, 1)
    circuit.compose(circuit.inverse())
    state = np.eye(4, dtype=complex)[:, basis]
    np.testing.assert_allclose(simulate(circuit, state).statevector, state, atol=1e-12)


def test_explicit_multi_controls_inverse_and_compose() -> None:
    original = Circuit(3)
    original.add_operation(CustomControlOperation(0, 1, operation=RXGate(2, 0.37)))
    circuit = Circuit(4)
    circuit.compose(original, qubits=[3, 1, 0])
    circuit.compose(original.inverse(), qubits=[3, 1, 0])
    state = np.arange(1, 17, dtype=complex) * np.exp(1j * np.arange(16))
    state /= np.linalg.norm(state)
    np.testing.assert_allclose(simulate(circuit, state).statevector, state, atol=1e-12)


def test_controlled_composite_keeps_unactivated_branch_unchanged() -> None:
    circuit = Circuit(2)
    body = Circuit(1)
    body.h(0)
    body.x(0)
    operation = body.to_circuit_operation("hx")
    circuit.add_operation(CustomControlOperation(1, operation=operation))
    np.testing.assert_allclose(simulate(circuit).statevector, [1, 0, 0, 0], atol=1e-12)


def test_mid_measurement_records_state_at_that_time() -> None:
    circuit = Circuit(1, 1)
    circuit.x(0)
    circuit.measure(0, 0)
    circuit.x(0)
    result = simulate(circuit)
    assert result.counts == {"1": 1024}
    np.testing.assert_allclose(result.statevector, [1, 0], atol=1e-12)


def test_measurement_removes_interference() -> None:
    circuit = Circuit(1, 1)
    circuit.h(0)
    circuit.measure(0, 0)
    circuit.h(0)
    result = simulate(circuit)
    assert result.statevector is None
    np.testing.assert_allclose(result.density_matrix, np.eye(2) / 2, atol=1e-12)
    assert result.job_metadata["state_kind"] == "mixed"
    assert set(result.counts) == {"0", "1"}
    serialized = json.loads(result.to_json())
    assert "statevector" not in serialized
    assert serialized["job_metadata"]["state_kind"] == "mixed"
    assert len(serialized["density_matrix"]) == 2


def test_reset_entangled_qubit_preserves_other_qubit_uncertainty() -> None:
    circuit = Circuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.reset(0)
    result = simulate(circuit)
    assert result.statevector is None
    # Basis order is |q1 q0>. Reset q0 leaves q1 equally likely 0 or 1.
    np.testing.assert_allclose(
        result.density_matrix, np.diag([0.5, 0, 0.5, 0]), atol=1e-12
    )
    assert np.isclose(np.trace(result.density_matrix), 1)


def test_reset_one_and_reset_superposition_return_pure_zero() -> None:
    for preparation in ("x", "h"):
        circuit = Circuit(1)
        getattr(circuit, preparation)(0)
        circuit.reset(0)
        result = simulate(circuit)
        np.testing.assert_allclose(result.statevector, [1, 0], atol=1e-12)
        assert result.density_matrix is None


def test_terminal_measurements_allow_fanout_and_classical_overwrite() -> None:
    circuit = Circuit(2, 3)
    circuit.x(0)
    circuit.measure(0, 0)
    circuit.measure(0, 1)
    circuit.measure(1, 0)
    result = simulate(circuit)
    assert result.counts == {"010": 1024}
    assert result.job_metadata["state_reference"] == "before_terminal_measurements"


@pytest.mark.parametrize("index", [True, 0.5, "0", -1])
def test_gate_rejects_invalid_indices(index: Any) -> None:
    with pytest.raises(ValueError):
        HGate(index)


@pytest.mark.parametrize("angle", [True, "0.3", float("nan"), float("inf"), 1j])
def test_gate_rejects_invalid_parameters(angle: Any) -> None:
    with pytest.raises(ValueError):
        RXGate(0, angle)


def test_controls_reject_duplicates_overlap_and_nonunitary_body() -> None:
    for controls in [(0, 0), (2,), ()]:
        with pytest.raises(ValueError):
            CustomControlOperation(*controls, operation=XGate(2))
    operation = CircuitOperation(OperationType.OPERATION, "measurement", [0])
    operation.order = [MeasureOperation(0, 0)]
    with pytest.raises(ValueError):
        CustomControlOperation(1, operation=operation)


def test_nonunitary_circuits_have_no_inverse() -> None:
    circuit = Circuit(1)
    circuit.reset(0)
    with pytest.raises(CircuitError, match="no unitary inverse"):
        circuit.inverse()


def test_qasm2_exports_defined_primitives_and_preserves_reset_barrier() -> None:
    circuit = Circuit(4)
    circuit.mcx([0, 1, 2], 3)
    circuit.reset(0)
    circuit.barrier(0, 1)
    qasm = circuit.to_qasm()
    assert "mcx " not in qasm
    assert "u3(" in qasm and "cx " in qasm
    assert "reset q[0];" in qasm and "barrier q[0], q[1];" in qasm


@pytest.mark.parametrize("controls", range(5))
def test_mcx_export_full_unitary_including_spare_qubit(controls: int) -> None:
    """Interpret the emitted primitive text independently, retaining all phases."""
    circuit = Circuit(controls + 2)
    circuit.mcx(list(range(controls)), controls)
    size = 2**circuit.num_qubits
    actual = np.eye(size, dtype=complex)
    for line in circuit.to_qasm().splitlines()[4:]:
        wires = [int(q) for q in re.findall(r"q\[(\d+)\]", line)]
        if line.startswith("cx "):
            permutation = [
                i ^ (1 << wires[1]) if (i >> wires[0]) & 1 else i for i in range(size)
            ]
            actual = actual[permutation]
        else:
            theta, phi, lam = map(float, line.split("(")[1].split(")")[0].split(","))
            matrix = np.array(
                [
                    [np.cos(theta / 2), -np.exp(1j * lam) * np.sin(theta / 2)],
                    [
                        np.exp(1j * phi) * np.sin(theta / 2),
                        np.exp(1j * (phi + lam)) * np.cos(theta / 2),
                    ],
                ]
            )
            for zero in range(size):
                if (zero >> wires[0]) & 1:
                    continue
                one = zero | (1 << wires[0])
                actual[[zero, one]] = matrix @ actual[[zero, one]]
    mask = (1 << controls) - 1
    permutation = [i ^ (1 << controls) if i & mask == mask else i for i in range(size)]
    expected = np.eye(size)[permutation]
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_dynamic_gate_api_rejects_out_of_range_qubit() -> None:
    circuit = Circuit(1)
    with pytest.raises(CircuitError, match="out of range"):
        getattr(circuit, "H")(1)


@pytest.mark.parametrize("available", [150, 300])
def test_mixed_execution_checks_branch_and_density_memory(
    monkeypatch: Any, available: int
) -> None:
    import psutil

    monkeypatch.setattr(
        psutil, "virtual_memory", lambda: SimpleNamespace(available=available)
    )
    circuit = Circuit(2, 1)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure(0, 0)
    circuit.h(0)
    message = "branches" if available == 150 else "density matrix"
    with pytest.raises(MemoryError, match=message):
        simulate(circuit)
