"""Quantum operations with explicit controls and probability-weighted branches."""

from numbers import Integral
from collections.abc import Callable, Iterable, Iterator

import numpy as np
import psutil

from ..icr.validation import validate_index
from ..icr.circuitoperation import CircuitOperation
from .gates import DECOMPOSED_GATES, X, decompose, gate_spec


def flatten(operations: Iterable[CircuitOperation]) -> Iterator[CircuitOperation]:
    for operation in operations:
        if (
            getattr(operation, "base_operation", None) is None
            and getattr(operation, "order", None) is not None
        ):
            yield from flatten(operation.order or [])
        else:
            yield operation


def apply_operation(
    state: np.ndarray,
    n: int,
    operation: CircuitOperation,
    apply_unitary: Callable,
    check_budget: Callable[[], None],
    controls: tuple[int, ...] = (),
) -> np.ndarray:
    check_budget()
    for q in [*controls, *operation.qubits]:
        validate_index(q)
        if q >= n:
            raise ValueError("Operation qubit index is outside the circuit")
    base = getattr(operation, "base_operation", None)
    if base is not None:
        return apply_operation(
            state,
            n,
            base,
            apply_unitary,
            check_budget,
            (*controls, *operation.control_qubits),
        )
    if getattr(operation, "order", None) is not None:
        for sub in operation.order or []:
            state = apply_operation(
                state, n, sub, apply_unitary, check_budget, controls
            )
        return state
    name = operation.gate_name.lower()
    if name == "barrier":
        return state
    if name in {"measure", "reset"}:
        raise ValueError("Non-unitary operations cannot be applied as controlled gates")
    if name in DECOMPOSED_GATES:
        from ..icr.circuitoperation import CircuitOperation, OperationType

        for sub_name, params, qubits in decompose(name, operation.qubits):
            item = CircuitOperation(
                OperationType.N_QUBIT_PARAMETRIC, sub_name, qubits, params
            )
            state = apply_operation(
                state, n, item, apply_unitary, check_budget, controls
            )
        return state
    qubits = operation.qubits
    if name == "mcx":
        controls = (*controls, *qubits[:-1])
        qubits = qubits[-1:]
        matrix = X
    else:
        _, matrix = gate_spec(name, operation.params or [], num_qubits=len(qubits))
    if matrix.shape != (2 ** len(qubits), 2 ** len(qubits)):
        raise ValueError("Gate qubit count does not match its matrix dimension")
    if not controls:
        return apply_unitary(state, n, qubits, matrix)
    if len(set([*controls, *qubits])) != len(controls) + len(qubits):
        raise ValueError("Controls and operation qubits must be distinct")
    axes = [n - 1 - q for q in [*controls, *qubits]]
    tensor = np.moveaxis(state.reshape([2] * n), axes, range(len(axes)))
    shape = tensor.shape
    blocks = tensor.reshape(2 ** len(controls), 2 ** len(qubits), -1).copy()
    blocks[-1] = matrix @ blocks[-1]
    return np.moveaxis(blocks.reshape(shape), range(len(axes)), axes).reshape(-1)


def project(
    state: np.ndarray, qubit: int, outcome: int
) -> tuple[float, np.ndarray | None]:
    mask = ((np.arange(len(state)) >> qubit) & 1) == outcome
    probability = float(np.sum(np.abs(state[mask]) ** 2))
    if probability <= 0:
        return 0.0, None
    collapsed = np.where(mask, state, 0.0) / np.sqrt(probability)
    return probability, collapsed


def execute(
    operations: Iterable[CircuitOperation],
    state: np.ndarray,
    n: int,
    n_cbits: int,
    shots: int,
    seed: int | None,
    apply_unitary: Callable,
    check_budget: Callable[[], None],
) -> tuple[list | None, list | None, dict[str, int], dict]:
    if isinstance(shots, bool) or not isinstance(shots, Integral) or shots <= 0:
        raise ValueError("shots must be a positive integer")
    operations = list(flatten(operations))
    for op in operations:
        for q in op.qubits:
            if validate_index(q) >= n:
                raise ValueError("Operation qubit index is outside the circuit")
        for c in op.clbits or []:
            if validate_index(c) >= n_cbits:
                raise ValueError(
                    "Measurement classical bit index is outside the circuit"
                )
    cut = len(operations)
    while cut and operations[cut - 1].gate_name.lower() in {"measure", "barrier"}:
        cut -= 1
    prefix, terminal = operations[:cut], operations[cut:]
    branches = [(1.0, state, [0] * n_cbits)]
    for op in prefix:
        check_budget()
        if op.gate_name.lower() not in {"measure", "reset"}:
            branches = [
                (
                    weight,
                    apply_operation(vector, n, op, apply_unitary, check_budget),
                    classical,
                )
                for weight, vector, classical in branches
            ]
            continue
        q = op.qubits[0]
        if len(branches) * len(state) * 16 * 4 > psutil.virtual_memory().available:
            raise MemoryError(
                "Insufficient RAM for measurement/reset statevector branches"
            )
        next_branches = []
        for weight, vector, classical in branches:
            check_budget()
            for outcome in (0, 1):
                probability, collapsed = project(vector, q, outcome)
                if collapsed is None:
                    continue
                bits = classical.copy()
                if op.gate_name.lower() == "measure":
                    bits[op.clbits[0]] = outcome
                elif outcome:
                    collapsed = apply_unitary(collapsed, n, [q], X)
                next_branches.append((weight * probability, collapsed, bits))
        branches = next_branches

    weights = np.asarray([b[0] for b in branches], dtype=float)
    weights /= weights.sum()
    rng = np.random.default_rng(seed)
    counts = {}
    has_measurements = any(op.gate_name.lower() == "measure" for op in operations)
    if has_measurements:
        chosen = (
            rng.choice(len(branches), size=shots, p=weights)
            if len(branches) > 1
            else np.zeros(shots, dtype=int)
        )
        for index, (_, vector, classical) in enumerate(branches):
            check_budget()
            size = int(np.count_nonzero(chosen == index))
            probabilities = np.abs(vector) ** 2
            outcomes = rng.choice(
                len(vector), size=size, p=probabilities / probabilities.sum()
            )
            for outcome, frequency in zip(*np.unique(outcomes, return_counts=True)):
                check_budget()
                bits = classical.copy()
                for op in terminal:
                    if op.gate_name.lower() == "measure":
                        bits[op.clbits[0]] = (int(outcome) >> op.qubits[0]) & 1
                key = "".join(str(b) for b in reversed(bits))
                counts[key] = counts.get(key, 0) + int(frequency)

    vector = branches[0][1]
    pure = all(
        np.isclose(abs(np.vdot(vector, branch[1])), 1.0, rtol=0, atol=1e-12)
        for branch in branches
    )
    density = None
    check_budget()
    if not pure:
        dim = len(vector)
        if dim * dim * 128 > psutil.virtual_memory().available:
            raise MemoryError(
                "Insufficient RAM to represent the mixed result density matrix"
            )
        density = np.zeros((dim, dim), dtype=complex)
        for weight, (_, branch, _) in zip(weights, branches):
            check_budget()
            density += weight * np.outer(branch, branch.conj())
        density = density.tolist()
    metadata = {
        "state_reference": "before_terminal_measurements"
        if any(op.gate_name.lower() == "measure" for op in terminal)
        else "end_of_circuit",
        "state_kind": "pure" if pure else "mixed",
        "branch_count": len(branches),
    }
    statevector = vector.tolist() if pure else None
    check_budget()
    return statevector, density, counts, metadata
