"""
Statevector Simulator
=====================
A self-contained statevector simulator that executes QpiAI Quantum
circuits locally using NumPy and Numba multi-core acceleration. It implements the
BaseSimulator interface and directly consumes IntermediateCircuitRepresentation
objects without needing QASM compilation.
"""

import os
import time
from typing import Any, Optional, TYPE_CHECKING

import numpy as np

from qpiai_quantum.icr.circuitoperation import OperationType

if TYPE_CHECKING:
    from qpiai_quantum.circuit import Circuit

from .base_simulator import BaseSimulator
from .gates import DECOMPOSED_GATES, decompose, gate_spec
from .result import QasmSimulatorResult

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import numba
    from numba import njit

    if TYPE_CHECKING:
        from builtins import range as prange
    else:
        from numba import prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


if NUMBA_AVAILABLE:

    @njit(parallel=True, fastmath=True)
    def _apply_1q_numba(state: np.ndarray, u: np.ndarray, q: int) -> None:
        """Parallel Numba kernel for 1-qubit gate application."""
        n = int(np.log2(len(state)))
        half_dim = 1 << (n - 1)
        step = 1 << q
        u00, u01 = u[0, 0], u[0, 1]
        u10, u11 = u[1, 0], u[1, 1]

        for i in prange(half_dim):
            idx0 = ((i >> q) << (q + 1)) | (i & (step - 1))
            idx1 = idx0 | step
            v0 = state[idx0]
            v1 = state[idx1]
            state[idx0] = u00 * v0 + u01 * v1
            state[idx1] = u10 * v0 + u11 * v1

    @njit(parallel=True, fastmath=True)
    def _apply_2q_numba(state: np.ndarray, u: np.ndarray, q0: int, q1: int) -> None:
        """Parallel Numba kernel for 2-qubit gate application."""
        n = int(np.log2(len(state)))
        quarter_dim = 1 << (n - 2)

        if q0 < q1:
            q_min, q_max = q0, q1
        else:
            q_min, q_max = q1, q0

        mask_low = (1 << q_min) - 1
        mask_mid = (1 << (q_max - 1)) - 1

        u00, u01, u02, u03 = u[0, 0], u[0, 1], u[0, 2], u[0, 3]
        u10, u11, u12, u13 = u[1, 0], u[1, 1], u[1, 2], u[1, 3]
        u20, u21, u22, u23 = u[2, 0], u[2, 1], u[2, 2], u[2, 3]
        u30, u31, u32, u33 = u[3, 0], u[3, 1], u[3, 2], u[3, 3]

        for i in prange(quarter_dim):
            low = i & mask_low
            mid = ((i & mask_mid) >> q_min) << (q_min + 1)
            high = (i >> (q_max - 1)) << (q_max + 1)
            base = high | mid | low

            b00 = base
            b01 = base | (1 << q1)
            b10 = base | (1 << q0)
            b11 = base | (1 << q0) | (1 << q1)

            v00 = state[b00]
            v01 = state[b01]
            v10 = state[b10]
            v11 = state[b11]

            state[b00] = u00 * v00 + u01 * v01 + u02 * v10 + u03 * v11
            state[b01] = u10 * v00 + u11 * v01 + u12 * v10 + u13 * v11
            state[b10] = u20 * v00 + u21 * v01 + u22 * v10 + u23 * v11
            state[b11] = u30 * v00 + u31 * v01 + u32 * v10 + u33 * v11


class StatevectorSimulator(BaseSimulator):
    """
    Local statevector simulator for QpiAI Quantum circuits.
    Executes directly from the ICR (Intermediate Circuit Representation).
    """

    @property
    def name(self) -> str:
        return "QpiAI-QSV-Local"

    def run(
        self,
        circuit: "Circuit",
        shots: int = 1024,
        seed: int | None = None,
        name: str | None = None,
        initial_state: np.ndarray | None = None,
    ) -> QasmSimulatorResult:
        """
        Execute the given circuit locally using statevector simulation.

        Args:
            circuit: The quantum circuit to execute.
            shots: Number of measurement shots to perform.
            seed: Optional RNG seed for reproducibility.
            name: Optional name for the result object.
            initial_state: Optional initial statevector for the simulation.

        Returns:
            A result with counts and a statevector when pure, or a density
            matrix when intermediate measurement/reset produces a mixture.
            Terminal measurements preserve the state immediately before them.
        """
        start_time = time.perf_counter()

        n_qubits = circuit.num_qubits
        n_cbits = circuit.num_clbits

        if n_qubits == 0:
            raise ValueError("Cannot simulate a circuit with 0 qubits.")

        if PSUTIL_AVAILABLE:
            req_ram_bytes = (2**n_qubits) * 16 * 2.0
            avail_ram_bytes = psutil.virtual_memory().available
            if req_ram_bytes > avail_ram_bytes:
                req_mb = req_ram_bytes / (1024**2)
                avail_mb = avail_ram_bytes / (1024**2)
                raise MemoryError(
                    f"Insufficient available RAM to simulate {n_qubits} qubits locally. "
                    f"Required: ~{req_mb:.1f} MB, Available: {avail_mb:.1f} MB. "
                    f"Please submit to QpiAI QCloud Backend Compute."
                )

        dim = 2**n_qubits
        if initial_state is not None:
            initial_state = np.asarray(initial_state, dtype=complex)
            if initial_state.ndim != 1 or not np.all(np.isfinite(initial_state)):
                raise ValueError(
                    "Initial state must be a finite one-dimensional vector"
                )
            if len(initial_state) != dim:
                raise ValueError(
                    f"Initial state dimension {len(initial_state)} does not match circuit qubit dimension {dim}."
                )
            if not np.isclose(np.linalg.norm(initial_state), 1.0):
                raise ValueError(
                    "Initial state vector must be normalized (norm must be 1.0)."
                )
            state = np.array(initial_state, dtype=complex)
        else:
            # Initialize state to |0...0>
            state = np.zeros(dim, dtype=complex)
            state[0] = 1.0

        from .operation_execution import execute

        statevector, density_matrix, counts, metadata = execute(
            circuit.icr.evolve,
            state,
            n_qubits,
            n_cbits,
            shots,
            seed,
            self._apply_unitary,
            lambda: None,
        )

        elapsed_time = time.perf_counter() - start_time

        return QasmSimulatorResult(
            name=name or circuit.name,
            counts=counts,
            statevector=statevector,
            density_matrix=density_matrix,
            job_metadata=metadata,
            shots=shots,
            execution_time=elapsed_time,
            method="statevector",
            job_status="completed",
            n_qubits=n_qubits,
            n_cbits=n_cbits,
        )

    # ----------------------------------------------------------------------
    # Internal math utilities
    # ----------------------------------------------------------------------

    @staticmethod
    def _apply_unitary(
        state: np.ndarray, n: int, qubits: list[int], U: np.ndarray
    ) -> np.ndarray:
        """Apply unitary U to specified qubits within statevector."""
        k = len(qubits)
        if NUMBA_AVAILABLE:
            if k == 1:
                _apply_1q_numba(state, U, qubits[0])
                return state
            elif k == 2:
                _apply_2q_numba(state, U, qubits[0], qubits[1])
                return state

        return StatevectorSimulator._apply_unitary_numpy(state, n, qubits, U)

    @staticmethod
    def _apply_unitary_numpy(
        state: np.ndarray, n: int, qubits: list[int], U: np.ndarray
    ) -> np.ndarray:
        """Fallback NumPy implementation for gate application."""
        k = len(qubits)
        tensor = state.reshape([2] * n)
        axes = [int(n - 1 - q) for q in qubits]
        tensor = np.moveaxis(tensor, axes, list(range(k)))
        shape = tensor.shape
        flat = tensor.reshape(2**k, -1)
        flat = U @ flat
        tensor = flat.reshape(shape)
        tensor = np.moveaxis(tensor, list(range(k)), axes)
        return tensor.reshape(-1)

    @staticmethod
    def _apply_reset(state: np.ndarray, n: int, qubit: int) -> np.ndarray:
        """Sample one reset trajectory; ensemble execution retains both outcomes."""
        from .operation_execution import project

        rng = np.random.default_rng()
        p0, _ = project(state, qubit, 0)
        outcome = int(rng.random() >= p0)
        _, collapsed = project(state, qubit, outcome)
        if collapsed is None:
            raise ValueError("Reset input must have nonzero norm")
        if outcome:
            from .gates import X

            collapsed = StatevectorSimulator._apply_unitary(collapsed, n, [qubit], X)
        return collapsed

    @staticmethod
    def _sample_counts(
        state: np.ndarray,
        n_qubits: int,
        n_cbits: int,
        measure_map: dict[int, int],
        shots: int,
        seed: int | None = None,
    ) -> dict[str, int]:
        """Sample measurement outcomes."""
        probs = np.abs(state) ** 2
        probs = probs / probs.sum()  # renormalise against fp drift
        rng = np.random.default_rng(seed)
        outcomes = rng.choice(len(probs), size=shots, p=probs)

        counts: dict[str, int] = {}
        for outcome in outcomes:
            cbits = ["0"] * n_cbits
            for qubit, cbit in measure_map.items():
                bit = (outcome >> qubit) & 1
                cbits[cbit] = str(bit)
            # Qiskit-style left-to-right ordering: highest cbit index first
            bitstring = "".join(reversed(cbits)) if n_cbits > 0 else ""
            counts[bitstring] = counts.get(bitstring, 0) + 1

        return counts
