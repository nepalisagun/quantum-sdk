"""Shared exact definitions for gates expressed as smaller operations.

Definitions preserve the full unitary, including scalar phase. Simulation,
inversion and export must use these definitions rather than duplicate recipes.
Native matrix implementations may coexist, provided they agree exactly.
"""

from collections.abc import Callable
from math import pi

from .circuitoperation import CircuitOperation, CXGate, PGate, U2Gate


def _dcx(qubits: list[int]) -> list[CircuitOperation]:
    a, b = qubits
    return [CXGate(a, b), CXGate(b, a)]


def _rccx(qubits: list[int]) -> list[CircuitOperation]:
    a, b, c = qubits
    return [
        U2Gate(c, 0.0, pi),
        PGate(c, pi / 4),
        CXGate(b, c),
        PGate(c, -pi / 4),
        CXGate(a, c),
        PGate(c, pi / 4),
        CXGate(b, c),
        PGate(c, -pi / 4),
        U2Gate(c, 0.0, pi),
    ]


_DEFINITIONS: dict[str, Callable[[list[int]], list[CircuitOperation]]] = {
    "dcx": _dcx,
    "rccx": _rccx,
}
DECOMPOSED_GATES = frozenset(_DEFINITIONS)


def gate_decomposition(
    name: str, qubits: list[int], params: list[float] | None = None
) -> list[CircuitOperation] | None:
    """Return fresh child operations, or None when no shared definition exists.

    An empty list is a known identity body, distinct from an unknown definition.
    Current named definitions are nonparametric. Reject parameters rather than
    silently discarding them. Future recipes must preserve exact phase and must
    terminate in native gates without recursive definition cycles.
    """
    definition = _DEFINITIONS.get(name.lower())
    if definition is None:
        return None
    if params:
        raise ValueError(f"Gate {name} does not accept parameters")
    return definition(qubits)
