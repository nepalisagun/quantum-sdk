"""Exact, ancilla-free lowering to the portable U3/CX gate basis.

Every emitted phase is retained, including phases that become relative phases
when the operation is controlled. No runtime compiler dependency is required.
"""

from itertools import combinations
from math import pi
from typing import Any

from ...icr.validation import validate_parameter

Primitive = tuple[str, list[float], list[int]]


def lower_operation(operation: Any) -> list[Primitive]:
    op = operation if isinstance(operation, dict) else operation.to_json()
    result: list[Primitive] = []

    def phase(wires: list[int], angle: float) -> None:
        """Apply exp(i*angle) only to the all-one state, by parity expansion."""
        if not wires or angle == 0:
            return
        # This exact construction is exponential; reject insufficient RAM before
        # building a large instruction list, rather than assuming an ancilla.
        import psutil

        estimate = (2 ** len(wires)) * len(wires) * 256
        if estimate > psutil.virtual_memory().available:
            raise MemoryError("Insufficient RAM for exact ancilla-free gate export")
        divisor = 2 ** (len(wires) - 1)
        for size in range(1, len(wires) + 1):
            value = angle * (-1 if size % 2 == 0 else 1) / divisor
            for subset in combinations(wires, size):
                pivot = subset[-1]
                for q in subset[:-1]:
                    result.append(("cx", [], [q, pivot]))
                result.append(("u3", [0.0, 0.0, value], [pivot]))
                for q in reversed(subset[:-1]):
                    result.append(("cx", [], [q, pivot]))

    def rotation_z(ctrl: list[int], target: int, angle: float) -> None:
        phase(ctrl, -angle / 2)
        phase([*ctrl, target], angle)

    def single(
        ctrl: list[int], target: int, theta: float, phi: float, lam: float
    ) -> None:
        if not ctrl:
            result.append(("u3", [theta, phi, lam], [target]))
            return
        # U3 = exp(i*(phi+lambda)/2) Rz(phi) Ry(theta) Rz(lambda).
        phase(ctrl, (phi + lam) / 2)
        rotation_z(ctrl, target, lam)
        emit("sdg", [], [target], [])
        emit("h", [], [target], [])
        rotation_z(ctrl, target, theta)
        emit("h", [], [target], [])
        emit("s", [], [target], [])
        rotation_z(ctrl, target, phi)

    def emit(name: str, p: list[float], q: list[int], ctrl: list[int]) -> None:
        name = name.lower()
        p = [validate_parameter(v) for v in p]
        if name in {"u", "u3"}:
            single(ctrl, q[0], *p)
        elif name == "u2":
            single(ctrl, q[0], pi / 2, *p)
        elif name in {"u1", "p"}:
            phase([*ctrl, q[0]], p[0])
        elif name in {"id", "i"}:
            return
        elif name == "x":
            emit("h", [], [q[0]], [])
            phase([*ctrl, q[0]], pi)
            emit("h", [], [q[0]], [])
        elif name == "y":
            single(ctrl, q[0], pi, pi / 2, pi / 2)
        elif name == "z":
            phase([*ctrl, q[0]], pi)
        elif name == "h":
            single(ctrl, q[0], pi / 2, 0.0, pi)
        elif name in {"s", "sdg", "t", "tdg"}:
            phase(
                [*ctrl, q[0]],
                {"s": pi / 2, "sdg": -pi / 2, "t": pi / 4, "tdg": -pi / 4}[name],
            )
        elif name in {"sx", "sxdg"}:
            emit("h", [], q, ctrl)
            emit("s" if name == "sx" else "sdg", [], q, ctrl)
            emit("h", [], q, ctrl)
        elif name in {"rx", "ry", "rz"}:
            if name == "rz":
                if ctrl:
                    rotation_z(ctrl, q[0], p[0])
                else:
                    # Include the scalar phase using U3 with theta=0 and
                    # a two-X conjugation of a phase gate.
                    emit("x", [], q, [])
                    phase(q, -p[0] / 2)
                    emit("x", [], q, [])
                    phase(q, p[0] / 2)
            elif name == "rx":
                single(ctrl, q[0], p[0], -pi / 2, pi / 2)
            else:
                single(ctrl, q[0], p[0], 0.0, 0.0)
        elif name == "cx":
            if not ctrl:
                result.append(("cx", [], q))
            else:
                emit("x", [], [q[-1]], [*ctrl, q[0]])
        elif name in {
            "cy",
            "cz",
            "ch",
            "cs",
            "csdg",
            "crx",
            "cry",
            "crz",
            "cp",
            "cu1",
            "cu3",
            "cu",
        }:
            base = {"cp": "p", "cu1": "p", "cu3": "u3", "cu": "u3"}.get(name, name[1:])
            if name == "cu":
                phase([*ctrl, q[0]], p[3])
                p = p[:3]
            emit(base, p, [q[1]], [*ctrl, q[0]])
        elif name in {"ccx", "mcx"}:
            emit("x", [], [q[-1]], [*ctrl, *q[:-1]])
        elif name in {"swap", "cswap"}:
            if name == "cswap":
                ctrl = [*ctrl, q[0]]
                q = q[1:]
            emit("cx", [], [q[0], q[1]], ctrl)
            emit("cx", [], [q[1], q[0]], ctrl)
            emit("cx", [], [q[0], q[1]], ctrl)
        elif name in {"iswap", "iswapdg"}:
            sign = "s" if name == "iswap" else "sdg"
            emit(sign, [], [q[0]], ctrl)
            emit(sign, [], [q[1]], ctrl)
            emit("cz", [], q, ctrl)
            emit("swap", [], q, ctrl)
        elif name == "dcx":
            emit("cx", [], q, ctrl)
            emit("cx", [], q[::-1], ctrl)
        elif name == "rccx":
            from ...simulator.gates import decompose

            for sub, values, wires in decompose(name, q):
                emit(sub, values, wires, ctrl)
        elif name in {"rzz", "rxx", "ryy"}:
            if name == "rxx":
                for wire in q:
                    emit("h", [], [wire], ctrl)
            elif name == "ryy":
                for wire in q:
                    emit("rx", [pi / 2], [wire], ctrl)
            emit("cx", [], q, ctrl)
            emit("rz", p, [q[1]], ctrl)
            emit("cx", [], q, ctrl)
            if name == "rxx":
                for wire in q:
                    emit("h", [], [wire], ctrl)
            elif name == "ryy":
                for wire in q:
                    emit("rx", [-pi / 2], [wire], ctrl)
        elif name == "ecr":
            emit("h", [], [q[1]], ctrl)
            emit("rzz", [pi / 2], q, ctrl)
            emit("h", [], [q[1]], ctrl)
            emit("x", [], [q[0]], ctrl)
        else:
            raise ValueError(f"Cannot export unsupported gate {name}")

    def visit(item: dict[str, Any], ctrl: list[int]) -> None:
        if "base_operation" in item:
            visit(item["base_operation"], [*ctrl, *item["control_qubits"]])
        elif "order" in item:
            for sub in item["order"]:
                visit(sub, ctrl)
        else:
            name = item["gate_name"].lower()
            if name in {"measure", "reset", "barrier"}:
                if ctrl:
                    raise ValueError(
                        "Non-unitary operations and barriers cannot be controlled"
                    )
                wires = item["qubits"] + (
                    item.get("clbits", []) if name == "measure" else []
                )
                result.append((name, [], wires))
            else:
                emit(name, item.get("params") or [], item["qubits"], ctrl)

    visit(op, [])
    return result
