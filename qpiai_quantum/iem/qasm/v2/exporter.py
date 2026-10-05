"""Self-contained OpenQASM 2 export in the portable U3/CX basis."""

from ..primitives import lower_operation
from ....circuit.circuit import Circuit


class QASM2:
    # These names are defined by the minimal OpenQASM 2 qelib1.inc contract.
    STANDARD_GATES = {
        "u3",
        "u2",
        "u1",
        "cx",
        "id",
        "u",
        "p",
        "x",
        "y",
        "z",
        "h",
        "s",
        "sdg",
        "t",
        "tdg",
        "rx",
        "ry",
        "rz",
        "cz",
        "cy",
        "ch",
        "ccx",
        "crz",
        "cu1",
        "cu3",
    }

    @staticmethod
    def generate(circuit: Circuit) -> str:
        lines = [
            "OPENQASM 2.0;",
            'include "qelib1.inc";',
            f"qreg q[{circuit.num_qubits}];",
        ]
        if circuit.num_clbits:
            lines.append(f"creg c[{circuit.num_clbits}];")
        for operation in circuit.icr.evolve:
            name = operation.gate_name.lower()
            if (
                name in QASM2.STANDARD_GATES
                and getattr(operation, "base_operation", None) is None
                and getattr(operation, "order", None) is None
            ):
                name = {"u": "u3", "p": "u1"}.get(name, name)
                primitives = [(name, operation.params or [], operation.qubits)]
            else:
                primitives = lower_operation(operation)
            for name, params, wires in primitives:
                if name == "measure":
                    lines.append(f"measure q[{wires[0]}] -> c[{wires[1]}];")
                else:
                    values = (
                        "(" + ", ".join(repr(float(p)) for p in params) + ")"
                        if params
                        else ""
                    )
                    operands = ", ".join(f"q[{q}]" for q in wires)
                    lines.append(f"{name}{values} {operands};")
        return "\n".join(lines)
