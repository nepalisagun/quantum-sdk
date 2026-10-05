from .circuitoperation import CircuitOperation, OperationType
from .validation import validate_index
import copy


class CustomControlOperation(CircuitOperation):
    """
    For a given operation this will transform the operation into a controlled operation.
    """

    def __init__(self, *control: int, operation: CircuitOperation):
        if not control:
            raise ValueError("At least one control qubit is required")
        control = tuple(validate_index(q) for q in control)
        if len(set(control)) != len(control):
            raise ValueError("Control qubits must be distinct")
        for control_qubit in control:
            if not isinstance(control_qubit, int):
                raise ValueError("Control qubit must be an integer")
            if control_qubit in operation.qubits:
                raise ValueError("Control qubit already in operation qubits")

        def has_non_unitary(item: CircuitOperation) -> bool:
            if item.operation_type in (
                OperationType.BARRIER,
                OperationType.MEASURE,
                OperationType.RESET,
            ):
                return True
            return any(
                has_non_unitary(child) for child in getattr(item, "order", None) or []
            )

        if has_non_unitary(operation):
            raise ValueError(
                f"Operation {operation.gate_name} cannot be controlled by another qubit"
            )

        super().__init__(
            operation.operation_type,
            f"{'C' * len(control)}{operation.gate_name}",
            [*control, *operation.qubits],
            operation.params,
            operation.clbits,
        )
        self.control_qubits = list(control)
        self.base_operation = copy.deepcopy(operation)
