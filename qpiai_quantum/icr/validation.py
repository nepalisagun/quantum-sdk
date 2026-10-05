"""Shared validation for circuit operations and numerical execution."""

import math
from numbers import Integral, Real


def validate_index(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError("Qubit and classical bit indices must be nonnegative integers")
    return int(value)


def validate_parameter(value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
    ):
        raise ValueError("Gate parameters must be finite real numbers")
    return float(value)
