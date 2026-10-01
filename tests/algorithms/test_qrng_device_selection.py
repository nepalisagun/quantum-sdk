"""Per-call QRNG device selection must preserve the public local default."""

from unittest.mock import MagicMock, patch

import pytest

from qpiai_quantum.algorithms.qrng import QRNG


@pytest.mark.parametrize("device_name", [None, "QpiAI-QSV-Simulator"])
def test_generate_forwards_device_and_preserves_output(device_name):
    result = MagicMock()
    result.get.return_value = {"counts": {"10": 1}}
    with patch.object(QRNG, "run", return_value=result) as run:
        rng = QRNG(n_bits=2)
        kwargs = {} if device_name is None else {"device_name": device_name}
        assert rng.generate(output_format="bitstring", **kwargs) == "10"
    run.assert_called_once_with(
        shots=1,
        experiment_name="Default Experiment",
        device_name=device_name or "QpiAI-QSV-Local",
    )
