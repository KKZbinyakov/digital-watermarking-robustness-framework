"""Common behavioral contract for dependency-free expertise metrics."""

import numpy as np
import pytest

from .helpers import EXPECTED_METRICS, PYIQA_METRICS, arguments_for, evaluate

CORE_METRICS = sorted(EXPECTED_METRICS - PYIQA_METRICS)


@pytest.mark.parametrize("name", CORE_METRICS)
def test_returns_float_on_valid_input(name, photo, distorted):
    value = evaluate(name, **arguments_for(name, photo, distorted))

    assert isinstance(value, float), f"{name} returned {type(value).__name__}"
    assert not np.isnan(value), f"{name} returned nan for valid representative input"


@pytest.mark.parametrize("name", CORE_METRICS)
def test_is_deterministic(name, photo, distorted):
    arguments = arguments_for(name, photo, distorted)

    assert evaluate(name, **arguments) == evaluate(name, **arguments)


@pytest.mark.parametrize("name", CORE_METRICS)
def test_does_not_modify_array_inputs(name, photo, distorted):
    arguments = arguments_for(name, photo, distorted)
    snapshots = {key: value.copy() for key, value in arguments.items() if isinstance(value, np.ndarray)}

    evaluate(name, **arguments)

    for key, before in snapshots.items():
        assert np.array_equal(arguments[key], before), f"{name} modified {key}"
