"""Opt-in integration checks for optional metrics backed by pyiqa."""

import importlib.util
import os

import numpy as np
import pytest

from .helpers import PYIQA_METRICS, arguments_for, evaluate

pytestmark = pytest.mark.skipif(
    os.environ.get("DWARF_RUN_PYIQA_INTEGRATION") != "1",
    reason="set DWARF_RUN_PYIQA_INTEGRATION=1 to run model-backed pyiqa integration tests",
)


@pytest.fixture(scope="module", autouse=True)
def require_pyiqa():
    """Skip the opt-in integration module when pyiqa is absent."""
    if importlib.util.find_spec("pyiqa") is None:
        pytest.skip("optional pyiqa dependency is not installed")


@pytest.mark.parametrize("name", sorted(PYIQA_METRICS))
def test_optional_metric_returns_finite_float_without_mutating_inputs(name, photo, distorted):
    arguments = arguments_for(name, photo, distorted)
    snapshots = {key: value.copy() for key, value in arguments.items() if isinstance(value, np.ndarray)}

    value = evaluate(name, **arguments)

    assert isinstance(value, float)
    assert np.isfinite(value)
    for key, before in snapshots.items():
        assert np.array_equal(arguments[key], before), f"{name} modified {key}"
