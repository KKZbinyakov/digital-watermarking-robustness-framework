"""Fixtures local to expertise tests."""

import numpy as np
import pytest


@pytest.fixture
def distorted(photo):
    """Return a deterministic noisy version of the shared photo fixture."""
    noise = np.random.default_rng(99).normal(0.0, 10.0, photo.shape)
    return np.clip(photo.astype(np.float64) + noise, 0, 255).astype(np.uint8)
