"""Dependency-free contract tests for the pyiqa metric adapters."""

import importlib
import sys
from types import ModuleType

import numpy as np
import pytest

from dwarf.ready_solutions.utils import expertise_utils

from .helpers import registered_metrics


class ScalarResult:
    """Minimal pyiqa-like scalar result used by adapter tests."""

    def __init__(self, value):
        self.value = value

    def item(self):
        """Return the wrapped scalar."""
        return self.value


@pytest.mark.parametrize(
    ("name", "module_name", "backend_name", "argument_names"),
    [
        ("BRISQUE", "brisque", "brisque", ("input",)),
        ("DISTS", "dists", "dists", ("distorted", "original")),
        ("LPIPS", "lpips", "lpips", ("distorted", "original")),
        ("NIQE", "niqe", "niqe", ("input",)),
    ],
)
def test_adapter_selects_backend_and_preserves_argument_order(
    name,
    module_name,
    backend_name,
    argument_names,
    monkeypatch,
):
    module = importlib.import_module(f"dwarf.ready_solutions.expertise_solutions.imperceptibility.{module_name}")
    images = {
        "input": np.full((2, 3, 3), 10, dtype=np.uint8),
        "original": np.full((2, 3, 3), 20, dtype=np.uint8),
        "distorted": np.full((2, 3, 3), 30, dtype=np.uint8),
    }
    selected_backends = []
    converted_images = []
    received_arguments = []

    def fake_to_tensor(image):
        converted_images.append(image)
        return next(label for label, candidate in images.items() if candidate is image)

    def fake_iqa_metric(selected_name):
        selected_backends.append(selected_name)

        def metric(*arguments):
            received_arguments.append(arguments)
            return ScalarResult(0.375)

        return metric

    monkeypatch.setattr(module, "to_tensor", fake_to_tensor)
    monkeypatch.setattr(module, "iqa_metric", fake_iqa_metric)

    if name in {"BRISQUE", "NIQE"}:
        value = registered_metrics()[name].expertise(input_image=images["input"])
    else:
        value = registered_metrics()[name].expertise(
            original_image=images["original"],
            distorted_image=images["distorted"],
        )

    assert value == 0.375
    assert selected_backends == [backend_name]
    assert received_arguments == [argument_names]
    assert len(converted_images) == len(argument_names)
    assert all(converted_images[index] is images[label] for index, label in enumerate(argument_names))


def test_iqa_metric_reports_missing_optional_dependency(monkeypatch):
    monkeypatch.setattr(expertise_utils, "_IQA_CACHE", {})
    monkeypatch.setitem(sys.modules, "pyiqa", None)

    with pytest.raises(RuntimeError, match="pyiqa package"):
        expertise_utils.iqa_metric("lpips")


def test_iqa_metric_creates_each_backend_once(monkeypatch):
    created = []
    metric = object()
    fake_pyiqa = ModuleType("pyiqa")

    def create_metric(name):
        created.append(name)
        return metric

    fake_pyiqa.create_metric = create_metric
    monkeypatch.setattr(expertise_utils, "_IQA_CACHE", {})
    monkeypatch.setitem(sys.modules, "pyiqa", fake_pyiqa)

    assert expertise_utils.iqa_metric("dists") is metric
    assert expertise_utils.iqa_metric("dists") is metric
    assert created == ["dists"]


def test_to_tensor_normalizes_and_reorders_rgb(monkeypatch):
    captured = []
    operations = []

    class FakeTensor:
        def permute(self, *dimensions):
            operations.append(("permute", dimensions))
            return self

        def unsqueeze(self, dimension):
            operations.append(("unsqueeze", dimension))
            return self

    fake_torch = ModuleType("torch")

    def from_numpy(array):
        captured.append(array.copy())
        return FakeTensor()

    fake_torch.from_numpy = from_numpy
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    image = np.array([[[0, 127, 255], [255, 64, 0]]], dtype=np.uint8)

    result = expertise_utils.to_tensor(image)

    assert isinstance(result, FakeTensor)
    assert captured[0].dtype == np.float32
    np.testing.assert_allclose(captured[0], image.astype(np.float32) / 255.0)
    assert operations == [("permute", (2, 0, 1)), ("unsqueeze", 0)]


def test_to_tensor_reports_missing_torch_dependency(monkeypatch, small_photo):
    monkeypatch.setitem(sys.modules, "torch", None)

    with pytest.raises(RuntimeError, match="torch"):
        expertise_utils.to_tensor(small_photo)
