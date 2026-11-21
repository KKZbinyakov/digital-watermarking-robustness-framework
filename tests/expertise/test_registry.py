"""Registration and category checks for expertise metrics."""

import pytest

from dwarf.core.expertise_orchestrator.expertise_core import (
    Expertise_Core,
    Ready_Imperceptibility_Expertise,
    Ready_Robustness_Expertise,
)

from .helpers import EXPECTED_METRICS, IMPERCEPTIBILITY_METRICS, ROBUSTNESS_METRICS, registered_metrics

PAPER_METRICS = frozenset({"Bit_Agreement", "Channel_PSNR"})


def test_registry_contains_every_builtin_metric_exactly_once():
    """A failed import must not silently remove a metric from the benchmark."""
    assert set(registered_metrics()) == EXPECTED_METRICS | PAPER_METRICS


@pytest.mark.parametrize("name", sorted(EXPECTED_METRICS | PAPER_METRICS))
def test_metric_is_reachable_through_orchestrator(name):
    metric = registered_metrics()[name]
    assert Expertise_Core.get_expertise_class_by_name(name) is metric
    assert getattr(Expertise_Core, name) is metric


@pytest.mark.parametrize("name", sorted(IMPERCEPTIBILITY_METRICS | {"Channel_PSNR"}))
def test_imperceptibility_metric_has_correct_category(name):
    assert issubclass(registered_metrics()[name], Ready_Imperceptibility_Expertise)


@pytest.mark.parametrize("name", sorted(ROBUSTNESS_METRICS | {"Bit_Agreement"}))
def test_robustness_metric_has_correct_category(name):
    assert issubclass(registered_metrics()[name], Ready_Robustness_Expertise)
