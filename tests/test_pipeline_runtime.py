"""Tests for runtime artifact and solution adapters."""

from types import SimpleNamespace

import numpy as np
import pytest

from dwarf.pipeline.artifacts import ExtractedWatermark, ImageArtifact, OriginalWatermark
from dwarf.pipeline.exceptions import ArtifactValidationError
from dwarf.pipeline.runtime import (
    EmbeddingRuntimeAdapter,
    ErasurePolicy,
    MetricRuntimeAdapter,
    derive_runtime_seed,
    materialize_watermark,
)


class IdentityEmbedding:
    @staticmethod
    def embedding(**args):
        return args["input_image"].copy()

    @staticmethod
    def extraction(**args):
        return np.zeros(args["num_bits"], dtype=np.int8)


class BitErrorRate:
    @staticmethod
    def expertise(**args):
        left = args["original_bits"]
        right = args["extracted_bits"]
        return sum(a != b for a, b in zip(left, right)) / len(left)


def test_random_watermark_materialization_is_reproducible():
    reference = SimpleNamespace(kind="random_bits", length=32, seed=123)
    config = SimpleNamespace(watermark=SimpleNamespace(type="random_bits", length=32))

    first = materialize_watermark(reference, config)
    second = materialize_watermark(reference, config)

    assert first == second
    assert first.length == 32


def test_runtime_seed_is_domain_separated():
    assert derive_runtime_seed(42, "attack", 0) == derive_runtime_seed(42, "attack", 0)
    assert derive_runtime_seed(42, "attack", 0) != derive_runtime_seed(42, "attack", 1)
    assert derive_runtime_seed(42, "attack", 0) != derive_runtime_seed(42, "embedding", 0)


def test_embedding_adapter_preserves_rgb_contract():
    variant = SimpleNamespace(
        solution_name="IdentityEmbedding",
        implementation=IdentityEmbedding,
        embedding_parameters={},
        extraction_parameters={},
        spec=None,
    )
    image = ImageArtifact(np.zeros((16, 16, 3), dtype=np.uint8))
    watermark = OriginalWatermark([0, 1, 0, 1])
    adapter = EmbeddingRuntimeAdapter()

    embedded = adapter.embed(variant, image, watermark)
    extracted = adapter.extract(variant, embedded, num_bits=4)

    assert embedded.shape == image.shape
    assert embedded.array.dtype == np.uint8
    assert extracted.length == 4


def test_ber_count_as_error_penalizes_erasures():
    metric = SimpleNamespace(
        solution_name="BER",
        implementation=BitErrorRate,
        inputs={
            "original_bits": "original_watermark",
            "extracted_bits": "extracted_watermark",
        },
        parameters={},
    )
    artifacts = {
        "original_watermark": OriginalWatermark([0, 1, 1]),
        "extracted_watermark": ExtractedWatermark([0, -1, 1]),
    }

    value = MetricRuntimeAdapter().evaluate(
        metric,
        artifacts,
        erasure_policy=ErasurePolicy.COUNT_AS_ERROR,
    )

    assert value == pytest.approx(1 / 3)


def test_ber_fail_policy_rejects_erasures():
    metric = SimpleNamespace(
        solution_name="BER",
        implementation=BitErrorRate,
        inputs={
            "original_bits": "original_watermark",
            "extracted_bits": "extracted_watermark",
        },
        parameters={},
    )
    artifacts = {
        "original_watermark": OriginalWatermark([0, 1]),
        "extracted_watermark": ExtractedWatermark([0, -1]),
    }

    with pytest.raises(ArtifactValidationError, match="erased"):
        MetricRuntimeAdapter().evaluate(
            metric,
            artifacts,
            erasure_policy=ErasurePolicy.FAIL,
        )
