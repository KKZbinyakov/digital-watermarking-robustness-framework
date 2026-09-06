"""Tests for the serial experiment executor."""

from types import SimpleNamespace

import numpy as np

from dwarf.pipeline.artifacts import ImageArtifact
from dwarf.pipeline.executors import SerialExecutor
from dwarf.pipeline.results import ResultStatus


class PixelEmbedding:
    embedding_calls = 0

    @staticmethod
    def embedding(**args):
        PixelEmbedding.embedding_calls += 1
        image = args["input_image"].copy()
        bits = np.asarray(args["watermark_bits"], dtype=np.uint8)
        image.reshape(-1, 3)[: bits.size, 0] = bits * 255
        return image

    @staticmethod
    def extraction(**args):
        image = args["input_image"]
        return (image.reshape(-1, 3)[: args["num_bits"], 0] > 127).astype(np.int8)


class AddAttack:
    received_first_pixels = []

    @staticmethod
    def attack(**args):
        image = args["input_image"]
        AddAttack.received_first_pixels.append(int(image[0, 0, 1]))
        image[0, 0, 1] = min(255, int(image[0, 0, 1]) + int(args["amount"]))
        return image


class EqualMetric:
    @staticmethod
    def expertise(**args):
        return float(np.array_equal(args["original_image"], args["distorted_image"]))


class BitErrorRate:
    @staticmethod
    def expertise(**args):
        left = args["original_bits"]
        right = args["extracted_bits"]
        return sum(a != b for a, b in zip(left, right)) / len(left)


def metric(metric_id, solution_name, implementation, checkpoint, inputs):
    return SimpleNamespace(
        id=metric_id,
        name=solution_name,
        implementation=implementation,
        spec=SimpleNamespace(
            operations={"expertise": SimpleNamespace(parameters={})},
        ),
        checkpoint=checkpoint,
        inputs=inputs,
        params=SimpleNamespace(fixed={}, grid={}, variants=()),
    )


def make_executor():
    PixelEmbedding.embedding_calls = 0
    AddAttack.received_first_pixels = []
    sample = SimpleNamespace(sample_id="sample")
    watermark = SimpleNamespace(kind="random_bits", length=4, seed=123)
    embedding = SimpleNamespace(
        variant_id="embedding",
        solution_name="PixelEmbedding",
        implementation=PixelEmbedding,
        embedding_parameters={},
        extraction_parameters={},
        spec=None,
    )
    cases = []
    for index, amount in enumerate((1, 2)):
        step = SimpleNamespace(
            variant_id=f"step-{amount}",
            solution_name="AddAttack",
            implementation=AddAttack,
            parameters={"amount": amount},
            spec=None,
        )
        attack = SimpleNamespace(variant_id=f"attack-{amount}", steps=(step,))
        cases.append(
            SimpleNamespace(
                case_id=f"case-{index}",
                work_unit_id="work-unit",
                ordinal=index,
                sample=sample,
                embedding=embedding,
                attack=attack,
                repeat_index=0,
                embedding_seed=11,
                case_seed=20 + index,
                watermark=watermark,
            )
        )
    unit = SimpleNamespace(
        work_unit_id="work-unit",
        sample=sample,
        embedding=embedding,
        watermark=watermark,
        embedding_seed=11,
        cases=tuple(cases),
    )

    class Plan:
        fingerprint = "plan"
        manifest = SimpleNamespace(fingerprint="manifest")
        counts = SimpleNamespace(case_count=2, work_unit_count=1)

        @staticmethod
        def iter_work_units():
            return iter((unit,))

    class Source:
        @staticmethod
        def load(sample, verify_integrity=True):
            return ImageArtifact(np.zeros((8, 8, 3), dtype=np.uint8))

    resolved = SimpleNamespace(
        config=SimpleNamespace(
            watermark=SimpleNamespace(type="random_bits", length=4),
            execution=SimpleNamespace(fail_fast=False),
        ),
        metrics=(
            metric(
                "image-equality",
                "EqualMetric",
                EqualMetric,
                "after_attack",
                {
                    "original_image": "embedded_image",
                    "distorted_image": "attacked_image",
                },
            ),
            metric(
                "ber",
                "BER",
                BitErrorRate,
                "after_extraction",
                {
                    "original_bits": "original_watermark",
                    "extracted_bits": "extracted_watermark",
                },
            ),
        ),
        catalog_fingerprint="catalog",
    )
    return SerialExecutor(Plan(), resolved, Source())


def test_serial_executor_reuses_embedding_and_isolates_attack_branches():
    result = make_executor().execute()

    assert PixelEmbedding.embedding_calls == 1
    assert AddAttack.received_first_pixels == [0, 0]
    assert len(result.cases) == 2
    assert all(case.status is ResultStatus.SUCCESS for case in result.cases)
    assert len(result.metrics) == 4


def test_serial_executor_is_deterministic_for_same_plan():
    first = make_executor().execute()
    second = make_executor().execute()

    assert [case.case_id for case in first.cases] == [case.case_id for case in second.cases]
    assert [metric.value for metric in first.metrics] == [metric.value for metric in second.metrics]
