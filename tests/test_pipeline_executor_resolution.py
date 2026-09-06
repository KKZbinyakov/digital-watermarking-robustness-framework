"""Regression tests for resolving data-only planner variants at execution time."""

from types import SimpleNamespace

import numpy as np

from dwarf.pipeline.artifacts import ImageArtifact
from dwarf.pipeline.executors import SerialExecutor
from dwarf.pipeline.results import ResultStatus


class PixelEmbedding:
    embedding_calls = 0
    extraction_calls = 0

    @staticmethod
    def embedding(**args):
        PixelEmbedding.embedding_calls += 1
        image = args["input_image"].copy()
        bits = np.asarray(args["watermark_bits"], dtype=np.uint8)
        image.reshape(-1, 3)[: bits.size, 0] = bits * 255
        return image

    @staticmethod
    def extraction(**args):
        PixelEmbedding.extraction_calls += 1
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


def test_executor_resolves_data_only_embedding_and_attack_variants():
    PixelEmbedding.embedding_calls = 0
    PixelEmbedding.extraction_calls = 0
    AddAttack.received_first_pixels = []

    sample = SimpleNamespace(sample_id="sample")
    watermark = SimpleNamespace(kind="random_bits", length=4, seed=123)
    embedding = SimpleNamespace(
        variant_id="embedding-variant",
        declaration_id="embedding-declaration",
        solution_name="PixelEmbedding",
        embedding_parameters={},
        extraction_parameters={},
    )

    cases = []
    for index, amount in enumerate((1, 2)):
        step = SimpleNamespace(
            variant_id=f"step-{amount}",
            step_index=0,
            solution_name="AddAttack",
            parameters={"amount": amount},
        )
        cases.append(
            SimpleNamespace(
                case_id=f"case-{index}",
                work_unit_id="work-unit",
                ordinal=index,
                sample=sample,
                embedding=embedding,
                attack=SimpleNamespace(
                    variant_id=f"attack-{amount}",
                    steps=(step,),
                ),
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
        embeddings=(
            SimpleNamespace(
                id="embedding-declaration",
                name="PixelEmbedding",
                implementation=PixelEmbedding,
                spec=None,
            ),
        ),
        attack_scenarios=(
            SimpleNamespace(
                id="attack-scenario",
                steps=(
                    SimpleNamespace(
                        index=0,
                        name="AddAttack",
                        implementation=AddAttack,
                        spec=None,
                    ),
                ),
            ),
        ),
        metrics=(),
        catalog_fingerprint="catalog",
    )

    result = SerialExecutor(Plan(), resolved, Source()).execute()

    assert PixelEmbedding.embedding_calls == 1
    assert PixelEmbedding.extraction_calls == 2
    assert AddAttack.received_first_pixels == [0, 0]
    assert result.successful_case_count == 2
    assert result.skipped_case_count == 0
    assert all(case.status is ResultStatus.SUCCESS for case in result.cases)
