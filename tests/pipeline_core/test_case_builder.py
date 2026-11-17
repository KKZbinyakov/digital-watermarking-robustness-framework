"""Observable parameter expansion, case counts and experiment seeds."""

import pytest

from dwarf.pipeline_core.case_builder import Case_Builder, expand
from dwarf.pipeline_core.config import ExperimentConfig, ParameterSpace


@pytest.mark.parametrize(
    ("declaration", "expected"),
    [
        ({"fixed": {"key": 7}}, [{"key": 7}]),
        (
            {"fixed": {"key": 7}, "grid": {"attack_angle": [20, 30], "num_blocks": [16, 32]}},
            [
                {"key": 7, "attack_angle": 20, "num_blocks": 16},
                {"key": 7, "attack_angle": 20, "num_blocks": 32},
                {"key": 7, "attack_angle": 30, "num_blocks": 16},
                {"key": 7, "attack_angle": 30, "num_blocks": 32},
            ],
        ),
        (
            {
                "fixed": {"key": 7},
                "variants": [{"attack_angle": 20, "num_blocks": 16}, {"attack_angle": 30, "num_blocks": 32}],
            },
            [{"key": 7, "attack_angle": 20, "num_blocks": 16}, {"key": 7, "attack_angle": 30, "num_blocks": 32}],
        ),
    ],
    ids=["fixed", "cartesian-grid", "linked-variants"],
)
def test_parameter_expansion_preserves_declared_combinations(declaration, expected):
    assert list(expand(ParameterSpace.model_validate(declaration))) == expected


def build_cases(data, shm_name="test-shared-memory"):
    """Build metadata without allocating shared memory or invoking a solution."""
    builder = Case_Builder(ExperimentConfig.model_validate(data))
    builder.init_case({"shm_name": shm_name, "shape": [128, 128, 3], "dtype": "uint8", "image_id": "host-1"})
    return builder, list(builder.build_cases())


def test_grid_chains_and_repeats_produce_the_expected_cases(experiment_data):
    embedding = experiment_data["embeddings"][0]
    del embedding["embedding"]["params"]["fixed"]["attack_angle"]
    embedding["embedding"]["params"]["grid"] = {"attack_angle": [20.0, 30.0]}
    del embedding["extraction"]["params"]["fixed"]["key"]
    embedding["extraction"]["params"]["grid"] = {"key": [7, 9]}
    experiment_data["attack_scenarios"][1]["steps"] = [
        {"name": "Jpeg", "params": {"grid": {"quality": [75, 95]}}},
        {"name": "Gamma_Correction", "params": {"grid": {"gamma": [0.8, 1.2]}}},
    ]
    experiment_data["experiment"]["repeats"] = 2

    builder, cases = build_cases(experiment_data)

    # Two embedding settings x two extraction keys x (clean + four chains) x two repeats.
    assert builder.count_cases() == 40
    assert len(cases) == 40
    assert len({case.case_id for case in cases}) == 40
    assert sum(not case.attack_chain for case in cases) == 8
    assert {case.repeat_index for case in cases} == {0, 1}
    chains = [case.attack_chain for case in cases if case.attack_chain]
    assert all([step.name for step in chain] == ["Jpeg", "Gamma_Correction"] for chain in chains)
    assert {(chain[0].params["quality"], chain[1].params["gamma"]) for chain in chains} == {
        (75, 0.8),
        (75, 1.2),
        (95, 0.8),
        (95, 1.2),
    }


def test_case_identity_does_not_depend_on_shared_memory_names(experiment_data):
    _, first = build_cases(experiment_data, "first-allocation")
    _, second = build_cases(experiment_data, "second-allocation")

    assert [(case.case_id, case.seed) for case in first] == [(case.case_id, case.seed) for case in second]


def test_experiment_seed_changes_case_random_streams(experiment_data):
    _, first = build_cases(experiment_data)
    experiment_data["experiment"]["seed"] = 43
    _, second = build_cases(experiment_data)

    assert [case.seed for case in first] != [case.seed for case in second]
