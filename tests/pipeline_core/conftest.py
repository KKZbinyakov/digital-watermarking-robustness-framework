"""Small experiment declarations shared by pipeline tests."""

import pytest


@pytest.fixture
def experiment_data(tmp_path):
    """Return a fresh valid declaration without creating datasets or processes."""
    parameters = {"spreading_length": 64, "num_blocks": 16, "key": 7}
    return {
        "schema_version": 1,
        "experiment": {"name": "pipeline-test", "seed": 42, "repeats": 1},
        "plugins": ["dwarf.ready_solutions"],
        "dataset": {"type": "directory", "path": str(tmp_path / "images")},
        "watermark": {"type": "fixed_bits", "bits": "1011"},
        "embeddings": [
            {
                "id": "adc",
                "name": "ADC_ABW",
                "embedding": {"params": {"fixed": {**parameters, "attack_angle": 30.0}}},
                "extraction": {"params": {"fixed": parameters}},
            }
        ],
        "attack_scenarios": [
            {"id": "clean", "steps": []},
            {"id": "jpeg", "steps": [{"name": "Jpeg", "params": {"fixed": {"quality": 95}}}]},
        ],
        "metrics": [
            {
                "id": "embedding-mse",
                "name": "MSE",
                "checkpoint": "after_embedding",
                "inputs": {"original_image": "original_image", "distorted_image": "embedded_image"},
            },
            {
                "id": "attack-mse",
                "name": "MSE",
                "checkpoint": "after_attack",
                "inputs": {"original_image": "embedded_image", "distorted_image": "attacked_image"},
            },
            {
                "id": "ber",
                "name": "BER",
                "checkpoint": "after_extraction",
                "inputs": {"original_bits": "original_watermark", "extracted_bits": "extracted_watermark"},
            },
        ],
        "execution": {"backend": "process", "workers": 1, "max_in_flight": 1, "start_method": "spawn"},
        "output": {"directory": str(tmp_path / "runs"), "overwrite": True, "save_images": "none"},
        "report": {"enabled": False},
    }
