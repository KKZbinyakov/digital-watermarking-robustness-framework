from dwarf.pipeline_core.pipeline import Parent, load_experiment_config
from pathlib import Path

if __name__ == "__main__":
    config = load_experiment_config(Path(__file__).parent / "examples" / "pipeline" / "dct_jpeg.yaml")
    Parent(config).run()