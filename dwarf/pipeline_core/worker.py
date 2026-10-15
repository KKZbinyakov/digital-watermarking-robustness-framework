import importlib
from hashlib import sha256
from multiprocessing import shared_memory

import numpy as np

from dwarf.core.attack_orchestrator.attack_core import Attack_Core
from dwarf.core.dwarf_exceptions import Dwarf_Exception
from dwarf.core.embedding_orchestrator.embedding_core import Embedding_Core
from dwarf.core.expertise_orchestrator.expertise_core import Expertise_Core
from dwarf.core.logs.logger import logger
from dwarf.pipeline_core.case_builder import Case

_WORKER_STATE = {}


def init_worker(plugin_paths, watermark_spec, experiment_seed, output_spec, output_directory):
    global _WORKER_STATE
    for path in plugin_paths:
        importlib.import_module(path)
    _WORKER_STATE["watermark_spec"] = watermark_spec
    _WORKER_STATE["experiment_seed"] = experiment_seed
    _WORKER_STATE["output_spec"] = output_spec
    _WORKER_STATE["output_directory"] = output_directory


def bind_inputs(bindings, runtime):
    out = {}
    for name, artifact in bindings.items():
        value = runtime[artifact]
        if isinstance(value, np.ndarray) and value.ndim == 2:
            value = np.stack([value] * 3, axis=-1)
        out[name] = value
    return out


def generate_watermark(spec: dict, base_seed: int, case_seed: int, image_id: str) -> np.ndarray:
    """Детерминированно порождает ЦВЗ для одного кейса.

    spec — model_dump() от RandomBitsWatermarkSpec или FixedBitsWatermarkSpec.
    Возвращает np.ndarray формы (length,) из 0/1, dtype=uint8.
    """
    if spec["type"] == "fixed_bits":
        bits_str = spec["bits"]
        return np.frombuffer(bits_str.encode("ascii"), dtype=np.uint8) - ord("0")

    # type == "random_bits"
    scope = spec["scope"]
    if scope == "fixed_for_run":
        seed_value = base_seed
    elif scope == "per_image":
        seed_value = int.from_bytes(
            sha256(f"{base_seed}:{image_id}".encode()).digest()[:8],
            "big",
        )
    elif scope == "per_case":
        seed_value = case_seed
    else:
        raise Dwarf_Exception(f"unknown watermark scope: {scope!r}")

    length = spec["length"]
    rng = np.random.default_rng(seed_value)
    return rng.integers(0, 2, size=length, dtype=np.uint8)


def run_metrics(checkpoint, metrics, runtime, result):
    for metric in metrics:
        if metric.checkpoint == checkpoint:
            bound_inputs = bind_inputs(metric.inputs, runtime)
            result["metrics"][metric.id] = Expertise_Core.get_expertise_class_by_name(metric.name).expertise(
                **bound_inputs, **metric.params
            )
    return result


def run_case(case: Case) -> dict:
    global _WORKER_STATE
    result = {
        "case_id": case.case_id,
        "status": "done",
        "metrics": {},
        "artifacts": {},
        "error": None,
    }
    shm = None
    try:
        shm = shared_memory.SharedMemory(name=case.shm_name)
        image = np.ndarray(shape=case.image_shape, dtype=case.image_dtype, buffer=shm.buf)
        image.flags.writeable = False
        watermark = generate_watermark(
            spec=_WORKER_STATE["watermark_spec"],
            base_seed=_WORKER_STATE["experiment_seed"],
            case_seed=case.seed,
            image_id=case.image_id,
        )

        runtime = {
            "input_image": image,
            "original_image": image,
            "watermark": watermark,
            "original_watermark": watermark,
            "watermark_bits": watermark,
            "original_bits": watermark,
            "num_bits": len(watermark),
            "embedded_image": None,
            "attacked_image": None,
            "distorted_image": None,
            "output_image": None,
            "extracted_watermark": None,
            "extracted_bits": None,
        }

        emb_cls = Embedding_Core.get_embedding_class_by_name(case.embedding_name)
        runtime["embedded_image"] = emb_cls.embedding(**runtime, **case.embedding_params)
        if runtime["embedded_image"] is None:
            raise Dwarf_Exception("embedding failed")
        runtime["input_image"] = runtime["embedded_image"]

        result = run_metrics("after_embedding", case.metrics, runtime, result)

        for step in case.attack_chain:
            atk_cls = Attack_Core.get_attack_class_by_name(step.name)
            runtime["attacked_image"] = atk_cls.attack(**runtime, **step.params)
            if runtime["attacked_image"] is None:
                raise Dwarf_Exception("attack failed")
            runtime["input_image"] = runtime["attacked_image"]

        if runtime["attacked_image"] is None:
            runtime["attacked_image"] = runtime["embedded_image"]

        result = run_metrics("after_attack", case.metrics, runtime, result)

        runtime["extracted_watermark"] = emb_cls.extraction(**runtime, **case.extraction_params)
        if runtime["extracted_watermark"] is None:
            raise Dwarf_Exception("extraction failed")

        result = run_metrics("after_extraction", case.metrics, runtime, result)
        result["status"] = "done"

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
    finally:
        shm.close()

    logger.log(f"case {case.case_id}: status={result['status']}, error={result['error']}, metrics={result['metrics']}")

    return result
