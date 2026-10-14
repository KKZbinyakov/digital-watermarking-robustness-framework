import dataclasses
import itertools
import json
from collections.abc import Iterator, Mapping
from hashlib import sha256
from typing import Any

from dwarf.pipeline_core.config import ParameterSpace


def compute_case_id(canonical_payload) -> str:
    """
    case id

    Учитывает preprocessing-спек.
    Возвращает 16 hex-символов.
    """
    return sha256(canonical_payload.encode("utf-8")).hexdigest()[:16]

def seed(base_seed: int, case_id: str) -> int:
    payload = f"{base_seed}:{case_id}".encode()
    return int.from_bytes(sha256(payload).digest()[:8], "big")

def canonical(object):
    object = dict(object)
    return json.dumps(object, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

def expand(space: ParameterSpace) -> Iterator[dict[str, Any]]:

    base = dict(space.fixed)

    variants = space.variants
    if variants:
        for variant in variants:
            merged = dict(base)
            merged.update(variant)
            yield merged
        return

    grid = space.grid
    if grid:
        keys = list(grid.keys())
        value_lists = [grid[k] for k in keys]
        for combo in itertools.product(*value_lists):
            merged = dict(base)
            for k, v in zip(keys, combo, strict=True):
                merged[k] = v
            yield merged
        return

    yield dict(base)

@dataclasses.dataclass(frozen=True)
class ResolvedAttackStep:
    name: str
    params: Mapping[str, Any]

@dataclasses.dataclass(frozen=True)
class ResolvedMetric:
    id: str
    name: str
    checkpoint: str
    inputs: Mapping[str, str]
    params: Mapping[str, Any]

@dataclasses.dataclass(frozen=True)
class Case:
    case_id: str
    image_id: str
    seed: int
    repeat_index: int

    shm_name: str
    image_shape: tuple[int, ...]
    image_dtype: str

    embedding_id: str
    embedding_name: str
    embedding_params: Mapping[str, Any]
    extraction_name: str
    extraction_params: Mapping[str, Any]

    attack_chain: tuple[ResolvedAttackStep, ...]
    metrics: tuple[ResolvedMetric, ...]

class Case_Builder:
    def __init__(self, config):
        self.config = config
        self.shm_name = None
        self.image_shape = None
        self.image_dtype = None
        self.image_id = None

    def init_case(self, handle):
        self.shm_name = handle.get("shm_name")
        self.image_shape = tuple(handle.get("shape"))
        self.image_dtype = str(handle.get("dtype"))
        self.image_id = handle.get("image_id")

    def count_cases(self):
        embedding_count = sum(
            emb.embedding.params.declared_variant_count
            * emb.extraction.params.declared_variant_count
            for emb in self.config.embeddings
        )

        attack_count = 0
        for scenario in self.config.attack_scenarios:
            scenario_count = 1
            for step in scenario.steps:
                scenario_count *= step.params.declared_variant_count
            attack_count += scenario_count

        metric_count = 1
        for metric in self.config.metrics:
            metric_count *= metric.params.declared_variant_count

        return (
            embedding_count
            * attack_count
            * metric_count
            * self.config.experiment.repeats
        )

    def make_case(self, emb, emb_params, ext_params, scen, attack_chain, metrics, repeat):
        payload = {
            "image_id": self.image_id,
            "embedding_id": emb.id,
            "embedding_name": emb.name,
            "embedding_params": dict(emb_params),
            "extraction_name": emb.name,
            "extraction_params": dict(ext_params),
            "attack_scenario_id": scen.id,
            "attack_chain": [(s.name, dict(s.params)) for s in attack_chain],
            "metrics": [(m.id, m.name, m.checkpoint, dict(m.inputs), dict(m.params)) for m in metrics],
            "repeat_index": repeat,
        }
        canonical_payload = None
        canonical_payload = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        case_id = compute_case_id(canonical_payload)
        case_seed = seed(self.config.experiment.seed, case_id)
        return Case(
            case_id=case_id, image_id=self.image_id, seed=case_seed,
            repeat_index=repeat,
            shm_name=self.shm_name, image_shape=self.image_shape, image_dtype=self.image_dtype,
            embedding_id=emb.id, embedding_name=emb.name, embedding_params=emb_params,
            extraction_name=emb.name, extraction_params=ext_params,
            attack_chain=attack_chain, metrics=metrics,
        )

    def build_cases(self):
        if self.shm_name is None:
            raise RuntimeError("Case builder not initialized")

        self.metric_combos = list(itertools.product(*[expand(m.params) for m in self.config.metrics]))
        self.step_combos_by_scen = {
            scen.id: list(itertools.product(*[expand(s.params) for s in scen.steps]))
            for scen in self.config.attack_scenarios
        }

        for emb in self.config.embeddings:
            for emb_params in expand(emb.embedding.params):
                for ext_params in expand(emb.extraction.params):
                    for scen in self.config.attack_scenarios:
                        for combo in self.step_combos_by_scen[scen.id]:
                            attack_chain = tuple(
                                ResolvedAttackStep(name=step.name, params=params)
                                for step, params in zip(scen.steps, combo, strict=True)
                            )
                            for combo in self.metric_combos:
                                metrics = tuple(
                                    ResolvedMetric(id=m.id, name=m.name, checkpoint=m.checkpoint,
                                                inputs=m.inputs, params=params)
                                    for m, params in zip(self.config.metrics, combo, strict=True)
                                )
                                for repeat in range(self.config.experiment.repeats):
                                    yield self.make_case(emb, emb_params, ext_params, scen, attack_chain, metrics, repeat)

