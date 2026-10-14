import json
import multiprocessing as mp
import os
import random
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from hashlib import sha256
from multiprocessing import shared_memory
from pathlib import Path
from queue import Empty, Full

from dwarf.core.ds_orchestrator.ds_core import DEAD_PILL, Ds_Core
from dwarf.core.dwarf_exceptions import Dwarf_Exception
from dwarf.core.logs.logger import logger
from dwarf.pipeline_core.case_builder import Case, Case_Builder
from dwarf.pipeline_core.config import ImagePreprocessingSpec
from dwarf.pipeline_core.config_loader import ExperimentConfig
from dwarf.pipeline_core.db import SQLite_DB
from dwarf.pipeline_core.execution import resolve_in_flight, resolve_start_method, resolve_workers
from dwarf.pipeline_core.worker import init_worker, run_case


def _run_loader(loader_config):
    Ds_Core(config=loader_config).main_loop()

def compute_run_id(config) -> str:
    cfg = {
        "schema_version": config.schema_version,
        "seed": config.experiment.seed,
        "repeats": config.experiment.repeats,
        "dataset": {
            "extensions": list(config.dataset.extensions),
            "recursive": config.dataset.recursive,
            "include": list(config.dataset.include),
            "exclude": list(config.dataset.exclude),
            "limit": config.dataset.limit,
            "shuffle": config.dataset.shuffle,
            "preprocessing": config.dataset.preprocessing.model_dump(mode="json"),
        },
        "watermark": config.watermark.model_dump(mode="json"),
        "embeddings": [e.model_dump(mode="json") for e in config.embeddings],
        "attack_scenarios": [s.model_dump(mode="json") for s in config.attack_scenarios],
        "metrics": [m.model_dump(mode="json") for m in config.metrics],
    }
    return sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False,).encode("utf-8")).hexdigest()[:12]

def canonical_preprocessing(spec: ImagePreprocessingSpec) -> str:
    """
    Каноническая строка preprocessing-спека.

    Гарантирует, что одинаковые по смыслу спеки дают одинаковую строку,
    независимо от порядка ключей в YAML и от того, был ли resize задан
    как tuple или list.
    """
    return json.dumps(
        spec.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )

def compute_image_id(
    relative_path: Path | str,
    preprocessing: ImagePreprocessingSpec,
) -> str:
    """
    Детерминированный id картинки в контексте эксперимента.

    Учитывает относительный путь и preprocessing-спек.
    Возвращает 16 hex-символов.
    """
    rel = Path(relative_path).as_posix()
    payload = f"{rel}|{canonical_preprocessing(preprocessing)}"
    return sha256(payload.encode("utf-8")).hexdigest()[:16]

def generate_file_manifest(
    source: Path,
    manifest_path: Path,
    extensions: tuple[str, ...],
    preprocessing: ImagePreprocessingSpec,
    limit: int | None = None,
    shuffle: bool = False,
    include: tuple[str, ...] = (),
    exclude: tuple[str, ...] = (),
    seed: int = 0,
) -> int:
    """Генерирует манифест картинок с учётом фильтров датасета.

    Порядок операций:
      1. Обход всех файлов в source с фильтром по extensions.
      2. Фильтр по include/exclude (glob-паттерны).
      3. Shuffle (детерминированный по seed) — если включён.
      4. Limit — обрезка до первых N.
      5. Атомарная запись в манифест.

    Args:
        source: корневая папка датасета.
        manifest_path: путь к выходному манифесту.
        extensions: разрешённые расширения (.png, .jpg, ...).
        preprocessing: спека для вычисления image_id.
        limit: максимум картинок в манифесте (None — все).
        shuffle: перемешать перед limit.
        include: glob-паттерны, которым должен соответствовать путь.
        exclude: glob-паттерны, которым путь соответствовать не должен.
        seed: seed для shuffle (для воспроизводимости между запусками).

    Returns:
        int: число записей в манифесте.
    """
    folder = Path(source).resolve()
    extensions_set = {ext.lower() for ext in extensions}

    candidates: list[tuple[Path, Path]] = []
    for file_path in folder.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in extensions_set:
            continue

        rel_path = file_path.relative_to(folder)

        if include and not any(rel_path.match(p) for p in include):
            continue
        if exclude and any(rel_path.match(p) for p in exclude):
            continue

        candidates.append((file_path, rel_path))

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(candidates)

    if limit is not None:
        candidates = candidates[:limit]

    tmp_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            for file_path, rel_path in candidates:
                image_id = compute_image_id(rel_path, preprocessing)
                entry = {
                    "image_id": image_id,
                    "path": str(file_path.resolve()),
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        os.replace(tmp_path, manifest_path)
        return len(candidates)
    except Exception as e:
        raise Dwarf_Exception(f"Failed to generate manifest: {e}") from e

class Parent:
    def __init__(self, config):
        self.config: ExperimentConfig = config
        self.db = SQLite_DB()

        self.queue_ready = mp.Queue(maxsize=2)
        self.queue_free = mp.Queue(maxsize=64)
        self.stop_event = mp.Event()

        self.pending: dict[Future, Case] = {}
        self.registry: dict[str, dict] = {}
        self.inflight_count = 0
        self.result_buffer = []
        self.retry_batches = {}
        self.unsaved_results_count = 0
        self.workers: int | None = None
        self.max_in_flight: int | None = None

        self.pending_lock = threading.Lock()
        self.registry_lock = threading.Lock()
        self.window_cond = threading.Condition()
        self.shutdown_event = threading.Event()

        self.loader: mp.Process | None = None
        self.executor: ProcessPoolExecutor | None = None

        self.run_id = compute_run_id(self.config)
        self.manifest_path = self.config.output.directory / "manifest.json"
        self.db_path = self.config.output.directory / self.config.output.database_filename

        self.shm_holders: dict[str, shared_memory.SharedMemory] = {}

    def setup(self):
        self.config.output.directory.mkdir(parents=True, exist_ok=True)

        count_of_images = generate_file_manifest(
            source=self.config.dataset.path,
            manifest_path=self.manifest_path,
            extensions=self.config.dataset.extensions,
            preprocessing=self.config.dataset.preprocessing,
            limit=self.config.dataset.limit,
            shuffle=self.config.dataset.shuffle,
            include=self.config.dataset.include,
            exclude=self.config.dataset.exclude,
            seed=self.config.experiment.seed,
        )

        self.db.open_db(self.db_path)
        self.db.init_db()

        self.db.parent_start(overwrite=self.config.output.overwrite)

        case_builder = Case_Builder(self.config)

        case_count = case_builder.count_cases()

        total_count = case_count * count_of_images

        if total_count > self.config.experiment.max_cases and not self.config.experiment.allow_large_plan:
            raise Dwarf_Exception(f"Too many cases: {total_count} > {self.config.experiment.max_cases}")

    def start_loader(self):
        loader_config = {
            "queue_ready": self.queue_ready,
            "queue_free": self.queue_free,
            "source": self.config.dataset.path,
            "image_color_model": self.config.dataset.preprocessing.color_mode,
            "image_geometry": self.config.dataset.preprocessing.resize,
            "manifest_file_path": self.manifest_path,
            "stop_event": self.stop_event,
            "run_id": self.run_id,
            "exif_transpose": self.config.dataset.preprocessing.exif_transpose,
            "resample": self.config.dataset.preprocessing.resample,
            "database_path": self.db_path,
            }
        self.loader = mp.Process(target=_run_loader, args=(loader_config,), daemon=False)
        self.loader.start()

    def start_pool(self):
        ctx = mp.get_context(resolve_start_method(self.config.execution))
        self.workers = resolve_workers(self.config.execution)
        self.max_in_flight = resolve_in_flight(self.config.execution, self.workers)
        self.executor = ProcessPoolExecutor(
            max_workers=self.workers,
            mp_context=ctx,
            initializer=init_worker,
            initargs=(
                list(self.config.plugins),
                self.config.watermark.model_dump(mode="json"),
                self.config.experiment.seed,
                self.config.output.model_dump(mode="json"),
                str(self.config.output.directory),
            ),
        )

    def mark_done(self, image_id: str):
        with self.registry_lock:
            entry = self.registry.get(image_id)
            if not entry:
                return

            entry["done"] += 1
            if entry["done"] != entry["expected"]:
                return

            del self.registry[image_id]
            shm = self.shm_holders.pop(image_id, None)

        if shm is not None:
            shm.close()

        while self.loader.is_alive() and not self.stop_event.is_set():
            try:
                self.queue_free.put(image_id, timeout=0.1)
                break
            except Full:
                continue

    def flush_results(self):
        new_failed_batch = None

        if self.result_buffer:
            batch = tuple(self.result_buffer)
            self.result_buffer = []

            if batch not in self.retry_batches:
                try:
                    self.db.flush_results(batch)
                except Exception as e:
                    new_failed_batch = batch
                    logger.error_with_context(e)

        for batch, attempts in list(self.retry_batches.items()):
            if attempts < 3:
                try:
                    self.db.flush_results(batch)
                except Exception as e:
                    attempts += 1
                    self.retry_batches[batch] = attempts
                    logger.error_with_context(e)
                else:
                    del self.retry_batches[batch]
                    continue

            if attempts >= 3:
                self.unsaved_results_count += len(batch)
                del self.retry_batches[batch]

        if new_failed_batch is not None:
            self.retry_batches[new_failed_batch] = 1

    def dispatcher_loop(self):
        while not self.shutdown_event.is_set():
            try:
                handle = self.queue_ready.get(timeout=1.0)
            except Empty:
                if not self.loader.is_alive():
                    logger.error_with_context(Dwarf_Exception("Loader is not alive"))
                    self.shutdown_event.set()
                    break
                continue

            if handle is DEAD_PILL:
                logger.log("Dispatcher got dead pill")
                self.shutdown_event.set()
                break

            image_id = handle["image_id"]

            shm = shared_memory.SharedMemory(name=handle["shm_name"])
            with self.registry_lock:
                self.shm_holders[image_id] = shm

            cb = Case_Builder(self.config)
            cb.init_case(handle)
            expected = cb.count_cases()

            with self.registry_lock:
                self.registry[image_id] = {"expected": expected, "done": 0}

            for case in cb.build_cases():
                status = self.db.try_register_case(case.case_id, image_id)
                if status in ("done", "failed"):
                    self.mark_done(image_id)
                    continue

                with self.window_cond:
                    while self.inflight_count >= self.max_in_flight:
                        self.window_cond.wait()

                fut = self.executor.submit(run_case, case)
                with self.pending_lock:
                    self.pending[fut] = case
                with self.window_cond:
                    self.inflight_count += 1

    def receiver_loop(self):
        while True:
            if self.shutdown_event.is_set():
                with self.pending_lock:
                    if not self.pending:
                        break

            with self.pending_lock:
                futures = list(self.pending.keys())

            if not futures:
                time.sleep(0.01)
                continue

            done, _ = wait(futures, timeout=0.1, return_when=FIRST_COMPLETED)

            for fut in done:
                with self.pending_lock:
                    case = self.pending.pop(fut, None)
                if case is None:
                    continue

                try:
                    result = fut.result()
                    status = result["status"]
                    metrics = result["metrics"]
                    error = result["error"]
                except Exception as e:
                    logger.error_with_context(e)
                    status = "failed"
                    error = str(e)
                    metrics = {}

                payload = json.dumps({"metrics": metrics}, ensure_ascii=False)
                # logger.log(f"metrics: {metrics} for case {case.case_id}")
                self.result_buffer.append((case.case_id, status, payload, error))

                self.mark_done(case.image_id)

                if len(self.result_buffer) >= 100:
                    self.flush_results()

                with self.window_cond:
                    self.inflight_count -= 1
                    self.window_cond.notify()
        self.flush_results()

    def run(self):
        logger.log("Starting pipeline!")
        self.setup()
        self.start_loader()
        self.start_pool()

        receiver = threading.Thread(target=self.receiver_loop, daemon=True)
        receiver.start()

        try:
            self.dispatcher_loop()
        finally:
            self.shutdown_event.set()
            self.stop_event.set()
            self.executor.shutdown(wait=True)
            receiver.join()
            with self.registry_lock:
                holders = list(self.shm_holders.values())
                self.shm_holders.clear()

            for shm in holders:
                shm.close()

            self.loader.join(timeout=60)

            while self.result_buffer or self.retry_batches:
                self.flush_results()

            self.db.close()

            if self.unsaved_results_count:
                logger.error_with_context(Dwarf_Exception(f"Ended pipeline with unsaved results: {self.unsaved_results_count}"))
            else:
                logger.log("Ended pipeline!")


