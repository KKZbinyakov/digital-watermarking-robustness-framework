from os import cpu_count

from dwarf.core.dwarf_exceptions import Dwarf_Exception
from dwarf.pipeline_core.config import ExecutionSpec


def resolve_workers(execution: ExecutionSpec) -> int:
    workers = execution.workers
    if workers == "auto":
        cpu = cpu_count() or 1
        cpu = min(cpu, 64)
        return max(1, cpu - 1)
    return workers


def resolve_start_method(execution: ExecutionSpec) -> str:
    method = execution.start_method
    if method != "auto":
        return method
    return "spawn"


def resolve_in_flight(execution: ExecutionSpec, workers: int) -> int:
    if isinstance(execution.max_in_flight, int):
        return execution.max_in_flight
    if execution.max_in_flight == "auto":
        return workers * 2
    raise Dwarf_Exception(f"unexpected max_in_flight: {execution.max_in_flight!r}")
