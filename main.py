"""Command-line entry point for validating and running DWARF experiments."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

from pydantic import ValidationError

from dwarf.core.dwarf_exceptions import Dwarf_Exception
from dwarf.pipeline_core.case_builder import Case_Builder
from dwarf.pipeline_core.config_loader import load_experiment_config
from dwarf.pipeline_core.pipeline import Parent


def _current_case_counts(config, database: Path) -> dict[str, int]:
    """Count expected cases in this manifest, excluding unrelated database history."""
    with closing(sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True)) as connection:
        connection.execute("CREATE TEMP TABLE selected_cases (case_id TEXT PRIMARY KEY) WITHOUT ROWID")
        builder = Case_Builder(config)
        with (config.output.directory / "manifest.json").open(encoding="utf-8") as manifest:
            for line in manifest:
                entry = json.loads(line)
                # Case IDs depend on the image ID and configuration, not pixels or IPC allocation.
                builder.init_case(
                    {"image_id": entry["image_id"], "shm_name": "status-check", "shape": (1, 1, 3), "dtype": "uint8"}
                )
                connection.executemany(
                    "INSERT INTO selected_cases VALUES (?)",
                    ((case.case_id,) for case in builder.build_cases()),
                )
        return dict(
            connection.execute(
                "SELECT COALESCE(c.status, 'missing'), COUNT(*) FROM selected_cases s "
                "LEFT JOIN cases c ON c.case_id = s.case_id GROUP BY COALESCE(c.status, 'missing')"
            )
        )


def main(argv: list[str] | None = None) -> int:
    """Validate the requested configuration and return a process exit status."""
    parser = argparse.ArgumentParser(description="Validate or run a DWARF experiment.")
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path(__file__).parent / "examples" / "pipeline" / "dct_jpeg.yaml",
        help="experiment YAML path (default: examples/pipeline/dct_jpeg.yaml)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="check YAML syntax and configuration schema without running an experiment",
    )
    arguments = parser.parse_args(argv)
    try:
        config = load_experiment_config(arguments.config)
    except (Dwarf_Exception, ValidationError, OSError) as error:
        parser.error(str(error))

    if arguments.validate_only:
        print("Configuration syntax and schema are valid.")
        return 0

    if config.execution.backend != "process":
        parser.error("the current executor only supports backend='process'; serial is not implemented")
    start_method = config.execution.start_method
    if start_method == "auto":
        start_method = "spawn"
    if start_method not in multiprocessing.get_all_start_methods():
        parser.error(f"start_method={start_method!r} is unavailable on this platform")

    try:
        pipeline = Parent(config)
        pipeline.run()
        database = config.output.directory / config.output.database_filename
        counts = _current_case_counts(config, database)
        completed = counts.get("done", 0)
        incomplete = sum(count for status, count in counts.items() if status != "done")
        if incomplete or pipeline.unsaved_results_count or not completed:
            print(
                f"Pipeline failed: {completed} completed, {incomplete} incomplete, "
                f"{pipeline.unsaved_results_count} unsaved cases. Results: {database}",
                file=sys.stderr,
            )
            return 1
        print(f"Completed {completed} cases. Results: {database}")
        return 0
    except Exception as error:
        print(f"Pipeline failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
