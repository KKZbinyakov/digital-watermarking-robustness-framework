import csv
import json
import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory


def write_metrics(db, images, output):
    writer = csv.writer(output, delimiter=";")
    writer.writerow(
        (
            "case_id",
            "image_id",
            "image_path",
            "metric_id",
            "value",
            "path_id",
            "operations_json",
            "case_status",
        )
    )

    # В старых БД таблицы путей ещё нет.
    has_paths = db.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'operation_paths'").fetchone()

    paths = {}
    if has_paths:
        paths = dict(db.execute("SELECT path_id, operations_json FROM operation_paths"))

    statuses = Counter()
    measurements = 0
    cases = db.execute("SELECT case_id, image_id, status, result_json FROM cases ORDER BY case_id")

    for case_id, image_id, status, payload in cases:
        statuses[status] += 1
        if payload is None:
            continue

        for metric_id, entry in json.loads(payload)["metrics"].items():
            if isinstance(entry, dict):
                value = entry["value"]
                path_id = entry["path_id"]
                operations = paths[path_id]
            else:
                # Старый формат: metrics[metric_id] содержит само число.
                value, path_id, operations = entry, None, None

            writer.writerow(
                (
                    case_id,
                    image_id,
                    images.get(image_id, ""),
                    metric_id,
                    value,
                    path_id,
                    operations,
                    status,
                )
            )
            measurements += 1

    return statuses, measurements


def export_metrics(run_dir: Path, database: str = "results.sqlite3"):
    images = {}
    with (run_dir / "manifest.json").open(encoding="utf-8") as manifest:
        for line in manifest:
            image = json.loads(line)
            images[image["image_id"]] = image["path"]

    uri = (run_dir / database).resolve().as_uri() + "?mode=ro"
    target = run_dir / "metrics.csv"

    with closing(sqlite3.connect(uri, uri=True)) as db, TemporaryDirectory(dir=run_dir) as tmp:
        db.execute("BEGIN")
        draft = Path(tmp) / target.name
        with draft.open("w", encoding="utf-8-sig", newline="") as output:
            statuses, measurements = write_metrics(db, images, output)
        draft.replace(target)

    print(f"CSV: {target.resolve()}")
    print(
        f"Кейсы: done={statuses['done']}, failed={statuses['failed']}, "
        f"dispatching={statuses['dispatching']}. Измерений: {measurements}."
    )
