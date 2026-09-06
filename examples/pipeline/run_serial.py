"""Run one serial DWARF experiment from a YAML configuration."""

from __future__ import annotations

import argparse
from pathlib import Path

from dwarf.pipeline.pipeline import Pipeline
from dwarf.pipeline.runtime import ErasurePolicy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        default=Path(__file__).with_name("dct_jpeg_serial.yaml"),
        help="Path to a versioned experiment YAML file.",
    )
    parser.add_argument(
        "--erasure-policy",
        choices=[policy.value for policy in ErasurePolicy],
        default=ErasurePolicy.COUNT_AS_ERROR.value,
        help="How BER handles -1 extraction decisions.",
    )
    parser.add_argument(
        "--skip-integrity-check",
        action="store_true",
        help="Skip checksum verification when loading dataset files.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    pipeline = Pipeline.from_yaml(args.config)
    plan = pipeline.plan()
    print(plan.summary())
    print()

    result = pipeline.run(
        erasure_policy=args.erasure_policy,
        verify_integrity=not args.skip_integrity_check,
    )
    print("Execution completed.")
    print(f"Duration:           {result.duration_seconds:.3f} s")
    print(f"Work units:         {len(result.work_units)}")
    print(f"Cases:              {len(result.cases)}")
    print(f"Successful cases:   {result.successful_case_count}")
    print(f"Partial cases:      {result.partial_case_count}")
    print(f"Failed cases:       {result.failed_case_count}")
    print(f"Skipped cases:      {result.skipped_case_count}")
    print(f"Metric evaluations: {len(result.metrics)}")
    print(f"Failed metrics:     {result.failed_metric_count}")
    return 0 if result.failed_case_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
