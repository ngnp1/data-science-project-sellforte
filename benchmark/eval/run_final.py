"""Evaluate a detector against the SEALED HOLD-OUT TEST SPLIT.

This is the one sanctioned reader of test-split ground truth, and running it is
meant to be a deliberate, recorded act rather than a convenience. It therefore:

  * refuses to run without an explicit --finalize flag;
  * verifies the test split's seal before reading anything, so a tampered or
    regenerated benchmark cannot quietly produce a number;
  * appends an immutable record to final_runs.jsonl -- timestamp, a SHA-256
    over every *.py file under detection/ (paths and contents; non-Python
    files are not covered), the sealed spec_hash, and the metrics. The record
    is appended even if rendering or writing --out fails, because by then the
    sealed split has already been read.

Run it ONCE, after the algorithms are final. A second run is possible but is
permanently visible in final_runs.jsonl, and the report is expected to say so.

    python -m benchmark.eval.run_final --detector module.path:function --finalize
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

from benchmark.eval import truth as T
from benchmark.eval.report import render_markdown
from benchmark.eval.run_dev import load_detector, source_hash
from benchmark.eval.runner import evaluate_split
from benchmark.harness import seal

FINAL_RUNS = Path(__file__).resolve().parent / "final_runs.jsonl"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _prior_run_count() -> int:
    """Number of final-run records already in FINAL_RUNS, or 0 if the file is
    absent or empty. Read fresh each time so it reflects FINAL_RUNS at the
    moment it's called -- including when a test monkeypatches the path."""
    if FINAL_RUNS.is_file() and FINAL_RUNS.read_text().strip():
        return len(FINAL_RUNS.read_text().splitlines())
    return 0


def _repeat_run_banner(run_index: int, prior_count: int) -> str:
    """A Markdown banner naming this as a repeat final run, so a reader of the
    rendered report (or the --out file) sees it without having to go count
    lines in final_runs.jsonl. Empty when this is the first run, so it
    composes cleanly with the report either way."""
    if not prior_count:
        return ""
    return (
        f"> **REPEAT FINAL RUN -- this is final run #{run_index}.** "
        f"{prior_count} prior final run(s) are already recorded in "
        f"`final_runs.jsonl`. This is NOT the first final evaluation of the "
        f"sealed test split.\n\n"
    )


def compose_report(results: dict, run_index: int, prior_count: int) -> str:
    """The exact string `main` prints and writes to --out: the repeat-run
    banner (empty on a first run) followed by the rendered report.

    Extracted so the composition itself is testable. Asserting
    `(banner + body).startswith(banner)` in a test proves nothing about this
    program; calling this function and checking what comes out does.
    """
    return _repeat_run_banner(run_index, prior_count) + render_markdown(results)


def build_record(results: dict, args, run_index: int, marker: dict) -> dict:
    """The audit line for final_runs.jsonl. Built BEFORE the report is
    rendered so the record can be appended no matter what happens during
    rendering or --out."""
    o = results["overall"]
    return {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "run_index": run_index,
        "detector": args.detector,
        "detection_source_hash": source_hash(PROJECT_ROOT / "detection"),
        "sealed_spec_hash": marker["spec_hash"],
        "sealed_at": marker["sealed_at"],
        "n_scenarios": results["n_scenarios"],
        "metrics": {"precision": o["precision"], "recall": o["recall"],
                    "f1": o["f1"], "mean_iou": results["iou"]["mean_iou"],
                    "null_fp_rate": results["null_fp_rate"]},
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--detector", required=True,
                   help="module.path:function returning list[Event]")
    p.add_argument("--finalize", action="store_true",
                   help="required. Confirms the algorithms are frozen and you "
                        "intend to read the sealed test split.")
    p.add_argument("--out", default=None, help="write the Markdown report here")
    p.add_argument("--load-data", action="store_true",
                   help="pass media/sales frames to the detector")
    args = p.parse_args(argv)

    if not args.finalize:
        print("ERROR: run_final reads the SEALED hold-out test split. Pass "
              "--finalize to confirm the algorithms are frozen.\n"
              "Use `python -m benchmark.eval.run_dev` while iterating.",
              file=sys.stderr)
        return 2

    ok, problems = seal.verify_seal("test")
    if not ok:
        print("ERROR: the test split's seal does not verify. Refusing to "
              "produce a number from a benchmark that has changed:",
              file=sys.stderr)
        for problem in problems[:10]:
            print(f"  {problem}", file=sys.stderr)
        return 3

    marker = json.loads(
        (PROJECT_ROOT / "benchmark/datasets/test/SEALED").read_text())

    prior_count = _prior_run_count()
    run_index = prior_count + 1

    if prior_count:
        print(f"*** NOTE: final_runs.jsonl already has {prior_count} record(s). "
              f"This is not the first final evaluation, and the report must "
              f"say so. ***\n", file=sys.stderr)

    results = evaluate_split(load_detector(args.detector), "test",
                             load_data=args.load_data)

    # From here on the sealed split HAS BEEN READ. The audit record is what
    # makes that irreversible act visible to the next run, so it is built
    # first and appended in a `finally`: a mistyped --out (an unwritable
    # directory, a typo'd path) used to raise between printing the metrics and
    # appending the record, leaving the numbers on the terminal, the seal
    # spent, and final_runs.jsonl empty -- so the NEXT run would record
    # run_index 1 and its report would claim to be the first final evaluation.
    # Spec sections 4.3 and 10 exist to make exactly that impossible.
    record = build_record(results, args, run_index, marker)
    try:
        report = compose_report(results, run_index, prior_count)
        print(report)
        if args.out:
            Path(args.out).write_text(report)
    finally:
        with FINAL_RUNS.open("a") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
        print(f"\nRecorded in {FINAL_RUNS.name}. detection/ hash "
              f"{record['detection_source_hash'][:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
