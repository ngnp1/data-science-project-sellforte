"""Evaluate a detector against the SEALED HOLD-OUT TEST SPLIT.

This is the one sanctioned reader of test-split ground truth, and running it is
meant to be a deliberate, recorded act rather than a convenience. It therefore:

  * refuses to run without an explicit --finalize flag;
  * verifies the test split's seal before reading anything, so a tampered or
    regenerated benchmark cannot quietly produce a number;
  * appends an immutable record to final_runs.jsonl -- timestamp, the SHA-256
    of the entire detection/ tree, the sealed spec_hash, and the metrics.

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

    if FINAL_RUNS.is_file() and FINAL_RUNS.read_text().strip():
        n = len(FINAL_RUNS.read_text().splitlines())
        print(f"*** NOTE: final_runs.jsonl already has {n} record(s). "
              f"This is not the first final evaluation, and the report must "
              f"say so. ***\n", file=sys.stderr)

    results = evaluate_split(load_detector(args.detector), "test",
                             load_data=args.load_data)
    report = render_markdown(results)
    print(report)
    if args.out:
        Path(args.out).write_text(report)

    o = results["overall"]
    record = {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "detector": args.detector,
        "detection_source_hash": source_hash(PROJECT_ROOT / "detection"),
        "sealed_spec_hash": marker["spec_hash"],
        "sealed_at": marker["sealed_at"],
        "n_scenarios": results["n_scenarios"],
        "metrics": {"precision": o["precision"], "recall": o["recall"],
                    "f1": o["f1"], "mean_iou": results["iou"]["mean_iou"],
                    "null_fp_rate": results["null_fp_rate"]},
    }
    with FINAL_RUNS.open("a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")

    print(f"\nRecorded in {FINAL_RUNS.name}. detection/ hash "
          f"{record['detection_source_hash'][:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
