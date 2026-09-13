"""Evaluate a detector against the DEVELOPMENT split. Free to run, as often as
you like -- that is what the dev split is for.

Every run appends to dev_history.jsonl, which becomes the development
trajectory in the final report: evidence that tuning happened where it was
permitted, and only there.

    python -m benchmark.eval.run_dev --detector module.path:function
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib
import json
from pathlib import Path

from benchmark.eval import truth as T
from benchmark.eval.report import render_markdown
from benchmark.eval.runner import evaluate_split

HISTORY = Path(__file__).resolve().parent / "dev_history.jsonl"


def source_hash(path: Path) -> str:
    """SHA-256 over every .py under `path`, so a report can name the exact code
    that produced it."""
    h = hashlib.sha256()
    if not path.is_dir():
        return "absent"
    for py in sorted(path.rglob("*.py")):
        h.update(py.relative_to(path).as_posix().encode())
        h.update(py.read_bytes())
    return h.hexdigest()


def load_detector(spec: str):
    """`module.path:function` -> the callable."""
    if ":" not in spec:
        raise SystemExit(f"--detector must be module.path:function, got {spec!r}")
    mod_name, func_name = spec.split(":", 1)
    return getattr(importlib.import_module(mod_name), func_name)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--detector", required=True,
                   help="module.path:function returning list[Event]")
    p.add_argument("--limit", type=int, default=None,
                   help="evaluate only the first N scenarios")
    p.add_argument("--label", default="",
                   help="short note recorded in dev_history.jsonl")
    p.add_argument("--out", default=None, help="write the Markdown report here")
    p.add_argument("--load-data", action="store_true",
                   help="pass media/sales frames to the detector")
    p.add_argument("--history", default=None,
                   help="path to the dev history JSONL "
                        f"(default: {HISTORY})")
    args = p.parse_args(argv)
    history_path = Path(args.history) if args.history else HISTORY

    detector = load_detector(args.detector)
    sids = T.list_scenarios("dev")
    if args.limit:
        sids = sids[: args.limit]

    results = evaluate_split(detector, "dev", sids=sids,
                             load_data=args.load_data)
    report = render_markdown(results)
    print(report)

    if args.out:
        Path(args.out).write_text(report)

    o = results["overall"]
    record = {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "label": args.label,
        "detector": args.detector,
        "n_scenarios": results["n_scenarios"],
        "source_hash": source_hash(Path(__file__).resolve().parents[2] / "detection"),
        "metrics": {"precision": o["precision"], "recall": o["recall"],
                    "f1": o["f1"], "mean_iou": results["iou"]["mean_iou"],
                    "null_fp_rate": results["null_fp_rate"]},
    }
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")

    print(f"\nAppended to {history_path.name} "
          f"(f1={o['f1']:.3f}, label={args.label or '-'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
