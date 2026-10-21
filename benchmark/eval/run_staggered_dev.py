"""Evaluate existing development data and report launch detection and readiness.

No generation, test evaluation, tuning, or tracked history writes occur here.
Example: python -m benchmark.eval.run_staggered_dev --exclude dev_034 dev_035
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import datetime as dt
import json
from pathlib import Path
import platform
import subprocess
import time

from benchmark.eval import truth as T
from benchmark.eval.adapter import to_event
from benchmark.eval.matching import match_events
from benchmark.eval.metrics import boundary_error, event_level, iou_stats
from benchmark.eval.run_dev import source_hash
from benchmark.eval.runner import evaluate_split
from benchmark.harness.runner import DATASETS_DIR, _is_complete
from benchmark.spec.scenarios import build_split
from detection.compose.staggered_launch import summarize_staggered_launches
from detection.pipeline import run_detection


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / ".cache" / "staggered-launch-dev"


def _json_compatible(value):
    # The existing evaluator uses tuple keys for its type-confusion matrix.
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(_json_compatible(value), indent=2,
                               default=str, allow_nan=False) + "\n", encoding="utf-8")


def _git(*args) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _compact(event) -> dict:
    return {"sid": event.sid, "country_code": event.country_code,
            "channel": event.channel, "start": str(event.start.date()),
            "end": str(event.end.date()), "pattern_id": event.pattern_id}


def _render_report(metadata: dict, detection: dict, readiness: dict,
                   launches: list[dict]) -> str:
    lines = [
        "# Staggered-launch development evaluation", "",
        f"Completed evaluation of {metadata['n_scenarios']} selected development scenarios.",
        f"Excluded: {', '.join(metadata['excluded_scenarios']) or 'none'}.", "",
        f"UTC timestamp: {metadata['timestamp']}. Commit: {metadata['commit'] or 'unavailable'}.",
        f"Working tree dirty: {metadata['working_tree_dirty']}.",
        f"Detection source SHA-256: `{metadata['source_hash']}`.", "",
        "## Detection correctness", "", "| Metric | Value |", "|---|---|",
    ]
    lines.extend(f"| {key} | {value} |" for key, value in detection["metrics"].items())
    boundary = detection["boundary_days"]
    lines += ["", f"Matched launch intervals: {boundary['n']}.",
              "Absolute interval boundary errors (days; matched launches only):", "",
              "```json", json.dumps(boundary, indent=2), "```", "",
              "## Missed launches and false alarms", "", "```json",
              json.dumps({key: detection[key] for key in ("missed_launches", "false_launches")}, indent=2),
              "```", "", "## Readiness screening", "", "```json",
              json.dumps(readiness, indent=2), "```", "",
              "Reason counts may overlap. Launch labels establish detection correctness, "
              "not readiness accuracy. Readiness is not proof of a causal comparison.", "",
              "## Analysis priority", "",
              "Ready launches first, then review cases, known rejections, and unassessed launches. "
              "Existing informativeness and confidence break ties. Every launch is retained.", "",
              "| Rank | Scenario | Market | Channel | Launch date | Readiness |",
              "|---|---|---|---|---|---|"]
    lines.extend(f"| {r['rank']} | {r['sid']} | {r['country_code']} | {r['channel']} | "
                 f"{r['launch_date']} | {r['readiness']} |" for r in launches)
    lines += ["", "This ordering is a triage policy, not validated ranking accuracy or "
              "a probability of usefulness. Detection, classification, readiness rules, "
              "and shared scores were not changed. No test-split evaluation was run.", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exclude", nargs="*", default=[], metavar="SID",
                        help="explicit development scenario IDs to leave outside the evaluation")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="output directory (default: .cache/staggered-launch-dev)")
    args = parser.parse_args(argv)

    # Select from the configured split, not only directories present on disk.
    # An entirely missing scenario must not silently disappear from coverage.
    configured = sorted(s.sid for s in build_split("dev"))
    excluded = sorted(set(args.exclude))
    unknown = sorted(set(excluded) - set(configured))
    if unknown:
        parser.error(f"Unknown development exclusions: {', '.join(unknown)}")
    selected = [sid for sid in configured if sid not in excluded]
    if not selected:
        parser.error("No development scenarios selected")
    incomplete = [sid for sid in selected if not _is_complete("dev", sid, DATASETS_DIR)]
    if incomplete:
        parser.error("Incomplete development inputs/answers: " + ", ".join(incomplete)
                     + ". Generate separately or explicitly exclude; nothing was evaluated.")

    started = time.perf_counter()
    timestamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    raw_events, predictions = [], []

    def detector(media, sales, sid):
        events = run_detection(media, sales, sid)
        raw_events.extend(events)
        adapted = [to_event(event) for event in events]
        predictions.extend(adapted)
        print(f"Evaluated {sid}: {len(events)} events", flush=True)
        return adapted

    results = evaluate_split(detector, "dev", root=DATASETS_DIR, sids=selected, load_data=True)
    launch_truth = [event for sid in selected for event in T.load_truth("dev", sid, DATASETS_DIR)
                    if event.event_type == "staggered_launch"]
    launch_pred = [event for event in predictions if event.event_type == "staggered_launch"]
    matches = match_events(launch_truth, launch_pred)
    detection = {
        "metrics": event_level(launch_truth, launch_pred),
        "boundary_days": boundary_error(matches), "iou": iou_stats(matches),
        "missed_launches": [_compact(event) for event in matches.unmatched_truth],
        "false_launches": [_compact(event) for event in matches.unmatched_pred],
        "matched_launches": [{"truth": _compact(match.truth), "prediction": _compact(match.pred),
                              "onset_error_days": abs((match.pred.end - match.truth.end).days)}
                             for match in matches.matches],
    }
    launches = summarize_staggered_launches(raw_events)
    readiness = {
        "outcomes": dict(Counter(row["readiness"] for row in launches)),
        "reason_codes": dict(Counter(code for row in launches for code in row["reason_codes"])),
        "peer_check_reasons": dict(Counter(
            check["code"] for event in raw_events if event.event_type == "staggered_launch"
            for peer in event.evidence.get("launch_verification", {}).get("peers", [])
            for check in peer["checks"].values() if check["status"] != "pass")),
    }
    git_status = _git("status", "--porcelain")
    metadata = {
        "timestamp": timestamp, "evaluation_status": "completed", "split": "dev",
        "n_scenarios": len(selected), "configured_scenarios": len(configured),
        "selected_scenarios": selected, "excluded_scenarios": excluded,
        "commit": _git("rev-parse", "HEAD"),
        "working_tree_dirty": None if git_status is None else bool(git_status),
        "source_hash": source_hash(ROOT / "detection"),
        "evaluation_source_hash": source_hash(ROOT / "benchmark" / "eval"),
        "detector": "detection.pipeline:run_detection via benchmark.eval.adapter:to_event",
        "python_version": platform.python_version(),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "metrics.json", {"detection": detection, "readiness": readiness,
                                       "development_evaluation": results})
    _write_json(out / "launch-summary.json", launches)
    _write_json(out / "run-metadata.json", metadata)
    (out / "predictions.jsonl").write_text("".join(
        json.dumps(asdict(event), default=str, allow_nan=False) + "\n" for event in raw_events),
        encoding="utf-8")
    (out / "report.md").write_text(_render_report(metadata, detection, readiness, launches), encoding="utf-8")
    with (out / "run-history.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({**metadata, "launch_metrics": detection["metrics"],
                                 "readiness_outcomes": readiness["outcomes"]}, allow_nan=False) + "\n")
    print(f"Completed {len(selected)} scenarios: {json.dumps(detection['metrics'])}", flush=True)
    print(f"Reports: {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
