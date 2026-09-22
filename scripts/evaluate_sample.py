"""Evaluate the committed demonstration data without touching benchmark logs.

Run from the repository root: python -m scripts.evaluate_sample
This is a regression check on a known sample, not an unseen test score.
"""
from pathlib import Path
import json

import pandas as pd
import yaml

from benchmark.eval.adapter import to_event
from benchmark.eval.matching import match_events
from benchmark.eval.metrics import event_level, per_type
from benchmark.eval.truth import _group_pulses, _to_event
from detection.pipeline import run_detection

SAMPLE = Path(__file__).resolve().parents[1] / "synthetic_data_generator"


def evaluate_sample():
    raw = yaml.safe_load((SAMPLE / "events_config.yaml").read_text())
    truth = _group_pulses("sample", [r for r in raw if r["pattern_type"] == "channel_pulse"])
    truth.extend(_to_event("sample", r, event_type=r["pattern_type"])
                 for r in raw if r["pattern_type"] != "channel_pulse")
    events = run_detection(pd.read_csv(SAMPLE / "data/media.csv"),
                           pd.read_csv(SAMPLE / "data/sales.csv"), "sample")
    pred = [to_event(e) for e in events]
    matched = match_events(truth, pred)
    pulses = [m for m in matched.matches if m.truth.event_type == "channel_pulse"]
    return {
        "dataset": "committed demonstration sample (not held out)",
        "overall": event_level(truth, pred),
        "per_type": per_type(truth, pred),
        "pulse_components_match": bool(pulses) and all(
            m.truth.components == m.pred.components for m in pulses),
    }


if __name__ == "__main__":
    print(json.dumps(evaluate_sample(), indent=2))
