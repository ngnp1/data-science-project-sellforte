"""Evaluate a detector on the committed demonstration data, without touching
benchmark logs.

Run from the repository root:

    python -m scripts.evaluate_sample                      # our rule-based library
    python -m scripts.evaluate_sample --detector team      # any registered detector

This is a regression check on a known sample, not an unseen test score.
"""
from pathlib import Path
import argparse
import json

import pandas as pd
import yaml

from benchmark.eval.adapter import for_detector
from benchmark.eval.matching import match_events
from benchmark.eval.metrics import event_level, per_type
from benchmark.eval.truth import _group_pulses, _to_event

SAMPLE = Path(__file__).resolve().parents[1] / "synthetic_data_generator"


def evaluate_sample(detector: str = "rule_based"):
    raw = yaml.safe_load((SAMPLE / "events_config.yaml").read_text())
    truth = _group_pulses("sample", [r for r in raw if r["pattern_type"] == "channel_pulse"])
    truth.extend(_to_event("sample", r, event_type=r["pattern_type"])
                 for r in raw if r["pattern_type"] != "channel_pulse")
    pred = for_detector(detector)(pd.read_csv(SAMPLE / "data/media.csv"),
                                  pd.read_csv(SAMPLE / "data/sales.csv"), "sample")
    matched = match_events(truth, pred)
    pulses = [m for m in matched.matches if m.truth.event_type == "channel_pulse"]
    return {
        "dataset": "committed demonstration sample (not held out)",
        "detector": detector,
        "overall": event_level(truth, pred),
        "per_type": per_type(truth, pred),
        "pulse_components_match": bool(pulses) and all(
            m.truth.components == m.pred.components for m in pulses),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--detector", default="rule_based",
                        help="key of a registered detector (default: rule_based)")
    print(json.dumps(evaluate_sample(parser.parse_args().detector), indent=2, default=str))
