"""Fit the confidence calibration on the DEVELOPMENT split and freeze it.

Spec section 8: bin confidence, measure empirical precision per bin, fit a
monotone PAV mapping, freeze it before the final run. Fitting on the test
split would be grading your own exam, so this script reads the development
split only.

Usage:
    PYTHONPATH=. synthetic_data_generator/.venv/bin/python scripts/fit_calibration.py
"""
from __future__ import annotations

import pathlib
import sys

import pandas as pd

from benchmark.eval.matching import match_events
from benchmark.eval.truth import list_scenarios, load_truth
from benchmark.eval.adapter import detect
from benchmark.harness.runner import DATASETS_DIR, dataset_dir
from detection.calibrate import fit_pav, render_module
from detection import params

OUT = pathlib.Path("detection/calibration_fit.py")
SPLIT = "dev"


def _load_frames(sid: str):
    """Read a scenario's two input CSVs -- the only files a detector reads."""
    d = dataset_dir(SPLIT, sid, DATASETS_DIR)
    media = pd.read_csv(d / "media.csv", parse_dates=["date"])
    sales = pd.read_csv(d / "sales.csv", parse_dates=["date"])
    return media, sales


def main() -> int:
    scores, correct = [], []
    for sid in list_scenarios(SPLIT):
        media, sales = _load_frames(sid)
        pred = detect(media, sales, sid)
        truth = load_truth(SPLIT, sid)
        result = match_events(truth, pred, min_iou=0.5, strict=True)
        matched = {id(m.pred) for m in result.matches}
        for p in pred:
            if p.detection_confidence is None:
                continue
            scores.append(p.detection_confidence)
            correct.append(id(p) in matched)

    knots = fit_pav(scores, correct, n_bins=params.CALIBRATION_BINS)
    OUT.write_text(render_module(knots))
    print(f"fitted on {len(scores)} detections, {sum(correct)} correct")
    print(f"wrote {OUT} with {len(knots)} knots")
    for upper, value in knots:
        print(f"  <= {upper:.2f} -> {value:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
