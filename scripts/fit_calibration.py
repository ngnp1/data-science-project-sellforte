"""Fit the confidence calibration on the DEVELOPMENT split and freeze it.

Spec section 8: bin confidence, measure empirical precision per bin, fit a
monotone PAV mapping, freeze it before the final run. Fitting on the test
split would be grading your own exam, so this script reads the development
split only.

SELF-PROTECTING BY CONSTRUCTION: this script fits on `raw_confidence`, read
straight out of each DetectedEvent's `evidence` dict, never on
`detection_confidence`. `detection_confidence` is `apply_calibration(raw,
KNOTS)` (see detection/pipeline.py) -- once a frozen `detection/
calibration_fit.py` exists, that field IS the calibrated value, and fitting
PAV on top of it would silently calibrate an already-calibrated score. Raw
confidence is computed before that remap and is therefore identical whether
or not a frozen fit is present, so this script needs no "delete the fit
before running" step and produces the same knots whether run once, fresh, or
a hundred times over an already-frozen module. That invariant is pinned by
tests/scripts/test_fit_calibration.py, which fits twice in a row WITHOUT
deleting the module between runs and asserts identical knots.

Reading raw scores means importing detection.pipeline.run_detection directly
rather than going through benchmark.eval.adapter.detect: the adapter's Event
drops `evidence` on the way across (by design -- it has no place in the
harness's scored vocabulary), taking `raw_confidence` with it. This script
still needs the harness's matcher to learn which detections were correct, so
it converts each DetectedEvent to an Event with adapter.to_event for matching
purposes only, and reads the score off the original DetectedEvent.

Usage:
    PYTHONPATH=. synthetic_data_generator/.venv/bin/python scripts/fit_calibration.py
"""
from __future__ import annotations

import pathlib
import sys

import pandas as pd

from benchmark.eval.adapter import to_event
from benchmark.eval.matching import match_events
from benchmark.eval.truth import list_scenarios, load_truth
from benchmark.harness.runner import DATASETS_DIR, dataset_dir
from detection.calibrate import fit_pav, render_module
from detection.pipeline import run_detection
from detection import params

OUT = pathlib.Path("detection/calibration_fit.py")
SPLIT = "dev"


def _load_frames(sid: str):
    """Read a scenario's two input CSVs -- the only files a detector reads."""
    d = dataset_dir(SPLIT, sid, DATASETS_DIR)
    media = pd.read_csv(d / "media.csv", parse_dates=["date"])
    sales = pd.read_csv(d / "sales.csv", parse_dates=["date"])
    return media, sales


def fit(split: str = SPLIT):
    """Return (knots, n_detections, n_correct) fitted on `split`.

    Shared by main() and the regression test, so the test exercises the exact
    same code path the frozen module was produced by.
    """
    scores, correct = [], []
    for sid in list_scenarios(split):
        media, sales = _load_frames(sid)
        detected = run_detection(media, sales, sid)   # raw evidence intact
        pred = [to_event(d) for d in detected]         # for matching only
        truth = load_truth(split, sid)
        result = match_events(truth, pred, min_iou=0.5, strict=True)
        matched = {id(m.pred) for m in result.matches}
        for d, p in zip(detected, pred):
            raw = d.evidence.get("raw_confidence")
            if raw is None:
                continue
            scores.append(raw)
            correct.append(id(p) in matched)

    knots = fit_pav(scores, correct, n_bins=params.CALIBRATION_BINS)
    return knots, len(scores), sum(correct)


def main() -> int:
    knots, n, n_correct = fit()
    OUT.write_text(render_module(knots))
    print(f"fitted on {n} detections, {n_correct} correct")
    print(f"wrote {OUT} with {len(knots)} knots")
    for upper, value in knots:
        print(f"  <= {upper:.2f} -> {value:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
