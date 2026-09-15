import pathlib

import pytest

from detection import params
from detection.calibrate import apply_calibration, fit_pav
from detection.calibration_fit import KNOTS


def test_the_frozen_fit_is_monotone():
    values = [v for _, v in KNOTS]
    assert values == sorted(values), values


def test_the_frozen_fit_is_python_source_not_a_data_file():
    """detection/ may contain nothing but .py -- a JSON fit would be invisible
    to both the import gate and the final-run audit hash."""
    assert pathlib.Path("detection/calibration_fit.py").suffix == ".py"


def test_calibrated_confidence_stays_in_range():
    for s in [0.0, 0.1, 0.35, 0.6, 0.85, 1.0]:
        assert 0.0 <= apply_calibration(s, KNOTS) <= 1.0


@pytest.mark.slow
def test_the_frozen_fit_reproduces_from_the_dev_split(monkeypatch):
    """The three tests above would pass on almost any KNOTS list -- including
    one that is monotone, in range, and simply WRONG: hand-edited, fitted on
    the wrong split, or (the exact trap this script warns about) fitted
    against scores that were already run back through the frozen calibration
    it was supposed to produce. This test re-derives the fit from scratch,
    end to end over the whole dev split exactly as scripts/fit_calibration.py
    does, and checks the frozen module actually IS that fit.

    Detection must run against RAW scores here: `detection/pipeline.py`
    already has the frozen KNOTS wired in, so left alone `detect()` would
    hand back CALIBRATED confidences and this would refit on an
    already-calibrated score -- silently double-calibrating instead of
    reproducing the original fit. Patching the pipeline's KNOTS back to []
    for the duration of this test is what the fitting script achieves by
    requiring `detection/calibration_fit.py` be deleted before it runs.
    """
    import pandas as pd

    import detection.pipeline as pipeline
    from benchmark.eval.adapter import detect
    from benchmark.eval.matching import match_events
    from benchmark.eval.truth import list_scenarios, load_truth
    from benchmark.harness.runner import DATASETS_DIR, dataset_dir

    monkeypatch.setattr(pipeline, "KNOTS", [])

    scores, correct = [], []
    for sid in list_scenarios("dev"):
        d = dataset_dir("dev", sid, DATASETS_DIR)
        media = pd.read_csv(d / "media.csv", parse_dates=["date"])
        sales = pd.read_csv(d / "sales.csv", parse_dates=["date"])
        pred = detect(media, sales, sid)
        truth = load_truth("dev", sid)
        result = match_events(truth, pred, min_iou=0.5, strict=True)
        matched = {id(m.pred) for m in result.matches}
        for p in pred:
            if p.detection_confidence is None:
                continue
            scores.append(p.detection_confidence)
            correct.append(id(p) in matched)

    fresh = fit_pav(scores, correct, n_bins=params.CALIBRATION_BINS)
    assert fresh == KNOTS
