"""Validate the harness against detectors whose correct scores are known in
advance. Without this, a broken metric would silently flatter every later
result and nothing would catch it."""
import pandas as pd
import pytest

from benchmark.eval import detectors_for_testing as D
from benchmark.eval import truth as T
from benchmark.eval.metrics import evaluate_scenario

DEV_WITH_EVENTS = [sid for sid in T.list_scenarios("dev")
                   if T.load_truth("dev", sid)]


def score(sid, detector):
    truth = T.load_truth("dev", sid)
    pred = detector(None, None, sid)
    return evaluate_scenario(truth, pred)


def test_there_are_dev_scenarios_with_events_to_test_against():
    assert len(DEV_WITH_EVENTS) >= 30


@pytest.mark.parametrize("sid", DEV_WITH_EVENTS)
def test_perfect_oracle_scores_exactly_one(sid):
    """THE harness validation. If this fails anywhere, the harness cannot
    recognise a correct detector and every later number is meaningless."""
    got = score(sid, D.perfect_oracle)["event_level"]
    assert got["precision"] == 1.0, sid
    assert got["recall"] == 1.0, sid
    assert got["f1"] == 1.0, sid


def test_perfect_oracle_scores_one_on_pulse_scenarios_specifically():
    """Called out separately because ungrouped pulse truth is the documented
    trap: a perfect detector would score zero on a third of the test events."""
    pulse_sids = [sid for sid in DEV_WITH_EVENTS
                  if any(e.event_type == "channel_pulse"
                         for e in T.load_truth("dev", sid))]
    assert pulse_sids, "no dev pulse scenarios found"
    for sid in pulse_sids:
        got = score(sid, D.perfect_oracle)["per_type"]["channel_pulse"]
        assert got["f1"] == 1.0, sid


def test_ungrouped_pulse_oracle_fails_exactly_as_documented():
    """The negative control for the grouping fix: emitting one event per
    off-window instead of one grouped event must NOT score 1.0. If this passes
    at 1.0, the loader is not grouping and the trap is still live."""
    pulse_sids = [sid for sid in DEV_WITH_EVENTS
                  if any(e.event_type == "channel_pulse"
                         for e in T.load_truth("dev", sid))]
    sid = pulse_sids[0]
    got = score(sid, D.ungrouped_pulse_oracle)["per_type"]["channel_pulse"]
    assert got["f1"] < 1.0, \
        "ungrouped pulses scored perfectly -- grouping is not being exercised"


@pytest.mark.parametrize("sid", DEV_WITH_EVENTS[:10])
def test_never_detect_scores_zero_recall_and_no_false_positives(sid):
    got = score(sid, D.never_detect)["event_level"]
    assert got["recall"] == 0.0 and got["n_fp"] == 0


def test_never_detect_is_perfect_on_null_scenarios():
    """A detector that reports nothing has a flawless false-positive rate. This
    is exactly why precision alone is not a sufficient headline."""
    nulls = [sid for sid in T.list_scenarios("dev")
             if T.load_meta("dev", sid)["family"] == "null"]
    assert nulls
    for sid in nulls:
        assert D.never_detect(None, None, sid) == []


def test_detect_everything_is_heavily_penalised_on_nulls():
    nulls = [sid for sid in T.list_scenarios("dev")
             if T.load_meta("dev", sid)["family"] == "null"]
    sid = nulls[0]
    pred = D.detect_everything(None, None, sid)
    assert len(pred) > 0
    got = evaluate_scenario([], pred)["event_level"]
    assert got["n_fp"] == len(pred)
    assert got["precision"] == 0.0


@pytest.mark.parametrize("shift", [1, 3, 10])
def test_shifted_oracle_degrades_iou_monotonically(shift):
    sid = DEV_WITH_EVENTS[0]
    base = score(sid, D.perfect_oracle)["iou"]["mean_iou"]
    got = score(sid, D.shifted_oracle(shift))["iou"]["mean_iou"]
    assert got <= base


def test_shifted_oracle_shows_up_in_boundary_error():
    sid = DEV_WITH_EVENTS[0]
    got = score(sid, D.shifted_oracle(3))["boundary"]
    assert got["start_median"] == 3.0


def test_perfect_oracle_produces_a_perfectly_calibrated_reliability_curve():
    """Everything in the top bin, and everything in it correct."""
    sid = DEV_WITH_EVENTS[0]
    curve = score(sid, D.perfect_oracle)["reliability"]
    assert curve, "oracle produced no reliability curve"
    assert all(b["empirical_precision"] == 1.0 for b in curve)
    assert curve[-1]["bin_hi"] == 1.0


def test_half_confident_oracle_is_visibly_miscalibrated():
    """Low-confidence detections here are just as correct as high-confidence
    ones, so the low bin must show high empirical precision. If the curve just
    echoed the confidence values back, this would not show up."""
    sid = DEV_WITH_EVENTS[0]
    curve = score(sid, D.half_confident_oracle)["reliability"]
    low = [b for b in curve if b["bin_hi"] <= 0.3]
    assert low, "no low-confidence bin produced"
    assert all(b["empirical_precision"] == 1.0 for b in low)


def test_operating_curve_keeps_everything_at_cut_zero():
    sid = DEV_WITH_EVENTS[0]
    curve = score(sid, D.perfect_oracle)["operating"]
    at_zero = [c for c in curve if c["cut"] == 0.0][0]
    assert at_zero["recall"] == 1.0


def test_operating_curve_drops_recall_as_the_cut_rises():
    sid = DEV_WITH_EVENTS[0]
    curve = score(sid, D.half_confident_oracle)["operating"]
    recalls = [c["recall"] for c in sorted(curve, key=lambda c: c["cut"])]
    assert recalls[0] >= recalls[-1]


def test_wrong_channel_oracle_is_caught_by_channel_accuracy_not_by_f1():
    """The whole reason accuracy uses a relaxed match: under the strict match
    this detector's errors become FN+FP and channel accuracy would read 100%."""
    sid = next(s for s in DEV_WITH_EVENTS
               if any(e.channel for e in T.load_truth("dev", s)))
    got = score(sid, D.wrong_channel_oracle)
    assert got["accuracy"]["channel_accuracy"] < 1.0
    assert got["accuracy"]["market_accuracy"] == 1.0
    assert got["event_level"]["f1"] < 1.0
