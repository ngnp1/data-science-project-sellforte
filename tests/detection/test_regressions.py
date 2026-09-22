"""Small examples of the correctness failures found during sample review."""
import numpy as np
import pandas as pd
import pytest

from detection.calibrate import apply_calibration, fit_pav
from detection.pipeline import run_detection


def media(series):
    return pd.DataFrame([
        {"date": day, "country_code": country, "advertising_channel": channel,
         "media_investment": spend, "impressions": spend * 10, "clicks": spend}
        for (country, channel), values in series.items()
        for day, spend in zip(pd.date_range("2024-01-01", periods=len(values)), values)
    ])


def paused(n=300, start=60, end=90):
    values = np.full(n, 100.0)
    values[start:end] = 0
    return values


def test_unrelated_shutdown_does_not_destroy_a_pulse_train():
    values = np.full(300, 100.0)
    for start in (30, 58, 86):
        values[start:start + 14] = 0
    values[175:230] = 0
    events = run_detection(media({("FI", "A"): values, ("FI", "B"): paused(start=175, end=230)}))
    assert sorted(e.event_type for e in events) == ["channel_pulse", "dark_period"]
    pulse = next(e for e in events if e.event_type == "channel_pulse")
    assert len(pulse.components) == 3
    assert pulse.evidence["confidence_sub_scores"]["corroboration"] == 1


@pytest.mark.parametrize("bad", [np.nan, np.inf, -1, "unknown"])
def test_unknown_or_invalid_spend_is_rejected(bad):
    data = media({("FI", "A"): [100, bad, 100]})
    with pytest.raises(ValueError, match="finite, non-negative"):
        run_detection(data)


def test_spend_only_input_has_unknown_corroboration():
    data = media({("FI", "A"): paused(), ("FI", "B"): np.full(300, 100.0)})
    events = run_detection(data.drop(columns=["impressions", "clicks"]))
    assert len(events) == 1
    assert events[0].evidence["confidence_sub_scores"]["corroboration"] == 0.6


def test_missing_rows_are_flagged_and_not_claimed_as_valid():
    data = media({("FI", "A"): paused(), ("FI", "B"): np.full(300, 100.0)})
    data = data.loc[~((data.advertising_channel == "A") & (data.media_investment == 0))]
    events = run_detection(data)
    assert len(events) == 1
    assert events[0].validity == "suspect_data_gap"


def test_no_siblings_means_no_sibling_control():
    event, = run_detection(media({("FI", "A"): paused()}))
    assert event.evidence["control_available"] == "none"
    assert "No control group" in event.explanation


def test_one_live_peer_day_is_not_a_control():
    events = run_detection(media({("FI", "A"): paused(), ("DE", "A"): paused(end=89)}))
    event = next(e for e in events if e.country_code == "FI")
    assert event.evidence["control_available"] == "none"


def test_partial_peer_activity_is_not_a_global_pause():
    events = run_detection(media({("FI", "A"): paused(), ("DE", "A"): paused(),
                                  ("SE", "A"): paused(end=89)}))
    event = next(e for e in events if e.country_code == "FI")
    assert "global_pause" not in event.tags
    assert event.evidence["control_available"] == "none"


def test_permanent_step_has_sharpness_and_honest_end():
    values = np.full(300, 100.0)
    values[80:] = 300
    event, = run_detection(media({("FI", "A"): values}))
    assert event.event_type == "step_change"
    assert event.evidence["confidence_sub_scores"]["edge_sharpness"] == 1
    assert event.evidence["censored_end"]
    assert "true end is unknown" in event.explanation
    assert "not a probability" in event.explanation
    assert event.detection_confidence == event.evidence["raw_confidence"]
    from benchmark.eval.adapter import to_event
    assert to_event(event).detection_confidence is None


def test_regular_restart_is_not_a_permanent_budget_change():
    events = run_detection(media({("FI", "A"): paused(start=60, end=115)}))
    assert [e.event_type for e in events] == ["dark_period"]


def test_overlapping_changes_receive_a_cleanliness_penalty():
    step = np.full(300, 100.0)
    step[80:160] = 300
    events = run_detection(media({("FI", "A"): paused(), ("FI", "B"): step}))
    assert len(events) == 2
    assert all("confounded" in e.tags for e in events)
    assert all(e.evidence["informativeness_drivers"]["cleanliness"] < 1 for e in events)


def test_calibration_uses_the_same_bin_at_fit_and_application():
    knots = fit_pav([0.1, 0.5], [False, True], 2)
    assert apply_calibration(0.499, knots) == 0
    assert apply_calibration(0.5, knots) == 1
    assert apply_calibration(1, knots) == 1


def test_committed_sample_finds_all_six_events_without_extra_detections():
    from scripts.evaluate_sample import evaluate_sample
    result = evaluate_sample()
    assert result["overall"] == {
        "precision": 1.0, "recall": 1.0, "f1": 1.0,
        "n_tp": 6, "n_fp": 0, "n_fn": 0,
    }
    assert result["pulse_components_match"]
