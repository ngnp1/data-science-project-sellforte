import pandas as pd
import dataclasses

import pytest

from detection.model import EVENT_TYPES, DetectedEvent


def ev(**kw):
    base = dict(sid="dev_001", country_code="DE", channel="TV",
                event_type="natural_holdout",
                start=pd.Timestamp("2024-03-01"), end=pd.Timestamp("2024-03-10"))
    base.update(kw)
    return DetectedEvent(**base)


def test_n_days_is_inclusive():
    assert ev().n_days == 10
    assert ev(end=pd.Timestamp("2024-03-01")).n_days == 1


def test_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        ev().start = pd.Timestamp("2024-01-01")


def test_optional_fields_default_sensibly():
    e = ev()
    assert e.magnitude_ratio is None
    assert e.components == () and e.tags == ()
    assert e.evidence == {}


def test_evidence_is_per_instance_not_shared():
    """A mutable default shared across instances would silently merge the
    evidence of every detection in a run."""
    a, b = ev(), ev()
    a.evidence["x"] = 1
    assert b.evidence == {}



def test_event_types_are_matchable_by_the_harness():
    """The detector cannot emit a label the harness is unable to score.

    Emitting a label outside MATCHABLE_TYPES produces events that can never
    match any truth row — false positives by construction.
    """
    from benchmark.eval.model import MATCHABLE_TYPES
    assert EVENT_TYPES <= MATCHABLE_TYPES


def test_channel_is_none_for_market_wide_events():
    e = ev(channel=None, event_type="dark_period")
    assert e.channel is None


def test_scores_default_to_unscored_not_to_a_number():
    """An unscored event must be distinguishable from a low-confidence one.
    Defaulting to 0.0 would put every unscored detection in the bottom
    reliability bin and make an unscored detector look badly calibrated
    rather than unscored."""
    e = ev()
    assert e.detection_confidence is None
    assert e.informativeness is None


def test_validity_defaults_to_ok_with_no_reasons():
    e = ev()
    assert e.validity == "ok"
    assert e.validity_reasons == ()


def test_scores_round_trip_through_the_constructor():
    e = DetectedEvent(
        sid="dev_001", country_code="DE", channel="TV",
        event_type="dark_period",
        start=pd.Timestamp("2024-03-01"), end=pd.Timestamp("2024-04-01"),
        detection_confidence=0.87, informativeness=0.42,
        validity="suspect_tracking_loss",
        validity_reasons=("spend zero while impressions continue",),
    )
    assert e.detection_confidence == 0.87
    assert e.informativeness == 0.42
    assert e.validity == "suspect_tracking_loss"
    assert e.validity_reasons == ("spend zero while impressions continue",)


def test_detection_model_does_not_import_the_harness():
    """The black-box boundary: detection/ must never reach into benchmark.eval."""
    import inspect

    import detection.model as m
    src = inspect.getsource(m)
    assert "benchmark.eval" not in src
    assert "ground_truth" not in src
