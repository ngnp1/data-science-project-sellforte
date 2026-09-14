import pandas as pd
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
    with pytest.raises(Exception):
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


def test_event_types_cover_everything_the_composer_can_emit():
    assert EVENT_TYPES == frozenset({
        "dark_period", "single_channel", "natural_holdout",
        "step_change", "channel_pulse", "staggered_launch",
    })


def test_channel_is_none_for_market_wide_events():
    e = ev(channel=None, event_type="dark_period")
    assert e.channel is None


def test_detection_model_does_not_import_the_harness():
    """The black-box boundary: detection/ must never reach into benchmark.eval."""
    import inspect

    import detection.model as m
    src = inspect.getsource(m)
    assert "benchmark.eval" not in src
    assert "ground_truth" not in src
