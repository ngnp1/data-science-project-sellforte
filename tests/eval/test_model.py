import pandas as pd
import pytest

from benchmark.eval.model import (
    Event, NON_EVENT_TYPES, TYPE_MAP, MATCHABLE_TYPES,
)


def ev(**kw):
    base = dict(
        sid="dev_001", country_code="DE", channel="TV",
        event_type="natural_holdout",
        start=pd.Timestamp("2024-03-01"), end=pd.Timestamp("2024-03-10"),
    )
    base.update(kw)
    return Event(**base)


def test_n_days_is_inclusive():
    assert ev().n_days == 10
    assert ev(start=pd.Timestamp("2024-03-01"),
              end=pd.Timestamp("2024-03-01")).n_days == 1


def test_event_is_hashable_and_frozen():
    e = ev()
    assert hash(e) is not None
    with pytest.raises(Exception):
        e.start = pd.Timestamp("2024-01-01")


def test_negative_controls_are_exactly_the_two_intended_types():
    assert NON_EVENT_TYPES == frozenset({"ramp_block", "intermittent_baseline"})


def test_type_map_covers_every_truth_type_the_generator_emits():
    """These are the pattern_types benchmark/spec/events.py can produce."""
    produced = {
        "dark_period", "single_channel", "natural_holdout", "step_change",
        "channel_pulse", "staggered_launch", "global_pause",
        "ramp_block", "intermittent_baseline",
    }
    assert produced <= set(TYPE_MAP)


def test_global_pause_maps_to_dark_period():
    """BENCHMARK.md: truth records global_pause as a pattern_type, but the
    detector labels the regime dark_period and applies global_pause as a tag."""
    assert TYPE_MAP["global_pause"] == "dark_period"


def test_identity_mappings_are_identity():
    for t in ["dark_period", "single_channel", "natural_holdout",
              "step_change", "channel_pulse", "staggered_launch"]:
        assert TYPE_MAP[t] == t


def test_negative_controls_are_not_matchable():
    assert MATCHABLE_TYPES.isdisjoint(NON_EVENT_TYPES)
    assert "dark_period" in MATCHABLE_TYPES


def test_tags_and_components_default_empty_and_are_tuples():
    e = ev()
    assert e.tags == () and e.components == ()


def test_scores_default_to_none_so_truth_events_carry_no_confidence():
    e = ev()
    assert e.detection_confidence is None and e.informativeness is None
    scored = ev(detection_confidence=0.9, informativeness=0.4)
    assert scored.detection_confidence == 0.9
