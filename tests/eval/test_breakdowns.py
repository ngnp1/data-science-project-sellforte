from benchmark.eval.breakdowns import CONFOUNDED_AXES, breakdown_by


def test_confounded_axes_are_named_exactly():
    """BENCHMARK.md: only noise_level is stratified within family. trend_p and
    market_spread are family-confounded on BOTH splits and must not be reported
    as axis effects."""
    assert CONFOUNDED_AXES == frozenset({"trend_p", "market_spread"})
    assert "noise_level" not in CONFOUNDED_AXES


def test_breakdown_groups_scenarios_by_axis_value():
    per_scenario = {
        "dev_001": {"event_level": {"n_tp": 1, "n_fp": 0, "n_fn": 0}},
        "dev_002": {"event_level": {"n_tp": 0, "n_fp": 1, "n_fn": 1}},
    }
    meta = {"dev_001": {"noise_level": "low"}, "dev_002": {"noise_level": "high"}}
    got = breakdown_by(per_scenario, meta, "noise_level")
    assert got["low"]["f1"] == 1.0
    assert got["high"]["f1"] == 0.0
    assert got["low"]["n_scenarios"] == 1


def test_confounded_axis_carries_a_warning_in_its_result():
    per_scenario = {"dev_001": {"event_level": {"n_tp": 1, "n_fp": 0, "n_fn": 0}}}
    meta = {"dev_001": {"trend_p": 0.0}}
    got = breakdown_by(per_scenario, meta, "trend_p")
    assert got["_warning"], "confounded axis reported without a warning"
    assert "confounded" in got["_warning"].lower()


# --- Event-level breakdowns: spec section 9 item 10's remaining three axes ---

import pandas as pd

from benchmark.eval.breakdowns import (EVENT_AXES, event_breakdown,
                                       event_breakdowns)
from benchmark.eval.matching import match_events
from benchmark.eval.model import Event


def ev(start, end, *, multiplier=None, country="DE", channel="TV",
       etype="natural_holdout", sid="dev_001"):
    return Event(sid=sid, country_code=country, channel=channel,
                 event_type=etype, multiplier=multiplier,
                 start=pd.Timestamp(start), end=pd.Timestamp(end))


def test_the_three_axes_meta_json_cannot_express_are_all_covered():
    """Spec section 9 item 10 lists duration, magnitude and near-zero versus
    exact-zero. `breakdown_by` buckets on scenario-level meta.json keys, which
    carry none of the three -- so they are implemented per truth EVENT, off
    `Event.n_days` and `Event.multiplier`."""
    assert set(EVENT_AXES) == {"duration", "magnitude", "zero_kind"}


def test_duration_bands_separate_short_events_from_long_ones():
    """A detector that finds long events and misses short ones must READ that
    way. Short-event recall is the number spec section 9 item 10 exists to
    expose, and a single overall recall hides it completely."""
    truth = [ev("2024-03-01", "2024-03-07", sid="dev_001"),    # 7 days
             ev("2024-03-01", "2024-03-21", sid="dev_002"),    # 21 days
             ev("2024-03-01", "2024-04-20", sid="dev_003"),    # 51 days
             ev("2024-03-01", "2024-07-01", sid="dev_004")]    # 123 days
    found_long_only = [t for t in truth if t.n_days >= 42]

    rows = {r["value"]: r for r in
            event_breakdown(match_events(truth, found_long_only), "duration")}
    assert rows["<14 days"]["recall"] == 0.0
    assert rows["14-41 days"]["recall"] == 0.0
    assert rows["42-89 days"]["recall"] == 1.0
    assert rows["90+ days"]["recall"] == 1.0
    assert rows["<14 days"]["n_events"] == 1


def test_duration_rows_come_back_in_duration_order_not_alphabetical():
    """Alphabetically `<14 days` sorts after `14-41`, `42-89` and `90+`, which
    would render the band table in an order that reads as nonsense."""
    truth = [ev("2024-03-01", "2024-03-07", sid="dev_001"),
             ev("2024-03-01", "2024-03-21", sid="dev_002"),
             ev("2024-03-01", "2024-04-20", sid="dev_003"),
             ev("2024-03-01", "2024-07-01", sid="dev_004")]
    rows = event_breakdown(match_events(truth, []), "duration")
    assert [r["value"] for r in rows] == ["<14 days", "14-41 days",
                                          "42-89 days", "90+ days"]


def test_magnitude_bands_separate_silence_from_a_level_change():
    truth = [ev("2024-03-01", "2024-03-21", multiplier=0.0, sid="dev_001"),
             ev("2024-03-01", "2024-03-21", multiplier=0.05, sid="dev_002"),
             ev("2024-03-01", "2024-03-21", multiplier=0.33, sid="dev_003"),
             ev("2024-03-01", "2024-03-21", multiplier=3.0, sid="dev_004")]
    rows = {r["value"]: r for r in
            event_breakdown(match_events(truth, truth[:1]), "magnitude")}
    assert rows["exact zero"]["recall"] == 1.0
    assert rows["near zero (0 < m < 0.1)"]["recall"] == 0.0
    assert rows["reduced (0.1 <= m < 1)"]["n_events"] == 1
    assert rows["amplified (m >= 1)"]["n_events"] == 1


def test_near_zero_and_exact_zero_are_scored_separately():
    """BENCHMARK.md: holdout magnitudes are drawn from [0, 0, 0.02, 0.05,
    0.08]. A detector that finds true silence but misses a channel running at
    5% of normal has exactly one weakness, and this is the axis that names it.
    Rolled together they would read as a single mediocre holdout recall."""
    truth = [ev("2024-03-01", "2024-03-21", multiplier=0.0, sid="dev_001"),
             ev("2024-03-01", "2024-03-21", multiplier=0.0, sid="dev_002"),
             ev("2024-03-01", "2024-03-21", multiplier=0.05, sid="dev_003"),
             ev("2024-03-01", "2024-03-21", multiplier=0.08, sid="dev_004")]
    finds_only_silence = [t for t in truth if t.multiplier == 0.0]

    rows = {r["value"]: r for r in
            event_breakdown(match_events(truth, finds_only_silence), "zero_kind")}
    assert rows["exact zero"]["recall"] == 1.0
    assert rows["near zero"]["recall"] == 0.0, (
        "near-zero misses are being credited to the exact-zero bucket")
    assert rows["exact zero"]["n_events"] == 2
    assert rows["near zero"]["n_events"] == 2


def test_zero_kind_excludes_events_that_are_neither():
    """A step change to 3x normal is not a zero of any kind and must not be
    filed under one."""
    truth = [ev("2024-03-01", "2024-03-21", multiplier=0.0, sid="dev_001"),
             ev("2024-03-01", "2024-03-21", multiplier=3.0, sid="dev_002"),
             ev("2024-03-01", "2024-03-21", multiplier=None, sid="dev_003")]
    rows = event_breakdown(match_events(truth, []), "zero_kind")
    assert {r["value"] for r in rows} == {"exact zero"}
    assert sum(r["n_events"] for r in rows) == 1


def test_event_breakdowns_credit_matches_by_identity_not_by_value():
    """Two truth events identical in content but in different buckets of a
    one-to-one assignment must not both be credited."""
    a = ev("2024-03-01", "2024-03-21", multiplier=0.0)
    b = ev("2024-03-01", "2024-03-21", multiplier=0.0)
    assert a == b and a is not b
    rows = event_breakdown(match_events([a, b], [a]), "magnitude")
    assert rows[0]["n_events"] == 2
    assert rows[0]["n_matched"] == 1
    assert rows[0]["recall"] == 0.5


def test_event_breakdowns_returns_every_axis():
    truth = [ev("2024-03-01", "2024-03-21", multiplier=0.0)]
    got = event_breakdowns(match_events(truth, truth))
    assert set(got) == set(EVENT_AXES)
