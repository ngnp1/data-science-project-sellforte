"""Truth-loader tests run against the DEV split only. Reading test-split
ground truth during development would defeat the entire project."""
import pandas as pd
import pytest

from benchmark.eval import truth as T
from benchmark.eval.model import NON_EVENT_TYPES


def test_lists_all_dev_scenarios():
    sids = T.list_scenarios("dev")
    assert len(sids) == 45
    assert sids[0] == "dev_001" and sids[-1] == "dev_045"


def test_meta_carries_the_breakdown_axes():
    m = T.load_meta("dev", "dev_001")
    assert m["sid"] == "dev_001"
    for key in ["family", "noise_level", "trend_p", "market_spread",
                "n_countries", "n_channels", "n_days", "years", "n_events"]:
        assert key in m


def test_null_scenarios_load_zero_events():
    for sid in T.list_scenarios("dev"):
        if T.load_meta("dev", sid)["family"] == "null":
            assert T.load_truth("dev", sid) == []


def test_intervals_are_inclusive_and_derived_from_end_day():
    """BENCHMARK.md: end_date in the CSV is slice-exclusive EXCEPT where it is
    clamped at the series end. We sidestep that entirely by taking end_day from
    scenario.json, where it is always the unclamped exclusive day number."""
    for sid in T.list_scenarios("dev"):
        scen = T.load_scenario("dev", sid)
        by_id = {e["pattern_id"]: e for e in scen["events"]}
        for e in T.load_truth("dev", sid) + T.load_non_events("dev", sid):
            if e.components:
                # Grouped channel_pulse events carry the FIRST row's pattern_id
                # (by module design) but span every row in the group, so a
                # naive 1:1 pattern_id lookup here would compare the group's
                # overall end against a single row's end_day and fail on any
                # scenario with a multi-row pulse train. That grouping
                # behaviour is independently and exactly verified by
                # test_pulse_rows_are_grouped_into_one_event_per_country_channel
                # (hardcoded end=2024-09-11 for dev_019), so it is out of
                # scope for this single-row check.
                continue
            for pid in ([e.pattern_id] if e.pattern_id else []):
                if pid in by_id:
                    src = by_id[pid]
                    expected_start = T.START_DATE + pd.Timedelta(days=src["start_day"])
                    expected_end = T.START_DATE + pd.Timedelta(days=src["end_day"] - 1)
                    assert e.start == expected_start, sid
                    assert e.end == expected_end, sid


def test_censored_end_reaches_the_last_day_of_the_series():
    """dev_037 is the censored_end edge case and is one of the four scenarios
    where the CSV's end_date is already inclusive. Getting this wrong shortens
    the window by a day and the edge case stops testing what it was built for."""
    meta = T.load_meta("dev", "dev_037")
    last_day = T.START_DATE + pd.Timedelta(days=meta["n_days"] - 1)
    events = T.load_truth("dev", "dev_037")
    assert any(e.end == last_day for e in events), \
        f"no event reaches {last_day.date()}"


def test_pulse_rows_are_grouped_into_one_event_per_country_channel():
    """THE critical reconciliation. dev_019 has four 14-day pulse rows; the
    detector emits ONE grouped event. Ungrouped, a perfect pulse detector scores
    IoU ~0.118 against any single row and matches nothing."""
    events = T.load_truth("dev", "dev_019")
    pulses = [e for e in events if e.event_type == "channel_pulse"]
    assert len(pulses) == 1, f"expected 1 grouped pulse, got {len(pulses)}"
    p = pulses[0]
    assert p.country_code == "DE" and p.channel == "Affiliate"
    assert p.start == pd.Timestamp("2024-05-16")
    assert p.end == pd.Timestamp("2024-09-11")   # end_day - 1, inclusive
    assert len(p.components) == 4
    assert all(isinstance(a, pd.Timestamp) and isinstance(b, pd.Timestamp)
               for a, b in p.components)


def test_pulse_grouping_is_per_country_channel_not_per_scenario():
    """Two pulse trains on different channels must stay two events."""
    for sid in T.list_scenarios("dev"):
        events = T.load_truth("dev", sid)
        pulses = [e for e in events if e.event_type == "channel_pulse"]
        keys = [(e.country_code, e.channel) for e in pulses]
        assert len(keys) == len(set(keys)), f"{sid}: duplicate pulse group"


def test_negative_controls_are_excluded_from_matchable_truth():
    for sid in T.list_scenarios("dev"):
        for e in T.load_truth("dev", sid):
            assert e.event_type not in NON_EVENT_TYPES, sid


def test_negative_controls_are_available_separately():
    """They must be loadable, because a detection inside one is a false
    positive and the reporting needs to say so."""
    hits = [sid for sid in T.list_scenarios("dev")
            if T.load_non_events("dev", sid)]
    assert hits, "no scenario exposed any negative-control window"
    for sid in hits:
        for e in T.load_non_events("dev", sid):
            assert e.event_type in NON_EVENT_TYPES


def test_global_pause_is_relabelled_to_dark_period_and_tagged():
    found = False
    for sid in T.list_scenarios("dev"):
        scen = T.load_scenario("dev", sid)
        if any(e["pattern_type"] == "global_pause" for e in scen["events"]):
            found = True
            events = T.load_truth("dev", sid)
            gp = [e for e in events if "global_pause" in e.tags]
            assert gp, sid
            assert all(e.event_type == "dark_period" for e in gp), sid
    assert found, "no dev scenario carries a global_pause"


def test_staggered_launch_is_fanned_out_per_market():
    """Convention chosen once and stated: the panel-level detector event is
    fanned out to one per market, so the section 9 matcher stays unchanged."""
    found = False
    for sid in T.list_scenarios("dev"):
        launches = [e for e in T.load_truth("dev", sid)
                    if e.event_type == "staggered_launch"]
        if launches:
            found = True
            assert all(e.country_code is not None for e in launches), sid
    assert found, "no dev scenario carries a staggered_launch"


def test_multiplier_is_carried_from_scenario_json_not_the_csv():
    """ground_truth.csv blanks multiplier for everything except step_change, so
    magnitude breakdowns must come from scenario.json."""
    seen = set()
    for sid in T.list_scenarios("dev"):
        for e in T.load_truth("dev", sid):
            if e.event_type == "natural_holdout" and e.multiplier is not None:
                seen.add(e.multiplier)
    assert 0.0 in seen, "no exact-zero holdout magnitude reached the loader"
    assert any(0 < m < 0.1 for m in seen), "no near-zero magnitude"


def test_every_matchable_event_has_a_positive_length():
    for sid in T.list_scenarios("dev"):
        for e in T.load_truth("dev", sid):
            assert e.n_days >= 1, (sid, e.pattern_id)


def test_loader_never_touches_the_test_split_truth(monkeypatch):
    """A guard against the single worst accident this project can have.

    Patches both builtins.open (belt) and pathlib.Path.read_text (suspenders).
    truth.py reads exclusively via Path.read_text, which does NOT go through
    builtins.open -- it calls io.open internally without ever invoking the
    name `open` that monkeypatch.setattr(builtins, "open", ...) replaces. A
    guard that only patches builtins.open here is therefore incapable of ever
    firing against this loader's actual code path and passes vacuously
    regardless of what the loader does. Verified empirically: pointing this
    loader at a directory path containing "test_truth" fired no AssertionError
    under a builtins.open-only patch, and did fire once Path.read_text was
    also patched. See task-2-report.md for the full experiment.
    """
    import builtins
    from pathlib import Path

    real_open = builtins.open
    real_read_text = Path.read_text

    def guarded_open(path, *a, **kw):
        if "test_truth" in str(path):
            raise AssertionError(f"loader touched sealed truth: {path}")
        return real_open(path, *a, **kw)

    def guarded_read_text(self, *a, **kw):
        if "test_truth" in str(self):
            raise AssertionError(f"loader touched sealed truth: {self}")
        return real_read_text(self, *a, **kw)

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    T.load_truth("dev", "dev_019")
