import pytest

from benchmark.spec import events

REQUIRED_KEYS = {"pattern_id", "pattern_type", "country", "channel",
                 "start_day", "end_day", "multiplier", "description"}


def _check_shape(entries, n_days=730):
    assert entries, "builder produced no entries"
    for e in entries:
        assert set(e) == REQUIRED_KEYS, f"bad keys: {set(e) ^ REQUIRED_KEYS}"
        assert 0 <= e["start_day"] < e["end_day"] <= n_days
        assert isinstance(e["pattern_id"], str) and e["pattern_id"]


def test_dark_targets_every_channel():
    got = events.dark("DE", start=100, length=42)
    _check_shape(got)
    assert len(got) == 1
    assert got[0]["pattern_type"] == "dark_period"
    assert got[0]["channel"] == "ALL"
    assert got[0]["multiplier"] == 0
    assert got[0]["end_day"] - got[0]["start_day"] == 42


def test_single_channel_names_the_channel_that_stays_on():
    got = events.single_channel("FI", keep="Google Search", start=250, length=30)
    _check_shape(got)
    assert got[0]["pattern_type"] == "single_channel"
    assert got[0]["channel"] == "Google Search"


def test_holdout_names_the_channel_that_goes_off():
    got = events.holdout("AT", "Facebook", start=400, length=42)
    _check_shape(got)
    assert got[0]["pattern_type"] == "natural_holdout"
    assert got[0]["channel"] == "Facebook"
    assert got[0]["multiplier"] == 0


def test_holdout_supports_near_zero_spend():
    got = events.holdout("AT", "Facebook", start=400, length=42, multiplier=0.04)
    assert got[0]["multiplier"] == 0.04


def test_step_records_its_multiplier():
    for mult in [0.33, 1.5, 3, 5]:
        got = events.step("CH", "Google Search", start=500, length=56, multiplier=mult)
        _check_shape(got)
        assert got[0]["pattern_type"] == "step_change"
        assert got[0]["multiplier"] == mult


def test_pulse_emits_one_entry_per_off_window_with_unique_ids():
    got = events.pulse("DE", "Radio", starts=[40, 68, 96], length=14)
    _check_shape(got)
    assert len(got) == 3
    assert len({e["pattern_id"] for e in got}) == 3
    assert all(e["pattern_type"] == "channel_pulse" for e in got)
    assert [e["start_day"] for e in got] == [40, 68, 96]


def test_launch_is_anchored_at_day_zero():
    got = events.launch("US", "Instagram", length=90)
    _check_shape(got)
    assert got[0]["start_day"] == 0
    assert got[0]["pattern_type"] == "staggered_launch"


def test_global_pause_covers_every_country():
    got = events.global_pause(["DE", "AT", "CH"], start=300, length=21)
    _check_shape(got)
    assert len(got) == 3
    assert {e["country"] for e in got} == {"DE", "AT", "CH"}
    assert all(e["pattern_type"] == "global_pause" for e in got)


def test_ramp_is_a_negative_control_made_of_increasing_blocks():
    got = events.ramp("SE", "TV", start=200, block=10,
                      multipliers=[1.2, 1.4, 1.6, 1.8, 2.0])
    _check_shape(got)
    assert len(got) == 5
    assert [e["multiplier"] for e in got] == [1.2, 1.4, 1.6, 1.8, 2.0]
    # Contiguous blocks, no gaps and no overlaps.
    assert [(e["start_day"], e["end_day"]) for e in got] == [
        (200, 210), (210, 220), (220, 230), (230, 240), (240, 250)]
    assert got[0]["pattern_type"] in events.NON_EVENT_TYPES


def test_intermittent_is_a_negative_control_of_many_short_gaps():
    got = events.intermittent("NL", "TikTok", n=20, off_len=3, period=14, start=30)
    _check_shape(got)
    assert len(got) == 20
    assert all(e["end_day"] - e["start_day"] == 3 for e in got)
    assert got[0]["pattern_type"] in events.NON_EVENT_TYPES


def test_negative_controls_are_exactly_the_two_intended_types():
    assert events.NON_EVENT_TYPES == frozenset({"ramp_block", "intermittent_baseline"})


def test_ids_are_unique_across_a_mixed_scenario():
    entries = (events.dark("DE", 100, 42)
               + events.holdout("AT", "Facebook", 400, 42)
               + events.pulse("DE", "Radio", [40, 68, 96], 14)
               + events.step("CH", "Google Search", 500, 56, 3))
    ids = [e["pattern_id"] for e in entries]
    assert len(ids) == len(set(ids))


def test_length_beyond_the_series_is_rejected():
    with pytest.raises(ValueError, match="exceeds"):
        events.dark("DE", start=700, length=60, n_days=730)
