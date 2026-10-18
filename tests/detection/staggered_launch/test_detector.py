from detection import params
from detection.compose.staggered_launch.detector import find_staggered_launches
from tests.detection.test_cross_market import build


def test_staggered_launch_is_emitted_once_per_LATE_market():
    """The harness's truth is PER-MARKET, so a country-less panel event scores
    zero. benchmark/eval/README.md states the fan-out is the detector's job."""
    p = build({
        "DE": {"TV": [100] * 200},
        "AT": {"TV": [0] * 60 + [100] * 140},
        "CH": {"TV": [0] * 120 + [100] * 80},
    })
    launches = find_staggered_launches(p, "dev_test")
    assert {e.country_code for e in launches} == {"AT", "CH"}
    assert all(e.event_type == "staggered_launch" for e in launches)
    assert all(e.country_code is not None for e in launches)


def test_a_launch_window_runs_from_series_start_to_the_day_before_activity():
    p = build({"DE": {"TV": [100] * 200}, "AT": {"TV": [0] * 60 + [100] * 140}})
    at = [e for e in find_staggered_launches(p, "dev_test")
          if e.country_code == "AT"][0]
    assert at.start == p.dates[0]
    assert at.end == p.dates[59]


def test_the_first_market_to_launch_is_the_comparison_group():
    """Nothing was live anywhere during the first market's own dormancy, so
    that dormancy has no peer evidence behind it and is not reported."""
    p = build({
        "DE": {"TV": [0] * 30 + [100] * 170},
        "AT": {"TV": [0] * 120 + [100] * 80},
    })
    assert [e.country_code for e in find_staggered_launches(p, "dev_test")] \
        == ["AT"]


def test_a_market_that_never_ran_the_channel_is_not_a_day_one_witness():
    """CH never bought TV, so it cannot be the market that proves TV was
    runnable from day one. Counting it as one would make DE -- itself a late
    starter, and the real comparison group here -- look like a launch too."""
    p = build({
        "DE": {"TV": [0] * 30 + [100] * 170, "Radio": [50] * 200},
        "AT": {"TV": [0] * 120 + [100] * 80, "Radio": [50] * 200},
        "CH": {"TV": [0] * 200, "Radio": [50] * 200},
    })
    assert [e.country_code for e in find_staggered_launches(p, "dev_test")] \
        == ["AT"]


def test_markets_starting_together_are_not_a_staggered_launch():
    """Coincidental start-up jitter is not a rollout."""
    p = build({
        "DE": {"TV": [0] * 30 + [100] * 170},
        "AT": {"TV": [0] * 32 + [100] * 168},
    })
    # The spread here is two days, and the rollout above spans ninety.
    assert 2 < params.ONSET_SPREAD <= 90
    assert find_staggered_launches(p, "dev_test") == []


def test_a_channel_live_everywhere_from_day_one_is_not_a_launch():
    p = build({"DE": {"TV": [100] * 200}, "AT": {"TV": [100] * 200}})
    assert find_staggered_launches(p, "dev_test") == []


def test_a_channel_no_market_ever_ran_is_not_a_launch():
    """A channel booked at zero everywhere is a channel the advertiser does
    not use. It has no onset anywhere, and must not raise."""
    p = build({
        "DE": {"TV": [100] * 200, "Print": [0] * 200},
        "AT": {"TV": [100] * 200, "Print": [0] * 200},
    })
    assert find_staggered_launches(p, "dev_test") == []


def test_each_launch_needs_its_own_separation_from_the_earliest_market():
    p = build({
        "DE": {"TV": [0] * 30 + [100] * 170},
        "AT": {"TV": [0] * 37 + [100] * 163},
        "CH": {"TV": [0] * 44 + [100] * 156},
    })
    events = find_staggered_launches(p, "dev_test")
    assert [e.country_code for e in events] == ["CH"]
    assert events[0].evidence["onset_separation_days"] == 14


def test_a_rejected_spike_is_not_a_day_one_comparison():
    p = build({"DE": {"TV": [100] + [0] * 199},
               "AT": {"TV": [0] * 60 + [100] * 140}})
    assert find_staggered_launches(p, "dev_test") == []
