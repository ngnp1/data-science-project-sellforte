import pandas as pd

from detection import params
from detection.primitives.onset import find_discontinuation, find_onset


def s(values, start="2024-01-01"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values)),
                     dtype=float)


def test_a_dormant_start_is_an_onset():
    o = find_onset(s([0.0] * 90 + [100.0] * 200))
    assert o is not None
    assert o.dormant_days == 90
    assert o.first_active == pd.Timestamp("2024-03-31")


def test_a_channel_running_from_day_one_has_no_onset():
    assert find_onset(s([100.0] * 200)) is None


def test_a_dormant_period_shorter_than_min_days_is_not_an_onset():
    assert find_onset(s([0.0] * (params.MIN_DAYS - 1) + [100.0] * 200)) is None


def test_a_mid_series_holdout_is_not_an_onset():
    assert find_onset(s([100.0] * 50 + [0.0] * 30 + [100.0] * 50)) is None


def test_a_channel_that_never_runs_has_no_onset():
    """All-zero is a market that does not use the channel, not a launch."""
    assert find_onset(s([0.0] * 200)) is None


def test_a_run_at_the_series_end_is_a_discontinuation():
    d = find_discontinuation(s([100.0] * 200 + [0.0] * 40))
    assert d is not None and d.touches_end and d.n_days == 40


def test_a_channel_running_to_the_end_has_no_discontinuation():
    assert find_discontinuation(s([100.0] * 200)) is None
