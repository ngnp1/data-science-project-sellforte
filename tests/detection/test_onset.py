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
    """Literal 5, not MIN_DAYS - 1. Sizing the input from the constant under
    test makes the input move with it, so the test passes for every value the
    constant could take and asserts nothing about the gate."""
    assert params.MIN_DAYS > 5, "test input must sit below the threshold"
    assert find_onset(s([0.0] * 5 + [100.0] * 200)) is None


def test_a_dormant_period_at_min_days_is_an_onset():
    """The other half of the boundary: at the threshold it must fire."""
    o = find_onset(s([0.0] * params.MIN_DAYS + [100.0] * 200))
    assert o is not None and o.dormant_days == params.MIN_DAYS


def test_a_short_tail_run_is_not_a_discontinuation():
    """Same gate on the other edge, and likewise sized with a literal."""
    assert params.MIN_DAYS > 5
    assert find_discontinuation(s([100.0] * 200 + [0.0] * 5)) is None


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


def test_brief_activity_is_not_a_confirmed_launch():
    for active_days in (1, 13):
        assert find_onset(s([0] * 30 + [100] * active_days + [0] * 60)) is None
    assert find_onset(s([0] * 30 + [100] * 14)) is not None


def test_missing_activation_rows_do_not_confirm_a_launch():
    spend = s([0] * 30 + [100] * 30)
    present = pd.Series(True, index=spend.index)
    present.iloc[35] = False
    assert find_onset(spend, present) is None


def test_missing_history_is_not_observed_inactivity():
    spend = s([0] * 30 + [100] * 30)
    present = pd.Series(True, index=spend.index)
    present.iloc[:25] = False
    assert find_onset(spend, present) is None
    present.iloc[24] = True
    present.iloc[23] = True
    onset = find_onset(spend, present)
    assert onset.observed_inactive_days == 7
    assert onset.active_confirmation_days == 14
