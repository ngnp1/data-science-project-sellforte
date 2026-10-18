import numpy as np
import pandas as pd

from pulse.holdout import find_holdouts

DAYS = pd.date_range("2024-01-01", periods=200)
ON = pd.Series(True, index=DAYS)


def _ch(off=()):
    s = pd.Series(100 * np.random.default_rng(0).lognormal(0, 0.2, len(DAYS)), index=DAYS)
    for a, n in off:
        s.iloc[a:a + n] = 0.0
    return s, ON


def test_one_channel_off_while_others_run_is_a_holdout():
    market = {"A": _ch([(60, 42)]), "B": _ch(), "C": _ch()}
    [(ch, a, b, conf)] = find_holdouts(market, {})
    assert (ch, a, b) == ("A", DAYS[60], DAYS[101]) and 0.9 < conf <= 1


def test_not_a_holdout_when_others_stop_too_or_it_is_a_pulse_pause():
    assert find_holdouts({"A": _ch([(60, 42)]), "B": _ch([(60, 42)]), "C": _ch([(60, 42)])}, {}) == []
    assert find_holdouts({"A": _ch([(60, 42)]), "B": _ch()}, {"A": [(DAYS[60], DAYS[101])]}) == []


def test_single_channel_needs_all_but_one_off_and_three_channels():
    from pulse.holdout import find_single_channel
    three = {"A": _ch([(60, 42)]), "B": _ch([(60, 42)]), "C": _ch()}
    [(ch, a, b, conf)] = find_single_channel(three)
    assert (ch, a, b) == ("C", DAYS[60], DAYS[101]) and 0.9 < conf <= 1
    assert find_holdouts(three, {}) == []          # not also a holdout
    assert find_single_channel({"A": _ch([(60, 42)]), "B": _ch()}) == []   # 2 channels: a holdout


def test_dark_period_needs_every_channel_off():
    from pulse.holdout import find_dark_periods
    market = {"A": _ch([(60, 42)]), "B": _ch([(60, 42)])}
    [(a, b, conf)] = find_dark_periods(market)
    assert (a, b) == (DAYS[60], DAYS[101]) and 0.9 < conf <= 1
    assert find_holdouts(market, {}) == []
    assert find_dark_periods({"A": _ch([(60, 42)]), "B": _ch()}) == []
    never = (pd.Series(0.0, index=DAYS), ON)                # a channel that never spends
    assert len(find_dark_periods({"A": _ch([(60, 42)]), "B": never})) == 1


def test_staggered_launch_is_a_late_start_relative_to_another_market():
    from pulse.holdout import find_staggered_launches, market_p_off
    markets = {"X": {"A": _ch(), "B": _ch()}, "Y": {"A": _ch([(0, 60)]), "B": _ch()}}
    ps = {c: market_p_off(mk) for c, mk in markets.items()}
    [(country, ch, a, b, conf)] = find_staggered_launches(markets, ps)
    assert (country, ch, a, b) == ("Y", "A", DAYS[0], DAYS[59]) and 0.9 < conf <= 1
    assert find_holdouts(markets["Y"], {}, ps["Y"]) == []      # not also a holdout
    same = {"X": {"A": _ch([(0, 60)])}, "Y": {"A": _ch([(0, 60)])}}
    assert find_staggered_launches(same, {c: market_p_off(mk) for c, mk in same.items()}) == []
