import pandas as pd

from detection import params
from detection.primitives.zero_runs import find_off_runs, notable_runs, off_mask


def s(values, start="2024-01-01"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values)),
                     dtype=float)


def run_of(level, n_on, n_off, n_on2):
    return s([level] * n_on + [0.0] * n_off + [level] * n_on2)


def test_exact_zero_is_off():
    assert list(off_mask(s([100.0, 0.0, 100.0]))) == [False, True, False]


def test_near_zero_is_off_at_the_rho_threshold():
    """The benchmark injects holdouts at 0.02-0.08x, so near-zero must count."""
    # .iloc, not [1]: this pandas version (3.0.5) no longer falls back to
    # positional indexing for an integer key on a DatetimeIndex Series -- it
    # raises KeyError instead. off_mask's return value is unchanged; this is
    # purely how the test addresses position 1.
    assert off_mask(s([100.0, 3.0, 100.0])).iloc[1]       # 0.03 * level
    assert not off_mask(s([100.0, 50.0, 100.0])).iloc[1]  # ordinary low-spend day


def test_a_run_shorter_than_min_days_is_not_notable():
    runs = find_off_runs(run_of(100.0, 20, params.MIN_DAYS - 1, 20))
    assert len(runs) == 1 and not runs[0].notable


def test_a_run_at_min_days_is_notable_when_the_series_is_otherwise_never_off():
    runs = find_off_runs(run_of(100.0, 20, params.MIN_DAYS, 20))
    assert len(runs) == 1 and runs[0].notable


def test_an_intermittent_channel_needs_a_much_longer_run():
    """THE guard. A flighting channel whose normal gaps are 3 days must not
    report every gap; a fixed 7-day rule would fire on all of them."""
    values = ([100.0] * 11 + [0.0] * 3) * 8           # 8 routine 3-day gaps
    values += [100.0] * 11 + [0.0] * 10 + [100.0] * 20  # one 10-day gap
    runs = find_off_runs(s(values))
    routine = [r for r in runs if r.n_days == 3]
    assert len(routine) == 8
    assert not any(r.notable for r in routine)
    long_run = [r for r in runs if r.n_days == 10]
    assert len(long_run) == 1 and long_run[0].notable


def test_the_intermittent_guard_floor_is_three_times_the_p90_gap():
    """Spec section 7: 'A flighting channel whose normal gaps are 3 days needs
    at least 9 off-days to register.' With 8 routine 3-day gaps, p90 of the
    other off-runs is 3, so the floor is RUN_RATIO * 3 = 9 -- an 8-day run
    falls one day short of that floor and must NOT be notable, even though it
    is longer than every routine gap and clears MIN_DAYS on its own."""
    values = ([100.0] * 11 + [0.0] * 3) * 8          # 8 routine 3-day gaps
    values += [100.0] * 11 + [0.0] * 8 + [100.0] * 20  # one 8-day gap, one
                                                        # short of the 9-day floor
    runs = find_off_runs(s(values))
    long_run = [r for r in runs if r.n_days == 8]
    assert len(long_run) == 1 and not long_run[0].notable


def test_a_never_otherwise_off_channel_needs_only_min_days():
    """Complement of the test above: the ratio rule must not punish a series
    that has no gap distribution to compare against."""
    runs = notable_runs(run_of(100.0, 100, params.MIN_DAYS, 100))
    assert len(runs) == 1


def test_depth_is_one_for_an_exact_zero_and_lower_for_near_zero():
    exact = find_off_runs(run_of(100.0, 20, 10, 20))[0]
    assert exact.depth == 1.0
    near = find_off_runs(s([100.0] * 20 + [5.0] * 10 + [100.0] * 20))[0]
    assert 0.9 < near.depth < 1.0


def test_kind_distinguishes_exact_zero_from_near_zero():
    assert find_off_runs(run_of(100.0, 20, 10, 20))[0].kind == "exact_zero"
    assert find_off_runs(s([100.0] * 20 + [5.0] * 10 + [100.0] * 20))[0].kind \
        == "near_zero"


def test_a_missing_row_is_reported_as_missing_not_as_a_pause():
    """A missing row and a zero-spend row are the same number after reindexing.
    Only the presence mask can tell them apart, and the difference is a real
    dark period versus broken ingestion."""
    values = [100.0] * 20 + [0.0] * 10 + [100.0] * 20
    present = pd.Series(True, index=pd.date_range("2024-01-01",
                                                  periods=len(values)))
    present.iloc[20:30] = False
    r = find_off_runs(s(values), present=present)[0]
    assert r.kind == "missing"


def test_runs_carry_inclusive_dates():
    r = find_off_runs(run_of(100.0, 3, 4, 3))[0]
    assert r.start == pd.Timestamp("2024-01-04")
    assert r.end == pd.Timestamp("2024-01-07")
    assert r.n_days == 4


def test_a_run_at_the_series_start_is_flagged_censored():
    r = find_off_runs(s([0.0] * 10 + [100.0] * 20))[0]
    assert r.touches_start and not r.touches_end


def test_a_run_at_the_series_end_is_flagged_censored():
    r = find_off_runs(s([100.0] * 20 + [0.0] * 10))[0]
    assert r.touches_end and not r.touches_start


def test_a_channel_that_never_ran_yields_no_runs():
    """All-zero means the market does not use this channel. Reporting the whole
    series as one enormous holdout would be a false positive on every market
    that simply does not run a channel."""
    assert find_off_runs(s([0.0] * 100)) == []


def test_edge_sharpness_is_high_for_a_clean_stop():
    r = find_off_runs(run_of(100.0, 20, 10, 20))[0]
    assert r.edge_sharpness > 0.9
