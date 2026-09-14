import pandas as pd

from detection.primitives.pulse import find_pulse_trains
from detection.primitives.zero_runs import find_off_runs, notable_runs


def s(values, start="2024-01-01"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values)),
                     dtype=float)


def pulsed(on, off, n):
    values = []
    for _ in range(n):
        values += [100.0] * on + [0.0] * off
    return s(values + [100.0] * on)


def test_a_repeating_on_off_pattern_becomes_one_train():
    """Spec section 7's P3: the detector emits ONE grouped event spanning first
    start to last end, with the individual windows as components. The harness
    matches against grouped truth, so emitting four separate events scores
    zero."""
    trains = find_pulse_trains(find_off_runs(pulsed(21, 14, 4)))
    assert len(trains) == 1
    t = trains[0]
    assert t.n_pulses == 4
    assert len(t.components) == 4
    assert t.start == t.components[0][0]
    assert t.end == t.components[-1][1]


def test_a_single_holdout_is_not_a_pulse_train():
    runs = find_off_runs(s([100.0] * 40 + [0.0] * 14 + [100.0] * 40))
    assert find_pulse_trains(runs) == []


def test_runs_of_wildly_different_length_are_not_a_train():
    """A 7-day gap and a 90-day shutdown are not the same phenomenon."""
    values = ([100.0] * 30 + [0.0] * 7 + [100.0] * 30 + [0.0] * 90
              + [100.0] * 30)
    assert find_pulse_trains(find_off_runs(s(values))) == []


def test_only_notable_runs_are_grouped():
    """An intermittent channel's routine 3-day gaps must not become a train."""
    values = ([100.0] * 11 + [0.0] * 3) * 8 + [100.0] * 20
    assert find_pulse_trains(find_off_runs(s(values))) == []


def test_components_are_inclusive_and_ordered():
    t = find_pulse_trains(find_off_runs(pulsed(21, 14, 3)))[0]
    starts = [a for a, _ in t.components]
    assert starts == sorted(starts)
    for a, b in t.components:
        assert (b - a).days == 13    # 14 inclusive days


def test_a_run_touching_the_series_edge_is_excluded_from_the_train():
    """A dormant open or a discontinuation at the series edge is P4's job, not
    a pulse component: an edge-touching run is censored (we never see it
    turn back off, or never saw it turn on), so it cannot be confirmed as one
    of a repeating pattern the way an interior run can."""
    values = ([0.0] * 14 + [100.0] * 21 + [0.0] * 14 + [100.0] * 21
              + [0.0] * 14 + [100.0] * 21)
    runs = find_off_runs(s(values))
    trains = find_pulse_trains(runs)
    assert len(trains) == 1
    t = trains[0]
    assert t.n_pulses == 2
    assert all(a != s(values).index[0] for a, _ in t.components)


def test_a_regular_train_of_six_or_more_windows_is_currently_suppressed():
    """KNOWN LIMITATION, pinned so it cannot change unnoticed.

    P1's intermittent guard and P3's pulse detector are in direct conflict, with
    a hard cliff at exactly six windows. A run becomes notable only if it also
    clears RUN_RATIO x p90 of the series' OTHER off-runs, and that rule switches
    on once five other runs exist. In a regular train every other run is the
    same length as this one, so p90 equals this run's own length and the floor
    becomes RUN_RATIO times it. Nothing in a regular train ever clears that:
    five windows work, six produce zero notable runs and no train at all.

    The more regular and the more numerous the flighting pattern, the more
    certainly it is discarded -- while spec section 7 calls pulse trains the
    only place adstock decay is observable and gives them the highest
    informativeness weight.

    The implementation is faithful to spec section 7 P1 as written; the conflict
    is in the spec. It cannot fire on this benchmark, whose pulse family draws
    2-4 windows, which is exactly why it is written down here: on real flighting
    data it is the normal case, and no benchmark score can reveal it.
    """
    n_on, n_off = 20, 10
    for n_pulses in (5, 6):
        values = []
        for _ in range(n_pulses):
            values += [100.0] * n_on + [0.0] * n_off
        values += [100.0] * n_on
        series = s(values)
        runs = notable_runs(series)
        if n_pulses == 5:
            assert len(runs) == 5, "a five-window train must still register"
            assert find_pulse_trains(find_off_runs(series)), "expected a train"
        else:
            assert runs == [], (
                "six regular windows: the ratio rule suppresses every run. "
                "If this now passes, the spec conflict has been resolved and "
                "this test should be replaced with the real expectation.")
