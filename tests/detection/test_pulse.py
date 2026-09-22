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


def test_regular_trains_remain_grouped_regardless_of_run_notability():
    for n in (5, 6, 12):
        trains = find_pulse_trains(find_off_runs(pulsed(20, 10, n)))
        assert len(trains) == 1
        assert trains[0].n_pulses == n
