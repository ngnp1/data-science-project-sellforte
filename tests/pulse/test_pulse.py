import pandas as pd

from pulse.onoff import RHO, off_mask, p_off
from pulse.pulse import find_pulse_trains, off_runs, pulse_probability

DAYS = pd.date_range("2024-01-01", periods=200)


def test_p_off_is_half_at_the_hard_cutoff_and_one_on_missing_days():
    s = pd.Series(100.0, index=DAYS)
    s.iloc[10] = RHO * 100.0
    present = pd.Series(True, index=DAYS)
    present.iloc[20] = False
    p = p_off(s, present)
    assert p.iloc[10] == 0.5
    assert p.iloc[20] == 1.0
    assert p.iloc[0] < 1e-6


def test_pauses_of_mixed_length_form_one_train():
    # A shortest-pause rule caps the group at 10.5 days and drops the 12-day
    # pause; against the median (10) it stays in.
    s = pd.Series(100.0, index=DAYS)
    for start, n in [(20, 7), (50, 10), (80, 12), (110, 10)]:
        s.iloc[start:start + n] = 0.0
    runs = off_runs(off_mask(s))
    trains = find_pulse_trains(runs)
    assert len(trains) == 1 and trains[0].n_pulses == 4
    assert 0 < pulse_probability(p_off(s), trains[0].components, runs) < 1
