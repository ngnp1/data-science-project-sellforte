import pandas as pd

from pulse.onoff import RHO, p_off
from pulse.pulse import find_pulse_trains, off_runs, pulse_probability

DAYS = pd.date_range("2024-01-01", periods=200)


def test_p_off_is_half_at_the_hard_cutoff():
    s = pd.Series(100.0, index=DAYS)
    s.iloc[10] = RHO * 100.0
    p = p_off(s)
    assert p.iloc[10] == 0.5
    assert p.iloc[0] < 1e-6


def test_pauses_of_mixed_length_form_one_train():
    # A shortest-pause rule caps the group at 10.5 days and drops the 12-day
    # pause; against the median (10) it stays in.
    s = pd.Series(100.0, index=DAYS)
    for start, n in [(20, 7), (50, 10), (80, 12), (110, 10)]:
        s.iloc[start:start + n] = 0.0
    runs = off_runs(p_off(s) >= 0.5)
    trains = find_pulse_trains(runs)
    assert len(trains) == 1 and trains[0].n_pulses == 4
    assert 0 < pulse_probability(p_off(s), trains[0].components, runs) < 1


def test_hsmm_absorbs_a_2_day_spike_and_ignores_a_short_outage():
    from pulse.hsmm import p_off_hsmm
    s = pd.Series(100.0, index=DAYS)
    s.iloc[20:34] = 0.0
    s.iloc[27:29] = 100.0                    # two days of full spend inside the pause
    s.iloc[80:83] = 0.0                      # a 3-day outage: too short to be a pause
    p = p_off_hsmm(s)
    assert (p.iloc[20:34] >= 0.5).all()
    assert (p.iloc[80:83] < 0.5).all()
    assert p.iloc[10] < 0.01 and p.iloc[150] < 0.01
