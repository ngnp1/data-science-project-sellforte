import numpy as np
import pandas as pd
import pytest

from pulse.step import find_steps

DAYS = pd.date_range("2024-01-01", periods=365)


def _series(seed=0, sd=0.3):
    return pd.Series(100 * np.random.default_rng(seed).lognormal(0, sd, len(DAYS)), index=DAYS)


@pytest.mark.parametrize("factor", [2.0, 0.6, 3.0])
def test_a_clean_step_is_found_with_its_window(factor):
    s = _series()
    s.iloc[100:160] *= factor
    [(a, b, ratio, conf)] = find_steps(s, s > 15)
    assert abs((a - DAYS[100]).days) <= 3 and abs((b - DAYS[159]).days) <= 3   # noisy series
    assert abs(np.log(ratio / factor)) < 0.2 and 0.9 < conf <= 1


def test_a_drop_to_30_percent_that_holds_is_a_step():   # above the 15 % cutoff: a step, not off
    s = _series()
    s.iloc[100:] *= 0.3
    [(a, b, ratio, _)] = find_steps(s, s > 15)
    assert a == DAYS[100] and b == DAYS[-1] and 0.25 < ratio < 0.35


def test_rates_over_30_seeds():
    """Steps are found, ramps and noise rarely fire. Measured here: 2x 30/30, ramp-then-hold 7/30,
    staircase 4/30, sigma-1.0 noise 1/30 (about 1 in 100 over more seeds; README "Step change")."""
    found = ramp = blocks = noise = 0
    for k in range(30):
        s = _series(k)
        s.iloc[100:160] *= 2
        found += len(find_steps(s, s > 15)) == 1
        r = _series(200 + k)
        r.iloc[100:160] *= np.linspace(1, 2, 60)
        r.iloc[160:] *= 2
        ramp += bool(find_steps(r, r > 15))
        b = _series(300 + k)
        for i, m in enumerate([1.2, 1.4, 1.6, 1.8, 2.0]):
            b.iloc[200 + 10 * i:210 + 10 * i] *= m
        blocks += bool(find_steps(b, b > 15))
        n = _series(400 + k, 1.0)
        noise += bool(find_steps(n, n > 15))
    assert found >= 25 and ramp <= 8 and blocks <= 6 and noise <= 2
