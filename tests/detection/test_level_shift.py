import numpy as np
import pandas as pd

from detection import params
from detection.primitives.level_shift import find_level_shifts, find_step_episodes


def s(values, start="2024-01-01"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values)),
                     dtype=float)


def noisy(level, n, rng, scale=0.05):
    return list(level * (1.0 + rng.normal(0, scale, n)))


def test_a_clean_step_up_is_found_with_the_right_ratio():
    rng = np.random.default_rng(0)
    series = s(noisy(100.0, 120, rng) + noisy(300.0, 120, rng))
    episodes = find_step_episodes(series)
    assert episodes, "no step found"
    best = max(episodes, key=lambda e: abs(np.log(e.ratio)))
    assert 2.6 < best.ratio < 3.4


def test_a_clean_step_down_is_found():
    rng = np.random.default_rng(1)
    series = s(noisy(300.0, 120, rng) + noisy(100.0, 120, rng))
    episodes = find_step_episodes(series)
    assert episodes
    best = max(episodes, key=lambda e: abs(np.log(e.ratio)))
    assert 0.25 < best.ratio < 0.45


def test_a_flat_noisy_series_produces_no_step():
    rng = np.random.default_rng(2)
    assert find_step_episodes(s(noisy(100.0, 240, rng))) == []


def test_the_benchmark_shaped_ramp_produces_no_step():
    """The benchmark injects a 50-day ramp rising 1.2x to 2.0x in blocks and
    records NO step change; a detector that fires here is wrong.

    This shape never reaches the sharpness gate: a rise spread over 50 days
    inflates the within-window MAD enough that no candidate clears Z_THRESH at
    all. Sharpness is what rejects the 3-30 day ramps -- see
    test_a_gradual_ramp_is_rejected_by_sharpness, which is the test that fails
    when SHARPNESS is disabled.
    """
    rng = np.random.default_rng(3)
    values = noisy(100.0, 120, rng)
    for mult in [1.2, 1.4, 1.6, 1.8, 2.0]:
        values += noisy(100.0 * mult, 10, rng)
    values += noisy(200.0, 120, rng)
    assert find_step_episodes(s(values)) == []


def test_a_gradual_ramp_is_rejected_by_sharpness():
    """THE ramp defence, on a ramp that actually reaches it.

    A 10-day 100->300 ramp clears the z gate (|z| ~ 9) AND the persistence gate
    -- a ramp's new level genuinely does hold, which is why persistence alone
    cannot reject it. Sharpness is the only filter left: the ramp lands ~0.23 of
    its change within SHARPNESS_WINDOW days, against a 0.6 threshold. Set
    params.SHARPNESS to 0 and this test fails; that is what makes the filter
    load-bearing rather than decorative.
    """
    rng = np.random.default_rng(9)
    values = noisy(100.0, 150, rng)
    for i in range(10):
        values += noisy(100.0 + 200.0 * (i + 1) / 10, 1, rng)
    values += noisy(300.0, 150, rng)
    assert find_step_episodes(s(values)) == []


def test_a_one_day_spike_produces_no_step():
    """A single-day excursion never moves a 21-day window median far enough to
    clear Z_THRESH, so it dies at the z gate rather than at persistence.
    Persistence is exercised by the short-excursion test below -- that is the
    one that fails when PERSIST is disabled.
    """
    rng = np.random.default_rng(4)
    values = noisy(100.0, 120, rng) + [900.0] + noisy(100.0, 120, rng)
    assert find_step_episodes(s(values)) == []


def test_a_decaying_excursion_is_rejected_by_persistence():
    """THE persistence defence, on the only shape that actually reaches it.

    A level that jumps and then DECAYS back is not a new level. The opening
    shift clears the z gate, and it clears sharpness too -- the jump itself is
    abrupt -- so persistence is the only filter that can reject it: PERSIST days
    later the level has not held.

    Verified by deleting the persistence block outright, not by zeroing PERSIST.
    Zeroing relocates the check instead of disabling it, which is how this gate
    sat with zero coverage while a mutation battery reported it healthy. The
    earlier rise-hold-revert fixture reaches this filter but is rejected by the
    z gate whether or not persistence runs, so it never tested anything.

    Sized with literals: a 25-day decay from 3x back to 1x.
    """
    rng = np.random.default_rng(5)
    values = (noisy(100.0, 120, rng)
              + [300.0 * (0.94 ** i) for i in range(25)]
              + noisy(100.0, 120, rng))
    assert find_step_episodes(s(values)) == []


def test_a_short_excursion_below_persist_is_rejected():
    rng = np.random.default_rng(5)
    values = (noisy(100.0, 120, rng) + noisy(300.0, params.PERSIST - 4, rng)
              + noisy(100.0, 120, rng))
    assert find_step_episodes(s(values)) == []


def test_a_bounded_step_episode_has_both_ends():
    rng = np.random.default_rng(6)
    values = noisy(100.0, 120, rng) + noisy(300.0, 60, rng) + noisy(100.0, 120, rng)
    episodes = [e for e in find_step_episodes(s(values)) if not e.open_ended]
    assert episodes, "expected a bounded episode"
    e = episodes[0]
    assert 40 <= (e.end - e.start).days <= 80


def test_an_unpaired_shift_runs_to_the_series_end():
    rng = np.random.default_rng(7)
    values = noisy(100.0, 150, rng) + noisy(300.0, 150, rng)
    episodes = find_step_episodes(s(values))
    assert any(e.open_ended for e in episodes)


def test_shifts_report_a_ratio_in_original_units():
    """3.02x must be readable straight from the output -- it is what an analyst
    compares against a briefed budget change."""
    rng = np.random.default_rng(8)
    shifts = find_level_shifts(s(noisy(100.0, 120, rng) + noisy(300.0, 120, rng)))
    assert shifts
    assert any(2.6 < abs(sh.ratio) < 3.4 for sh in shifts)


def test_a_series_that_never_ran_produces_nothing():
    assert find_step_episodes(s([0.0] * 200)) == []


def test_a_short_series_does_not_crash():
    assert find_step_episodes(s([100.0] * 5)) == []


def test_a_step_out_of_zero_reports_no_ratio_rather_than_infinity():
    """The back_to_back shape: a holdout occupying the window before a step, so
    the pre-shift median is exactly 0. Dividing by it once produced inf, which
    serialises as `Infinity` (not valid JSON) and is not a magnitude an analyst
    can read. None is the honest answer; the fact is kept in evidence upstream.
    """
    rng = np.random.default_rng(12)
    values = (noisy(100.0, 60, rng) + [0.0] * 42
              + noisy(300.0, 120, rng))
    for sh in find_level_shifts(s(values)):
        assert sh.ratio is None or np.isfinite(sh.ratio), f"non-finite: {sh.ratio}"
    for ep in find_step_episodes(s(values)):
        assert ep.ratio is None or np.isfinite(ep.ratio)


def test_a_zero_baseline_yields_none_not_inf_directly():
    """Pins the branch itself: 21 zero days then a jump. Sized with literals."""
    rng = np.random.default_rng(13)
    values = [0.0] * 60 + noisy(500.0, 200, rng)
    shifts = find_level_shifts(s(values))
    assert shifts, "expected a shift out of the zero baseline"
    assert any(sh.ratio is None for sh in shifts)
    assert not any(sh.ratio == float("inf") for sh in shifts)


def test_a_far_larger_opposite_shift_does_not_close_an_episode():
    """EPISODE_MATCH_BAND decides where every bounded step_change ENDS, and
    nothing pinned it: widening it to 1e9 changed no test in the suite.

    A modest rise (100 -> 160) followed later by a far larger fall (160 -> 20)
    is not that rise reverting; the two deltas differ by more than the band, so
    neither shift may close the other. Both episodes stay open-ended. Widen the
    band and they pair into one bounded episode, which is the regression this
    catches."""
    rng = np.random.default_rng(21)
    values = (noisy(100.0, 120, rng) + noisy(160.0, 60, rng)
              + noisy(20.0, 120, rng))
    episodes = find_step_episodes(s(values))
    assert episodes, "expected both shifts to be detected"
    assert all(e.open_ended for e in episodes), (
        "a far larger opposite shift must not be treated as the reversal")


def test_a_level_that_barely_holds_is_rejected_by_persist_fraction():
    """PERSIST_FRACTION is the "how much must still hold" half of persistence,
    and nothing behavioural pinned it: setting it to 0 was only ever caught by
    the hardcoded-threshold guard, an artifact of the literal appearing in other
    modules, not coverage. My first attempt at this test was itself decorative
    -- the mutant survived it -- which is the same trap it exists to close.

    The shape: 100 for 80 days, a sharp jump to 400 held 20 days, then a settle
    to 115 -- barely above where it started. The onset clears the z gate and
    clears sharpness (the jump is abrupt), and PERSIST days later the level has
    given back almost all of the move while still sitting a little ABOVE the
    old one, so the sign check cannot reject it either. Only the magnitude half
    of persistence can. Disable it and a shift appears at the onset.
    """
    rng = np.random.default_rng(41)
    values = noisy(100.0, 80, rng) + noisy(400.0, 20, rng) + noisy(115.0, 160, rng)
    onset_shifts = [sh for sh in find_level_shifts(s(values))
                    if 70 <= sh.index <= 90]
    assert onset_shifts == [], (
        "a level that gives back almost all of its move has not held")
