import pytest
import pathlib

import numpy as np
import pandas as pd

from detection import params
from detection.io.panel import build_panel
from detection.primitives.level_shift import find_level_shifts, find_step_episodes
from detection.primitives.zero_runs import find_off_runs

DEV = pathlib.Path(__file__).resolve().parents[2] / "benchmark" / "datasets" / "dev"


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
    """A SYNTHETIC stand-in for the benchmark's ramp: a 50-day rise in blocks,
    flat noise, no seasonality. A detector that fires here is wrong.

    This is not the benchmark's actual ramp and does not die at the same gate --
    see test_the_benchmarks_own_blocked_ramp_emits_no_step, which loads the real
    series and where SHARPNESS is what holds. On this cleaner synthetic shape
    the z gate gets there first: spreading the rise over 50 days inflates the
    within-window MAD enough that no candidate clears Z_THRESH at all (max |z|
    3.25, measured). Both tests are kept: the real one pins the defence that
    matters, this one pins the smooth-and-quiet end of the family.
    """
    rng = np.random.default_rng(3)
    values = noisy(100.0, 120, rng)
    for mult in [1.2, 1.4, 1.6, 1.8, 2.0]:
        values += noisy(100.0 * mult, 10, rng)
    values += noisy(200.0, 120, rng)
    assert find_step_episodes(s(values)) == []


def test_a_gradual_ramp_is_rejected_by_sharpness():
    """THE ramp defence, on a ramp that actually reaches it.

    A 10-day 100->300 ramp clears the z gate (max |z| ~ 9.5) AND the persistence
    gate -- a ramp's new level genuinely does hold, which is why persistence
    alone cannot reject it. Sharpness is the only filter left: no candidate that
    clears z and persistence lands more than 0.43 of its change within
    SHARPNESS_WINDOW days, against a 0.6 threshold. Delete the sharpness block
    and this test fails; that is what makes the filter load-bearing rather than
    decorative.
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

    A level that jumps and then DECAYS back is not a new level, and must yield
    no step. It no longer reaches persistence to be rejected: measuring the z's
    scale on each comparison window separately (Task 8) means the window AFTER
    the jump is scored on its own spread, and a window spanning a decay from 3x
    to 1x has an enormous spread, so |z| peaks at 0.93 and the z gate rejects
    the onset first. Deleting the persistence block therefore no longer makes
    this test fail -- the shape is still rejected, by an earlier gate.

    The persistence gate's own behavioural pin is
    test_a_level_that_barely_holds_is_rejected_by_persist_fraction, which does
    die when the block is deleted. Both were verified by deleting the block
    outright, not by zeroing PERSIST: zeroing relocates the check instead of
    disabling it, which is how this gate once sat with zero coverage while a
    mutation battery reported it healthy.

    Sized with literals: a 25-day decay from 3x back to 1x.
    """
    rng = np.random.default_rng(5)
    values = (noisy(100.0, 120, rng)
              + [300.0 * (0.94 ** i) for i in range(25)]
              + noisy(100.0, 120, rng))
    assert find_step_episodes(s(values)) == []


def test_a_ten_day_excursion_is_rejected_before_it_reaches_persistence():
    """A brief excursion must yield no step -- but NOT via the gate its old
    name claimed.

    This test used to be called ...below_persist and to size its excursion
    `params.PERSIST - 4`, a fixture derived from the parameter it purported to
    exercise. It never reached that parameter. Measured on this exact series:
    the largest |z| anywhere is 0.692 against a Z_THRESH of 3.5, so no
    candidate survives to the persistence block at all, and deleting
    persistence AND sharpness together leaves this test green. A ten-day
    excursion simply cannot move a W-day window median far enough -- both
    comparison windows span the excursion and the centred rolling median
    flattens what is left.

    Sized with a literal now, and kept for what it does pin: the z gate's
    insensitivity to an excursion much shorter than its own window. The
    persistence gate's behavioural pin is
    test_a_level_that_barely_holds_is_rejected_by_persist_fraction, which is
    the one that dies when the block is deleted. Same annotation, and same
    reason, as test_a_decaying_excursion_is_rejected_by_persistence above.
    """
    rng = np.random.default_rng(5)
    values = (noisy(100.0, 120, rng) + noisy(300.0, 10, rng)
              + noisy(100.0, 120, rng))
    series = s(values)
    assert find_step_episodes(series) == []
    assert find_level_shifts(series) == [], (
        "no candidate should even clear the z gate on this shape")


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


def test_a_step_following_a_holdout_is_not_swallowed_by_the_pairing():
    """The back_to_back shape, which exists on both splits: a holdout, then a
    step change starting the day the channel comes back.

    Three shifts are detected -- the drop into the holdout, the rise out of it,
    and the step's close. Greedy pairing used to let the drop claim the rise as
    its reversal, leaving the step holding only its closing shift; it went
    open-ended and was dropped, so the step vanished even though every shift
    that defines it was found. Passing the off-window excludes the drop, and the
    rise (which has no finite ratio, coming out of zero) pairs with the close on
    direction alone.
    """
    rng = np.random.default_rng(3)
    values = (noisy(1000.0, 200, rng) + [0.0] * 42
              + noisy(3000.0, 56, rng) + noisy(1000.0, 150, rng))
    series = s(values)
    off = [(r.start, r.end) for r in find_off_runs(series) if r.notable]
    assert off, "fixture must produce a notable off-run"

    episodes = find_step_episodes(series, exclude=off)
    bounded = [e for e in episodes if not e.open_ended]
    assert bounded, "the step following the holdout was lost to pairing"
    e = bounded[0]
    # True window is 2024-08-30..2024-10-24; boundaries land within a day.
    assert abs((e.start - pd.Timestamp("2024-08-30")).days) <= 1
    assert abs((e.end - pd.Timestamp("2024-10-24")).days) <= 1


def test_the_drop_into_an_off_window_is_excluded_but_the_rise_out_is_kept():
    """Pins WHICH edge is excluded. Dropping both loses the step entirely (the
    rise out of an off-window is also the opening shift of whatever follows);
    dropping neither lets the drop claim the rise.

    The second assertion used to compare the episode start against the
    off-run's START. A shift on that edge is never dated at the first off day
    -- the centred rolling median puts it a day or so either side of it -- so
    the comparison was true whatever `exclude` did, and neutering `exclude`
    left it passing. It now names the drop shift itself and asserts that no
    episode opens ON it, which is the mechanism: delete the exclusion filter in
    find_step_episodes and this fails.
    """
    rng = np.random.default_rng(3)
    values = (noisy(1000.0, 200, rng) + [0.0] * 42
              + noisy(3000.0, 56, rng) + noisy(1000.0, 150, rng))
    series = s(values)
    off = [(r.start, r.end) for r in find_off_runs(series) if r.notable]
    kept = find_level_shifts(series)
    lo, hi = off[0]
    assert any(sh.at >= hi for sh in kept), "the rise out must survive"

    drops_in = [sh for sh in kept if sh.delta < 0 and sh.at <= hi]
    assert len(drops_in) == 1, f"fixture must show one drop in: {drops_in}"
    drop = drops_in[0]

    episodes = find_step_episodes(series, exclude=off)
    assert not any(e.start == drop.at for e in episodes), (
        f"no episode may open on the drop into the off-window ({drop.at}); "
        f"episodes opened at {[e.start for e in episodes]}")


def test_a_launch_out_of_zero_does_not_pair_with_an_unrelated_later_cut():
    """The zero-origin pairing waiver, scoped.

    A shift out of zero has no measurable magnitude, so
    test_a_step_following_a_holdout_is_not_swallowed_by_the_pairing waives the
    EPISODE_MATCH_BAND check for it. Waived for EVERY rise out of zero, that
    also covered an ordinary channel launch: the channel starts, and the next
    opposite shift -- here a cut to half, ten months later and about four times
    outside the band -- was accepted as its reversal, reporting the whole span
    as one step change. No development scenario has this shape, so the score
    could not reveal it.

    The launch has no previous level for a later shift to revert TO, which is
    what separates it from a pause: its off-window abuts the start of the
    series. Both shifts must therefore stay open-ended, which is what the
    composition layer drops. Reproduced through the exact
    find_step_episodes(series, exclude=off) call the composition layer makes.
    """
    rng = np.random.default_rng(0)
    values = ([0.0] * 45 + noisy(1000.0, 300, rng) + noisy(500.0, 300, rng))
    series = s(values)
    off = [(r.start, r.end) for r in find_off_runs(series) if r.notable]
    assert off and off[0][0] == series.index[0], (
        "fixture must open with a not-yet-launched off-window")

    shifts = find_level_shifts(series)
    launch = [sh for sh in shifts if sh.ratio is None]
    assert launch, "fixture must produce a rise out of zero"
    cut = [sh for sh in shifts if sh.delta < 0]
    assert cut, "fixture must produce a later cut"
    ratio = abs(cut[0].delta / launch[0].delta)
    assert not 1 / params.EPISODE_MATCH_BAND <= ratio <= params.EPISODE_MATCH_BAND, (
        f"fixture is pointless unless the cut is outside the band: {ratio}")

    episodes = find_step_episodes(series, exclude=off)
    assert episodes, "expected both shifts to be reported"
    assert all(e.open_ended for e in episodes), (
        "a launch out of zero must not claim an unrelated later cut as its "
        f"reversal: {[(e.start, e.end, e.open_ended) for e in episodes]}")


def test_a_noiseless_series_does_not_fire_on_a_trivial_shift():
    """SIGMA_FLOOR, whose spec risk note ("too low makes flat series fire
    spuriously") was until now unverified: every other fixture here carries 5%
    noise, so the floor never binds and mutating it changed no test.

    On a series with no noise at all the MAD is zero, so the robust sigma
    collapses and any shift divides by almost nothing. A 0.6% change is not a
    budget decision; the floor is what stops it reading as one. Drop the floor
    and both shifts below fire.
    """
    assert find_step_episodes(s([100.0] * 120 + [100.6] * 120)) == []
    assert find_step_episodes(s([100.0] * 120 + [103.0] * 120)) == []


def weekly(level, n, rng, scale=0.12, amplitude=0.15, phase=0):
    """A level carrying day-of-week structure as well as noise.

    The flat `noisy` fixtures above are cleaner than any real media series: at
    5% noise and no weekly shape a step is found at its own change point by
    luck alone, which is why they could not show the defect the two tests below
    exist for. These numbers are the shape of the series the detector actually
    runs on.
    """
    return [level * (1 + amplitude * np.sin(2 * np.pi * (phase + i) / 7))
            * (1 + rng.normal(0, scale)) for i in range(n)]


def bounded_step(seed, pre=150, length=56, mult=3.0, post=150):
    rng = np.random.default_rng(seed)
    return (weekly(100.0, pre, rng)
            + weekly(100.0 * mult, length, rng, phase=pre)
            + weekly(100.0, post, rng, phase=pre + length))


def test_both_edges_of_a_bounded_step_are_detected():
    """Step recall was 2/13 on the development split because one edge or both
    went missing on eleven of the thirteen step changes -- most often the
    CLOSING one, which left the episode open-ended and dropped.

    The cause was the z's scale: measured over the two comparison windows
    CONCATENATED, it folded the candidate's own step into the denominator, and
    hardest at the one index where the step is real -- there the pooled sample
    is an even mixture of the two levels and |z| collapses to about
    2 / MAD_TO_SIGMA whatever the step. |z| had a notch exactly where sharpness
    has its peak, so a step cleared both gates only when noise happened to
    leave one index in the overlap.

    A 56-day step to 3x, on a series with day-of-week structure, over eight
    seeds so no single draw can carry it. Restore the concatenated scale and
    six of the eight lose an edge.
    """
    for seed in range(8):
        ats = [sh.index for sh in find_level_shifts(s(bounded_step(seed)))]
        assert any(abs(i - 150) <= 2 for i in ats), \
            f"seed {seed}: opening shift missing: {ats}"
        assert any(abs(i - 206) <= 2 for i in ats), \
            f"seed {seed}: closing shift missing: {ats}"


def test_the_z_at_the_change_point_itself_is_not_collapsed_by_its_own_step():
    """The mechanism behind the test above, pinned on its own index.

    The shift must be found AT the change point, not two weeks off it where the
    comparison windows no longer straddle the transition -- and with a |z| that
    reflects the step rather than the mixture. Restore the concatenated scale
    and no candidate survives within two days of the change point at all.
    """
    found = [sh for sh in find_level_shifts(s(bounded_step(0)))
             if abs(sh.index - 150) <= 2]
    assert found, "the step must be found at the change point itself"
    assert abs(found[0].z) >= params.Z_THRESH


def test_no_shift_is_found_on_a_battery_of_series_with_no_level_change():
    """The precision half of measuring the z's scale on each window separately.

    Not folding the between-window difference into the denominator raises |z|
    generally, so the cost has to be measured rather than argued. These series
    have no level change to find: Gaussian noise at 5% and at 20%, and a series
    whose swing is day-of-week seasonality rather than a budget decision. Forty
    seeds each; any firing here is a false positive of exactly the kind the
    null scenarios on the development split count.
    """
    for seed in range(40):
        rng = np.random.default_rng(seed)
        fixtures = {
            "gaussian 5%": noisy(100.0, 400, rng),
            "gaussian 20%": noisy(100.0, 400, rng, scale=0.2),
            "day-of-week": [100.0 * (1 + 0.3 * np.sin(2 * np.pi * i / 7))
                            * (1 + rng.normal(0, 0.1)) for i in range(400)],
        }
        for name, values in fixtures.items():
            found = find_level_shifts(s(values))
            assert found == [], (
                f"seed {seed}, {name}: no level change to find, but reported "
                f"{[(sh.index, round(sh.z, 2)) for sh in found]}")


@pytest.mark.benchmark_data
def test_the_benchmarks_own_blocked_ramp_emits_no_step():
    """THE ramp defence, on the benchmark's own gradual ramp rather than a
    synthetic stand-in.

    dev_044's FI/Radio series is a five-block drift that the benchmark records
    as NO step change. It is not the shape the synthetic ramp fixtures above
    model -- it carries seasonality and day-of-week structure, and it is blocked
    rather than smooth -- and it does not die at the same gate. Measured through
    the current gates:

        candidates clearing the z gate : 5 (not zero)
        max |z| on the series          : 4.761 at index 201
        persistence                    : all five pass
        best surviving sharpness       : 0.355, about 41% below SHARPNESS
        shifts emitted                 : none
        step episodes emitted          : none

    So SHARPNESS alone rejects it. The module docstring used to name the z gate
    as what held this shape, with a thin margin; that was measured on the
    synthetic fixture and was never true of this series. Lower SHARPNESS far
    enough to admit a 0.355 candidate and this test fails -- which is what makes
    it a pin on the real defence rather than an incidental pass.
    """
    media = pd.read_csv(DEV / "dev_044" / "media.csv", parse_dates=["date"])
    series = build_panel(media, None, "dev_044").series("FI", "Radio")

    assert find_level_shifts(series) == [], (
        "the benchmark's own gradual ramp must not report a level shift")
    assert find_step_episodes(series) == [], (
        "the benchmark's own gradual ramp must not report a step episode")
