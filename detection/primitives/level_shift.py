"""P2 -- robust level shifts on a scale-free series.

Everything is medians and a MAD-scaled z on log1p(spend / level). A Gaussian z
on raw euros breaks on both the 15x market-size spread and the heavy right tail
of daily spend; a robust z on a log ratio is scale-free, tolerant of
multiplicative noise, and prints legibly.

Two filters do different jobs, and BOTH are needed:

- persistence rejects short excursions -- a run that reverts inside PERSIST
  days is not a new level. (A one-day spike never reaches this gate; it cannot
  move a 21-day window median far enough to clear Z_THRESH in the first place.)
- sharpness rejects gradual ramps -- a genuine step concentrates its change
  into a couple of days, while a ramp spreads it out. Persistence alone does
  NOT reject a ramp, because a ramp's new level genuinely does hold.

The z's scale is measured on each comparison window separately and never
across the pair -- see _noise_sigma, which is where the closing shift of a
bounded step change used to be lost.

WHICH gate rejects a gradual ramp was measured rather than assumed, and the
answer differs between the benchmark's own ramp and a synthetic one. The
distinction matters: they are not the same shape and they do not die at the
same gate.

- The BENCHMARK's gradual ramp -- a five-block drift, on a real series carrying
  seasonality and day-of-week structure -- CLEARS the z gate (five candidates
  clear it; max |z| on the series is 4.761) and CLEARS persistence (all five).
  SHARPNESS alone rejects it: the best surviving candidate lands 0.355 of its
  change inside SHARPNESS_WINDOW days, roughly 41% below the threshold. No
  shift and no episode is emitted. Sharpness is the load-bearing gate for this
  shape; lowering it removes the only defence that is holding. Pinned by
  test_the_benchmarks_own_blocked_ramp_emits_no_step.
- A SYNTHETIC flat rise of the same total size, spread evenly over L days with
  no seasonality and no blocking, behaves differently. Over an eight-seed
  sweep: L of four days or fewer reads as a step, L of seven or more never
  does, and the crossover sits near five days. Those ramps are rejected by
  sharpness as well (at most 0.43 of the change inside the window); only the
  longest and smoothest of them is stopped earlier, by the z gate.

Known sensitivity: SHARPNESS, not the z gate, is what holds a blocked gradual
ramp, and on the benchmark's own it holds with room to spare rather than
marginally. The residual risk is the opposite shape -- a real budget change
phased in slowly on a quiet series reads as a ramp and is dropped, and a
phase-in of four days or fewer now reads as a step where it once did not. Both
are sharpness decisions and neither involves the z gate; see the Plan 4
handover.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from detection import params
from detection.io.normalize import active_level, scale_free


@dataclass(frozen=True)
class LevelShift:
    at: pd.Timestamp
    index: int
    z: float
    delta: float
    ratio: float | None      # None when the level before the shift was zero
    sharpness: float


@dataclass(frozen=True)
class StepEpisode:
    start: pd.Timestamp
    end: pd.Timestamp
    ratio: float | None      # None when the level before the shift was zero
    z: float
    open_ended: bool


def _robust_sigma(values: np.ndarray) -> float:
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    return max(float(params.MAD_TO_SIGMA * mad), params.SIGMA_FLOOR)


def _noise_sigma(before: np.ndarray, after: np.ndarray) -> float:
    """Noise scale on either side of a candidate, never measured across it.

    The scale in the z must describe the spread the two levels are NOT
    supposed to have -- the day-to-day noise -- and nothing else. Measuring it
    on the concatenation of the two windows folds the candidate's own step into
    the scale it is being divided by, and it does so hardest at the one index
    where the step is real: there the pooled sample is an even mixture of the
    two levels, its MAD is about half the delta, and |z| collapses to roughly
    2 / MAD_TO_SIGMA no matter how large or clean the step is. Move a few days
    off the true edge and the mixture goes lopsided, the pooled median falls
    inside the majority level, the MAD drops back to the within-level noise and
    |z| leaps. The z gate therefore had a notch exactly where a step change
    sits, while the sharpness gate has its peak there -- the two gates were
    maximised at different indices and a genuine step passed both only when
    noise happened to leave one index in the overlap.

    Taking the LARGER of the two one-sided scales removes the notch. `max` is
    the conservative combiner of the two: it cannot come out below the noise on
    either side, and it keeps a quiet window from lending its quiet to a noisy
    neighbour. What it deliberately no longer absorbs is the DIFFERENCE between
    the two window medians. Where the two windows sit at the same level that
    difference is only sampling noise and the two forms are close; where they
    sit at different levels it is the step itself, which is what the z is meant
    to be measuring rather than dividing by.

    Not absorbing it does raise |z| generally, so the precision cost is
    measured rather than argued: a battery of series with no level change --
    Gaussian noise at two scales and strong day-of-week seasonality, forty
    seeds each -- yields no shift at all (tests/detection/test_level_shift.py),
    the ramp and spike defences are unmoved, and the development split's
    null-scenario false-positive rate stayed at zero.
    """
    return max(_robust_sigma(before), _robust_sigma(after))


def find_level_shifts(s: pd.Series) -> list[LevelShift]:
    level = active_level(s)
    if not np.isfinite(level) or level <= 0 or len(s) < 2 * params.W + 1:
        return []

    y = scale_free(s).rolling(params.ROLLING, center=True,
                              min_periods=1).median().values
    raw = s.values
    n = len(y)

    candidates: list[LevelShift] = []
    for t in range(params.W, n - params.W):
        before, after = y[t - params.W:t], y[t:t + params.W]
        delta = float(np.median(after) - np.median(before))
        sigma = _noise_sigma(before, after)
        z = delta / sigma
        if abs(z) < params.Z_THRESH:
            continue

        # Persistence: the new level must still hold PERSIST days later.
        # t <= n - W - 1, so n - t >= W + 1 and this tail always holds at
        # least W + 1 - PERSIST days. No length guard is reachable here; one
        # used to sit at this line and no input could trip it.
        tail_end = min(n, t + params.PERSIST + params.W)
        held = float(np.median(y[t + params.PERSIST:tail_end])
                     - np.median(before))
        if (abs(held) < abs(delta) * params.PERSIST_FRACTION
                or np.sign(held) != np.sign(delta)):
            continue

        # Sharpness: how much of the change lands within a few days.
        k = params.SHARPNESS_WINDOW
        near = float(np.median(y[t:t + k]) - np.median(y[max(0, t - k):t]))
        sharpness = abs(near) / abs(delta) if delta else 0.0
        if sharpness < params.SHARPNESS:
            continue

        before_raw = np.median(raw[t - params.W:t])
        after_raw = np.median(raw[t:t + params.W])
        # A step OUT OF zero has no finite ratio. Reporting inf was worse than
        # reporting nothing: it serialises as `Infinity`, which is not valid
        # JSON, and spec section 8 wants a magnitude an analyst can read against
        # a briefed budget change. None says "rose from nothing" honestly.
        # Reachable on the benchmark's back_to_back shape, where a holdout
        # occupies most of the window before the shift.
        ratio = float(after_raw / before_raw) if before_raw > 0 else None

        candidates.append(LevelShift(at=s.index[t], index=t, z=float(z),
                                     delta=delta, ratio=ratio,
                                     sharpness=float(sharpness)))

    # Keep the strongest candidate in each W-wide neighbourhood, so one step
    # does not report as a cluster of adjacent change points.
    kept: list[LevelShift] = []
    for c in sorted(candidates, key=lambda c: -abs(c.z)):
        if all(abs(c.index - k.index) >= params.W for k in kept):
            kept.append(c)
    return sorted(kept, key=lambda c: c.index)


def find_step_episodes(
    s: pd.Series,
    exclude: list[tuple[pd.Timestamp, pd.Timestamp]] | None = None,
) -> list[StepEpisode]:
    """Pair opposing level shifts into episodes.

    `exclude` names windows where the channel was off, and the shifts inside
    them must be removed BEFORE pairing rather than after.

    Pairing is greedy in index order, so the drop INTO an off-window claims the
    rise OUT of it as its reversal. When a genuine step follows the restart --
    the benchmark's back_to_back shape -- that rise is also the step's opening
    shift, so the step is left holding only its closing shift, becomes
    open-ended, and is dropped. Filtering completed episodes cannot recover it:
    by then the pairing has already gone wrong.

    Only the drop in is excluded, NOT the rise out. The drop is an artefact of
    the channel stopping. The rise carries real information -- the level after
    the restart is compared against the level before the pause, so a channel
    that comes back at three times its old budget shows that here, and it is the
    only shift that can open the step that follows.
    """
    # A shift's date is resolved only to within the centred rolling median's
    # half-width: y[t] is a median over ROLLING days centred on t, so a shift
    # sitting on the edge of an off-window can be dated a few days either side
    # of the first or last off day.
    smear = pd.Timedelta(days=params.ROLLING // 2)
    shifts = find_level_shifts(s)
    if exclude:
        # Widen only the LEADING edge. The drop INTO an off-window can be dated
        # before the first off day and so fall outside the window that is meant
        # to exclude it; when it does it survives to claim the rise out as its
        # reversal and the step that follows is destroyed -- the exact failure
        # the exclusion exists to prevent. The rise out sits at the trailing
        # edge and must survive untouched.
        shifts = [sh for sh in shifts
                  if not any(lo - smear <= sh.at < hi for lo, hi in exclude)]
    if not shifts:
        return []

    # Trailing edges of the excluded windows that a channel can come BACK from:
    # a window that abuts the start of the series is a channel that had not
    # launched yet, not one that paused. See `restarted_out_of_a_pause` below.
    pause_ends = [hi for lo, hi in (exclude or []) if lo > s.index[0]]

    episodes: list[StepEpisode] = []
    used: set[int] = set()
    for i, up in enumerate(shifts):
        if i in used:
            continue
        partner = None
        for j in range(i + 1, len(shifts)):
            if j in used:
                continue
            other = shifts[j]
            # Opposite sign and comparable magnitude closes an episode.
            band = params.EPISODE_MATCH_BAND
            # A shift that rises OUT OF zero has no comparable magnitude: in log
            # space it is unboundedly large, so the band can never match it with
            # the ordinary closing shift, and the episode is left open-ended and
            # dropped. That is the back_to_back shape -- a holdout, then a step.
            # Its magnitude is already reported as None for the same reason
            # (there is no finite ratio out of zero), so pair it on direction
            # alone and let the closing shift supply the extent.
            #
            # That waiver is for that shape ALONE. Granted to every rise out of
            # zero it also let an ordinary channel LAUNCH pair with the next
            # opposite shift whatever its size -- a cut ten months later, four
            # times outside the band -- and report the whole span as one step
            # change. A launch is not a step out of a previous level, because
            # there is no previous level for a later shift to be a reversal of;
            # a channel that PAUSED does have one, which is why the waiver is
            # scoped to a restart out of an interior off-window this call was
            # asked to exclude.
            restarted_out_of_a_pause = (
                up.ratio is None
                and any(hi <= up.at <= hi + smear for hi in pause_ends))
            comparable = (restarted_out_of_a_pause
                          or 1 / band <= abs(other.delta / up.delta) <= band)
            if np.sign(other.delta) != np.sign(up.delta) and comparable:
                partner = j
                break
        if partner is not None:
            other = shifts[partner]
            used.update({i, partner})
            episodes.append(StepEpisode(
                start=up.at, end=s.index[other.index - 1], ratio=up.ratio,
                z=up.z, open_ended=False))
        else:
            used.add(i)
            episodes.append(StepEpisode(
                start=up.at, end=s.index[-1], ratio=up.ratio, z=up.z,
                open_ended=True))
    return episodes
