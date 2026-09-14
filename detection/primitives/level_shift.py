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

Measured on the ramp-length sweep in tests/detection/test_level_shift.py, the
two gates divide the work by steepness: ramps of ~3-30 days clear both z and
persistence and are rejected by sharpness alone (measuring 0.17-0.45 against
the SHARPNESS threshold), while the benchmark's own 50-day ramp never reaches sharpness at
all -- spreading the rise over 50 days inflates the within-window MAD until no
candidate clears Z_THRESH (max |z| 2.38). Both shapes correctly yield no step.

Known sensitivity: a 2-day phase-in measures a sharpness a hair under the
SHARPNESS threshold, so it reads as a ramp rather than a step. Real budget changes
that phase in over a couple of days are therefore a false-negative risk; see
the Plan 4 handover.
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
        sigma = _robust_sigma(np.concatenate([before, after]))
        z = delta / sigma
        if abs(z) < params.Z_THRESH:
            continue

        # Persistence: the new level must still hold PERSIST days later.
        tail_end = min(n, t + params.PERSIST + params.W)
        if tail_end - (t + params.PERSIST) < 2:
            continue
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
    shifts = find_level_shifts(s)
    if exclude:
        shifts = [sh for sh in shifts
                  if not any(lo <= sh.at < hi for lo, hi in exclude)]
    if not shifts:
        return []

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
            comparable = (up.ratio is None
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
