"""P1 -- maximal runs of days where a channel was effectively off.

Notability is judged RELATIVE TO THE SERIES' OWN HISTORY, which is what lets
one rule serve two opposite cases: a flighting channel whose normal gaps are
three days needs a much longer run before anything is reported, while a channel
that is otherwise never off needs only MIN_DAYS. A single global threshold
fails one of those two whichever value is chosen.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from detection import params
from detection.io.normalize import active_level


@dataclass(frozen=True)
class OffRun:
    start: pd.Timestamp
    end: pd.Timestamp
    n_days: int
    depth: float
    kind: str            # "exact_zero" | "near_zero" | "missing"
    notable: bool
    edge_sharpness: float
    touches_start: bool
    touches_end: bool


def off_mask(s: pd.Series, present: pd.Series | None = None) -> pd.Series:
    level = active_level(s)
    if not np.isfinite(level) or level <= 0:
        return pd.Series(False, index=s.index)
    threshold = max(params.EPS_ABS, params.RHO * level)
    mask = s <= threshold
    if present is not None:
        mask = mask | ~present.astype(bool)
    return mask


def _spans(mask: pd.Series) -> list[tuple[int, int]]:
    """Maximal [start, end] index pairs where mask is True."""
    out, start = [], None
    values = list(mask.values)
    for i, flag in enumerate(values):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(values) - 1))
    return out


def _edge_sharpness(s: pd.Series, lo: int, hi: int, level: float) -> float:
    """How cleanly spend stops and restarts, as a fraction of the active level.

    A clean stop scores near 1; a gradual wind-down scores lower.
    """
    before = s.iloc[max(0, lo - params.ROLLING):lo]
    after = s.iloc[hi + 1:hi + 1 + params.ROLLING]
    flank = pd.concat([before, after])
    if flank.empty or level <= 0:
        return 0.0
    return float(min(1.0, flank.median() / level))


def find_off_runs(s: pd.Series, present: pd.Series | None = None) -> list[OffRun]:
    level = active_level(s)
    if not np.isfinite(level) or level <= 0:
        # The market never ran this channel. Reporting the whole series as one
        # enormous holdout would be a false positive on every market that
        # simply does not use a channel.
        return []

    mask = off_mask(s, present)
    spans = _spans(mask)
    if not spans:
        return []

    lengths = np.array([hi - lo + 1 for lo, hi in spans], dtype=float)

    runs: list[OffRun] = []
    for idx, (lo, hi) in enumerate(spans):
        n_days = int(lengths[idx])
        window = s.iloc[lo:hi + 1]

        others = np.delete(lengths, idx)
        floor = params.MIN_DAYS
        if others.size >= params.MIN_RUNS_FOR_RATIO:
            # The ratio guard's reference distribution is the OTHER off-runs
            # that already clear MIN_DAYS on their own -- i.e. routine long
            # pauses, such as a weekly flighting channel's regular week off.
            # Short blips well under MIN_DAYS are already excluded by
            # MIN_DAYS alone and must not inflate the bar for a distinct,
            # much-longer run: with a homogeneous 3-day cadence, RUN_RATIO *
            # p90(3) = 9 would make an honest 8-day dark period unreportable,
            # even though nothing in that channel's history is anywhere near
            # 8 days. Comparing only against gaps that were themselves
            # already "long" keeps the guard targeted at genuine periodic
            # long-pause channels instead of penalizing short-blip noise.
            qualifying = others[others >= params.MIN_DAYS]
            if qualifying.size:
                floor = max(params.MIN_DAYS,
                            params.RUN_RATIO * float(np.percentile(qualifying, 90)))
        notable = n_days >= floor

        if present is not None and not present.iloc[lo:hi + 1].any():
            kind = "missing"
        elif float(window.max()) <= params.EPS_ABS:
            kind = "exact_zero"
        else:
            kind = "near_zero"

        runs.append(OffRun(
            start=s.index[lo], end=s.index[hi], n_days=n_days,
            depth=float(1.0 - window.mean() / level),
            kind=kind, notable=bool(notable),
            edge_sharpness=_edge_sharpness(s, lo, hi, level),
            touches_start=lo == 0, touches_end=hi == len(s) - 1,
        ))
    return runs


def notable_runs(s: pd.Series, present: pd.Series | None = None) -> list[OffRun]:
    return [r for r in find_off_runs(s, present) if r.notable]
