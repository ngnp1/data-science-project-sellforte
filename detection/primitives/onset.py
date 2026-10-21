"""P4 -- runs anchored at the start or end of a series.

A dormant start followed by sustained activity is a launch CANDIDATE only. It
becomes a staggered launch when the cross-market layer confirms the channel was
live elsewhere during that window; on its own it is a censored holdout, and
calling it a launch without peers would be a guess.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from detection import params
from detection.primitives.zero_runs import OffRun, find_off_runs, off_mask


@dataclass(frozen=True)
class Onset:
    first_active: pd.Timestamp
    dormant_days: int
    observed_inactive_days: int
    active_confirmation_days: int


def sustained_activity(s: pd.Series, present: pd.Series | None = None) -> bool:
    """Confirm activity on every observed day of the opening confirmation window."""
    window = s.iloc[:params.LAUNCH_ACTIVE_DAYS]
    known = present.iloc[:params.LAUNCH_ACTIVE_DAYS] if present is not None else None
    # Use the full series' level, not a baseline estimated from a brief spike.
    active = ~off_mask(s, present).iloc[:params.LAUNCH_ACTIVE_DAYS]
    return bool(len(window) == params.LAUNCH_ACTIVE_DAYS and active.all()
                and (known is None or known.all()))


def find_onset(s: pd.Series, present: pd.Series | None = None) -> Onset | None:
    runs = find_off_runs(s, present)          # empty if the channel never ran
    for r in runs:
        # A run cannot touch both edges here: find_off_runs returns []
        # when active_level is not positive, so an all-off series never
        # reaches this loop. The redundant edge check that used to sit in
        # this condition was unreachable, and this file now follows the
        # same rule the composer does -- no gate that no input can trip.
        if r.touches_start and r.n_days >= params.MIN_DAYS:
            idx = s.index.get_loc(r.end)
            following = ~off_mask(s, present).iloc[idx + 1:idx + 1 + params.LAUNCH_ACTIVE_DAYS]
            observed = (int(present.iloc[:idx + 1].sum())
                        if present is not None else r.n_days)
            if (observed >= params.MIN_DAYS
                    and len(following) == params.LAUNCH_ACTIVE_DAYS
                    and following.all()):
                return Onset(s.index[idx + 1], r.n_days, observed,
                             params.LAUNCH_ACTIVE_DAYS)
    return None


def find_discontinuation(s: pd.Series,
                         present: pd.Series | None = None) -> OffRun | None:
    for r in find_off_runs(s, present):
        if r.touches_end and r.n_days >= params.MIN_DAYS:
            return r
    return None
