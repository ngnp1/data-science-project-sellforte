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
from detection.primitives.zero_runs import OffRun, find_off_runs


@dataclass(frozen=True)
class Onset:
    first_active: pd.Timestamp
    dormant_days: int


def find_onset(s: pd.Series, present: pd.Series | None = None) -> Onset | None:
    runs = find_off_runs(s, present)          # empty if the channel never ran
    for r in runs:
        if r.touches_start and not r.touches_end and r.n_days >= params.MIN_DAYS:
            idx = s.index.get_loc(r.end)
            return Onset(first_active=s.index[idx + 1], dormant_days=r.n_days)
    return None


def find_discontinuation(s: pd.Series,
                         present: pd.Series | None = None) -> OffRun | None:
    for r in find_off_runs(s, present):
        if r.touches_end and not r.touches_start and r.n_days >= params.MIN_DAYS:
            return r
    return None
