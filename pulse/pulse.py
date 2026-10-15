"""Pulse trains: a channel switched off repeatedly for pauses of similar length."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MIN_DAYS = 7          # shortest pause that counts
MIN_RUNS = 2          # pauses needed to call it a train
LEN_RATIO = 0.5       # a pause joins a group if <= (1 + LEN_RATIO) x the group's median length
MAX_GAP_MULT = 6      # pauses further apart than this x the shorter pause start a new train

Span = tuple[pd.Timestamp, pd.Timestamp]


@dataclass(frozen=True)
class PulseTrain:
    start: pd.Timestamp
    end: pd.Timestamp
    components: tuple[Span, ...]
    n_pulses: int


def off_runs(mask: pd.Series) -> list[Span]:
    """Maximal runs of off days, excluding runs touching either end of the data."""
    m = mask.to_numpy(bool)
    edges = np.flatnonzero(np.diff(np.r_[0, m.astype(int), 0]))
    return [(mask.index[lo], mask.index[hi - 1]) for lo, hi in zip(edges[::2], edges[1::2])
            if lo > 0 and hi < len(m)]


def _days(span: Span) -> int:
    return (span[1] - span[0]).days + 1


def find_pulse_trains(runs: list[Span]) -> list[PulseTrain]:
    groups: list[list[Span]] = []
    for run in sorted((r for r in runs if _days(r) >= MIN_DAYS), key=lambda r: (_days(r), r[0])):
        # Against the group's median, not its shortest pause, so one short
        # pause cannot cap how long the train's other pauses may be.
        if not groups or _days(run) > np.median([_days(r) for r in groups[-1]]) * (1 + LEN_RATIO):
            groups.append([])
        groups[-1].append(run)
    trains = []
    for group in groups:
        chunks: list[list[Span]] = []
        for run in sorted(group):
            if not chunks or (run[0] - chunks[-1][-1][1]).days - 1 > MAX_GAP_MULT * min(_days(run), _days(chunks[-1][-1])):
                chunks.append([])
            chunks[-1].append(run)
        trains += [PulseTrain(c[0][0], c[-1][1], tuple(c), len(c)) for c in chunks if len(c) >= MIN_RUNS]
    return sorted(trains, key=lambda t: t.start)


def pulse_probability(p_off: pd.Series, components) -> float:
    """Mean P(off) over the train's off-days, times how alike its pauses are.

    ponytail: 1 - CV of pause lengths is a crude regularity term; swap in a
    proper spacing model if irregular-but-real flighting gets under-scored.
    """
    days = pd.concat([p_off.loc[a:b] for a, b in components])
    lengths = np.array([_days(c) for c in components], dtype=float)
    return float(days.mean() * (1 - min(1.0, lengths.std() / lengths.mean())))
