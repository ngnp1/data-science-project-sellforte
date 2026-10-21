# Pulse trains: a channel switched off repeatedly for pauses of similar length.
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MIN_DAYS = 7          # shortest pause that counts
MIN_RUNS = 2          # pauses needed to call it a train
LEN_RATIO = 0.5       # a pause joins a group if <= (1 + LEN_RATIO) x the group's median length
MAX_GAP_MULT = 6      # pauses further apart than this x the shorter pause start a new train

Span = tuple[pd.Timestamp, pd.Timestamp]
Run = tuple[pd.Timestamp, pd.Timestamp, bool]  # (start, end, censored at a data edge)


@dataclass(frozen=True)
class PulseTrain:
    start: pd.Timestamp
    end: pd.Timestamp
    components: tuple[Span, ...]
    n_pulses: int


def off_runs(mask: pd.Series, present: pd.Series | None = None) -> list[Run]:
    # Maximal off runs as (start, end, censored); missing days trimmed off the ends.
    m = mask.to_numpy(bool)
    pr = np.ones(len(m), bool) if present is None else present.astype(bool).to_numpy()
    edges = np.flatnonzero(np.diff(np.r_[0, m.astype(int), 0]))
    runs = []
    for lo, hi in zip(edges[::2], edges[1::2]):
        a, b = lo, hi
        while a < b and not pr[a]:
            a += 1
        while b > a and not pr[b - 1]:
            b -= 1
        if a < b:
            runs.append((mask.index[a], mask.index[b - 1], lo == 0 or hi == len(m)))
    return runs


def _days(span) -> int:
    return (span[1] - span[0]).days + 1


def _similar(lengths) -> bool:
    # Every length within (1 + LEN_RATIO)x of the group's median, either way.
    med = float(np.median(lengths))
    return max(lengths) <= med * (1 + LEN_RATIO) and min(lengths) * (1 + LEN_RATIO) >= med


def _close(a, b) -> bool:
    return (b[0] - a[1]).days - 1 <= MAX_GAP_MULT * min(_days(a), _days(b))


def find_pulse_trains(runs: list[Run]) -> list[PulseTrain]:
    runs = [r for r in runs if _days(r) >= MIN_DAYS]
    inner = sorted((r for r in runs if not r[2]), key=lambda r: (_days(r), r[0]))
    edge = [r for r in runs if r[2]]
    # Longest sorted stretch passing the median rule as a whole; ponytail: O(n^2) in pauses.
    groups, i = [], 0
    while i < len(inner):
        j = max(k for k in range(i + 1, len(inner) + 1) if _similar([_days(r) for r in inner[i:k]]))
        groups.append(inner[i:j])
        i = j
    trains = []
    for group in groups:
        chunks: list[list[Run]] = []
        for run in sorted(group):
            if not chunks or not _close(chunks[-1][-1], run):
                chunks.append([])
            chunks[-1].append(run)
        for c in (c for c in chunks if len(c) >= MIN_RUNS):
            # Censored edge pause: may extend a train (length is a lower bound), never start one.
            cap = float(np.median([_days(r) for r in c])) * (1 + LEN_RATIO)
            for e in edge:
                if _days(e) <= cap and (_close(e, c[0]) if e[1] < c[0][0] else _close(c[-1], e)):
                    c = sorted(c + [e])
            trains.append(PulseTrain(c[0][0], c[-1][1], tuple((r[0], r[1]) for r in c), len(c)))
    return sorted(trains, key=lambda t: t.start)


def pulse_probability(p_off: pd.Series, components, runs: list[Run]) -> float:
    # Mean P(off) over the pauses x (1 - CV of pause lengths) x coverage of the window.
    days = pd.concat([p_off.loc[a:b] for a, b in components])
    lengths = np.array([_days(c) for c in components], dtype=float)
    lo, hi = components[0][0], components[-1][1]
    inside = sum(_days((max(a, lo), min(b, hi))) for a, b, _ in runs
                 if _days((a, b)) >= MIN_DAYS and a <= hi and b >= lo)
    coverage = lengths.sum() / inside if inside else 1.0
    return float(days.mean() * (1 - min(1.0, lengths.std() / lengths.mean())) * min(1.0, coverage))


def off_spells(p: pd.Series, s: pd.Series, present: pd.Series | None, cut: float) -> list[Run]:
    # Off runs whose median spend (days with data) is <= cut.
    ok = s if present is None else s[present.astype(bool)]
    return [r for r in off_runs(p >= 0.5, present) if ok.loc[r[0]:r[1]].median() <= cut]
