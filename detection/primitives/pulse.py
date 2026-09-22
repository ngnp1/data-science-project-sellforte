"""Group nearby off-runs of similar duration into pulse trains.

Repeated pauses need not be unusually long compared with each other. Short
routine gaps remain excluded; unrelated long shutdowns form separate groups.
"""
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from detection import params
from detection.primitives.zero_runs import OffRun

@dataclass(frozen=True)
class PulseTrain:
    start: pd.Timestamp
    end: pd.Timestamp
    components: tuple[tuple[pd.Timestamp, pd.Timestamp], ...]
    n_pulses: int


def find_pulse_trains(runs: list[OffRun]) -> list[PulseTrain]:
    eligible = sorted((r for r in runs if r.n_days >= params.MIN_DAYS
                       and not r.touches_start and not r.touches_end),
                      key=lambda r: (r.n_days, r.start))
    groups: list[list[OffRun]] = []
    for run in eligible:
        if not groups or run.n_days > groups[-1][0].n_days * (1 + params.PULSE_LEN_IQR_RATIO):
            groups.append([])
        groups[-1].append(run)
    trains = []
    for group in groups:
        chunks: list[list[OffRun]] = []
        for run in sorted(group, key=lambda r: r.start):
            if (not chunks or (run.start - chunks[-1][-1].end).days - 1
                    > params.PULSE_MAX_GAP_MULT * min(run.n_days, chunks[-1][-1].n_days)):
                chunks.append([])
            chunks[-1].append(run)
        for chunk in chunks:
            if len(chunk) >= params.PULSE_MIN_RUNS:
                trains.append(PulseTrain(chunk[0].start, chunk[-1].end,
                    tuple((r.start, r.end) for r in chunk), len(chunk)))
    return sorted(trains, key=lambda t: t.start)
