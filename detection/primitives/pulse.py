"""P3 -- repeated off-runs on one series, grouped into a single pulse train.

Spec section 7 emits ONE event spanning first start to last end, with the
individual windows attached as components. That shape is not cosmetic: the
benchmark's truth is grouped the same way, and a detector that emits one event
per off-window scores an IoU too low to match against the grouped truth, so a
perfectly working pulse detector would match nothing.

Pulse trains are also the only place adstock decay is observable, which is why
they carry the highest informativeness weight in spec section 8.

KNOWN LIMITATION -- a regular train of six or more windows is never grouped
here. P1 calls a run notable only if it also clears RUN_RATIO x the p90 of the
series' OTHER off-runs, and that rule switches on once five other runs exist;
in a regular train every other run is this run's own length, so the floor
becomes RUN_RATIO times it and nothing clears it. Five windows group, six
group nothing.

What that costs is the GROUPING, not the finding. Regime segmentation in
detection/compose/label.py cuts the timeline on the active-channel set and
never consults notability, so the windows are still emitted -- as one
separate event per window, typed natural_holdout (or dark_period in a
one-channel market) rather than channel_pulse, each carrying the lower type
prior and none carrying the pulse's components. The failure mode to expect on
real flighting data is therefore type error and fragmentation, not silence.
Measured end to end and pinned in tests/detection/test_pipeline.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
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
    notable = [r for r in runs if r.notable and not r.touches_start
               and not r.touches_end]
    if len(notable) < params.PULSE_MIN_RUNS:
        return []

    lengths = np.array([r.n_days for r in notable], dtype=float)
    median = float(np.median(lengths))
    if median <= 0:
        return []
    iqr = float(np.percentile(lengths, 75) - np.percentile(lengths, 25))
    if iqr / median > params.PULSE_LEN_IQR_RATIO:
        # Runs of wildly different length are not one phenomenon: a 7-day gap
        # and a 90-day shutdown do not belong in the same train.
        return []

    ordered = sorted(notable, key=lambda r: r.start)
    return [PulseTrain(
        start=ordered[0].start,
        end=ordered[-1].end,
        components=tuple((r.start, r.end) for r in ordered),
        n_pulses=len(ordered),
    )]
