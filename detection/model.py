"""The detector's own output type.

Deliberately independent of the evaluation harness. `detection/` is the
deliverable; the harness is scaffolding that scores it, and a detector that
imported its own scorer's vocabulary could not be trusted to be blind to the
answers. tests/eval/test_gating.py enforces that boundary by asserting no file
here mentions the harness or any truth loader.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

EVENT_TYPES: frozenset[str] = frozenset({
    "dark_period",
    "single_channel",
    "natural_holdout",
    "step_change",
    "channel_pulse",
    "staggered_launch",
})


@dataclass(frozen=True)
class DetectedEvent:
    """One informative period the detector believes it has found.

    Intervals are INCLUSIVE on both ends. `channel` is None for an event that
    covers every channel in a market (a dark period); `country_code` is None
    only for a panel-level event, which the cross-market layer fans out before
    the event reaches the harness.
    """
    sid: str
    country_code: str | None
    channel: str | None
    event_type: str
    start: pd.Timestamp
    end: pd.Timestamp
    magnitude_ratio: float | None = None
    components: tuple[tuple[pd.Timestamp, pd.Timestamp], ...] = ()
    tags: tuple[str, ...] = ()
    evidence: dict = field(default_factory=dict)

    @property
    def n_days(self) -> int:
        return int((self.end - self.start).days) + 1
