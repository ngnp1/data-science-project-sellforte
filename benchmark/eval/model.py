"""Shared vocabulary for the evaluation harness.

Deliberately logic-free: every other eval module imports these names, so
keeping them in one dependency-free place stops the import graph tangling.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

# Pattern types the generator injects that change the data but must NEVER be
# reported as detections. Anything a detector finds inside their windows is a
# false positive -- that is the point of them.
NON_EVENT_TYPES: frozenset[str] = frozenset({"ramp_block", "intermittent_baseline"})

# Truth pattern_type -> the detector label that should match it.
# Per BENCHMARK.md, `global_pause` is the one genuine rename: truth records it
# as a pattern_type, while spec section 7 labels the regime `dark_period` and
# applies `global_pause` as a cross-market TAG. Matching on the string would
# fail on every one of those events.
TYPE_MAP: dict[str, str] = {
    "dark_period": "dark_period",
    "single_channel": "single_channel",
    "natural_holdout": "natural_holdout",
    "step_change": "step_change",
    "channel_pulse": "channel_pulse",
    "staggered_launch": "staggered_launch",
    "global_pause": "dark_period",
    # Negative controls map to themselves so the loader can recognise and drop
    # them explicitly rather than by silently failing a lookup.
    "ramp_block": "ramp_block",
    "intermittent_baseline": "intermittent_baseline",
}

MATCHABLE_TYPES: frozenset[str] = frozenset(
    label for truth_type, label in TYPE_MAP.items()
    if truth_type not in NON_EVENT_TYPES
)


@dataclass(frozen=True)
class Event:
    """One interval, either from ground truth or from a detector.

    Intervals are INCLUSIVE on both ends. `country_code` may be None for a
    panel-level event that spans markets; `channel` may be None for an event
    that covers every channel (a dark period).
    """
    sid: str
    country_code: str | None
    channel: str | None
    event_type: str
    start: pd.Timestamp
    end: pd.Timestamp
    pattern_id: str | None = None
    multiplier: float | None = None
    # Spec section 8's three scores. Truth events leave these None; detections
    # set them, and spec section 9 items 8 and 9 (the reliability and operating
    # curves) are computed from detection_confidence.
    detection_confidence: float | None = None
    informativeness: float | None = None
    tags: tuple[str, ...] = ()
    # For grouped events (a pulse train), the individual windows that compose it.
    components: tuple[tuple[pd.Timestamp, pd.Timestamp], ...] = ()

    @property
    def n_days(self) -> int:
        return int((self.end - self.start).days) + 1
