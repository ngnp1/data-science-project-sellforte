"""The teammates' algorithms (the `team` package) as one detector.

Each period is taken from the teammate who owns it, converted to the shared
DetectedEvent contract and tagged with its author. The algorithms themselves
are not changed here.

team.pulse.detect and team.step_change build the evaluation harness's Event
record type. That is a plain dataclass, not the answer key; no truth loader
comes with it.
"""
from __future__ import annotations

import pandas as pd

from detection.model import EVENT_TYPES, DetectedEvent
from detectors.base import Detector
from team.holdout import run_detection as thai
from team.pulse.detect import detect as triet
from team.single_channel import run_detection as nam_phu
from team.step_change import detect as david

# Teammate -> (their entry point, the patterns taken from it).
SOURCES = (
    ("Nam Phu", lambda media, sales: nam_phu(media, sales, ""), {"single_channel"}),
    ("Thai", lambda media, sales: thai(media, sales, ""), {"natural_holdout", "staggered_launch"}),
    ("Triet", lambda media, sales: triet(media, sales, ""), {"channel_pulse", "dark_period"}),
    ("David", lambda media, sales: david(media, sales, ""), {"step_change"}),
)


def _convert(event, owner: str) -> DetectedEvent:
    return DetectedEvent(
        sid="",
        country_code=event.country_code,
        channel=event.channel,
        event_type=event.event_type,
        start=pd.Timestamp(event.start),
        end=pd.Timestamp(event.end),
        magnitude_ratio=getattr(event, "multiplier", None),
        components=tuple((pd.Timestamp(a), pd.Timestamp(b))
                         for a, b in (getattr(event, "components", ()) or ())),
        tags=tuple(getattr(event, "tags", ()) or ()),
        evidence={"source": owner},
        detection_confidence=getattr(event, "detection_confidence", None),
        explanation=f"Found by {owner}'s algorithm.",
    )


class TeamDetector(Detector):
    key = "team"
    name = "Team (combined)"
    event_types = EVENT_TYPES

    def detect(self, media: pd.DataFrame,
               sales: pd.DataFrame | None) -> list[DetectedEvent]:
        if media is None or media.empty:
            return []
        findings = []
        for owner, run, patterns in SOURCES:
            findings += [_convert(e, owner) for e in run(media, sales)
                         if e.event_type in patterns]
        return sorted(findings, key=lambda f: (str(f.country_code), f.start, str(f.channel)))


DETECTORS = [TeamDetector()]
