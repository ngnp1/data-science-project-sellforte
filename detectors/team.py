"""The teammates' algorithms (the `team` package) as one detector.

Each period is taken from the teammate who owns it, converted to the shared
DetectedEvent contract and tagged with its author. The algorithms themselves
are not changed here.

team.pulse.detect and team.step_change build the evaluation harness's Event
record type. That is a plain dataclass, not the answer key; no truth loader
comes with it.
"""
from __future__ import annotations

from dataclasses import replace

import pandas as pd

from detection.io.panel import build_panel
from detection.model import EVENT_TYPES, DetectedEvent
from detection.scoring import score_findings
from detectors.base import Detector
from team.pulse.detect import detect as triet
from team.single_channel import run_detection as nam_phu
from team.staggered_launch import run_detection as tvh_ds

# Teammate -> (their entry point, the periods taken from it). Chosen by the
# development split: Triet's detector is the strongest on steps and holdouts,
# Nam Phu's on single-channel periods; tvh-ds's launches match the others and
# carry readiness checks. Thai's and David's modules stay in the team package.
SOURCES = (
    ("Nam Phu", lambda media, sales: nam_phu(media, sales, ""), {"single_channel"}),
    ("tvh-ds", lambda media, sales: tvh_ds(media, sales, ""), {"staggered_launch"}),
    ("Triet", lambda media, sales: triet(media, sales, ""),
     {"channel_pulse", "dark_period", "natural_holdout", "step_change"}),
)


def _drop_renamed_holdouts(findings: list[DetectedEvent]) -> list[DetectedEvent]:
    """In a two-channel market one channel pausing while the other runs is
    both a holdout and a single-channel period. When Triet's holdout covers
    exactly the window of Nam Phu's single-channel period, keep only the
    single-channel name -- the benchmark's convention for that shape."""
    singles = {(f.country_code, f.start, f.end) for f in findings
               if f.event_type == "single_channel"}
    return [f for f in findings if not (f.event_type == "natural_holdout"
                                        and (f.country_code, f.start, f.end) in singles)]


def _convert(event, owner: str) -> DetectedEvent:
    if isinstance(event, DetectedEvent):
        # Already our type (tvh-ds): keep its evidence, e.g. the readiness checks.
        return replace(event, sid="", evidence={**event.evidence, "source": owner})
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
        evidence={"source": owner, "source_confidence": getattr(event, "detection_confidence", None)},
        detection_confidence=getattr(event, "detection_confidence", None),
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
        findings = _drop_renamed_holdouts(findings)
        # Rank, check and explain them exactly as our library does, so the app
        # treats both detectors alike. Who found each one stays in evidence
        # only; the app never shows it.
        return score_findings(findings, build_panel(media, sales))


DETECTORS = [TeamDetector()]
