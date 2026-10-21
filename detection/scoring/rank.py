"""The order findings are shown in."""
from __future__ import annotations

from detection.model import DetectedEvent


def ranked(events: list[DetectedEvent]) -> list[DetectedEvent]:
    """Most informative first; confidence breaks ties; unscored events last."""
    def key(e: DetectedEvent):
        info, conf = e.informativeness, e.detection_confidence
        return (info is None, -(info or 0.0), conf is None, -(conf or 0.0),
                str(e.country_code), e.start)
    return sorted(events, key=key)
