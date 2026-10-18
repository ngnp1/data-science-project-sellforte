"""Our rule-based library (`detection/`) as a Detector."""
from __future__ import annotations

import pandas as pd

from detection.model import EVENT_TYPES, DetectedEvent
from detection.pipeline import run_detection
from detectors.base import Detector


class RuleBasedDetector(Detector):
    key = "rule_based"
    name = "Rule-based (ours)"
    event_types = EVENT_TYPES

    def detect(self, media: pd.DataFrame,
               sales: pd.DataFrame | None) -> list[DetectedEvent]:
        return run_detection(media, sales, sid="")


DETECTORS = [RuleBasedDetector()]
