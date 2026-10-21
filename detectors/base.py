"""The Detector interface.

A detector reads the two input CSVs as data frames -- media (required) and
sales (optional) -- and returns findings as `detection.model.DetectedEvent`,
the shared output contract. Required on every finding: market, event type,
inclusive dates inside the data, the channel (None only for a dark period)
and, for a pulse, its off-windows. Scores, data-quality verdicts,
explanations and evidence are optional: a viewer shows what is there and says
what is not.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from detection.model import DetectedEvent


class Detector(ABC):
    #: Stable identifier, used by the registry and the harness.
    key: str
    #: Name shown to people.
    name: str
    #: The event types this detector can report; a subset of EVENT_TYPES.
    event_types: frozenset[str]

    @abstractmethod
    def detect(self, media: pd.DataFrame,
               sales: pd.DataFrame | None) -> list[DetectedEvent]:
        """Findings for one dataset."""
