"""Pipeline runner module for detection.

Provides the standardized `run_detection(media_df, sales_df, sid)` entrypoint
for evaluation and production usage.
"""

from __future__ import annotations

from .algorithms import (
    Event,
    run_detection,
)

__all__ = [
    "Event",
    "run_detection",
]
