"""Detection package for identifying informative periods in marketing data.

This package implements detection algorithms for:
- Single-channel periods
- Channel pulses
"""

from .algorithms import (
    Event,
    run_detection,
)

__all__ = [
    "Event",
    "run_detection",
]
