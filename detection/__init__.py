"""Detection package for identifying informative periods in marketing data."""

from .algorithms import (
    Event,
    run_detection,
)

__all__ = [
    "Event",
    "run_detection",
]
