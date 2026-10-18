"""Registered detectors, in registration order. The first is the default."""
from __future__ import annotations

from detectors.base import Detector

_DETECTORS: dict[str, Detector] = {}


def register(detector: Detector) -> Detector:
    if detector.key in _DETECTORS:
        raise ValueError(f"a detector with key {detector.key!r} is already registered")
    _DETECTORS[detector.key] = detector
    return detector


def get(key: str) -> Detector:
    try:
        return _DETECTORS[key]
    except KeyError:
        raise KeyError(f"no detector {key!r}; available: {', '.join(_DETECTORS)}") from None


def available() -> list[Detector]:
    return list(_DETECTORS.values())


def default() -> Detector:
    return next(iter(_DETECTORS.values()))


# Our library is always available, and first, so it is the default.
from detectors.rule_based import RuleBasedDetector  # noqa: E402

register(RuleBasedDetector())
