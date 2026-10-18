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


# The modules that provide detectors, in order; each exposes a DETECTORS list.
# Ours comes first, so it is the default. A detector module that is not listed
# here is never imported and never registers -- tests/detectors/test_registry.py
# fails if a file in this package is missing from the list.
MODULES = ("detectors.rule_based",)


def _load() -> None:
    import importlib
    for name in MODULES:
        for detector in importlib.import_module(name).DETECTORS:
            register(detector)


_load()
