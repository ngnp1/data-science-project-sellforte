"""Adapt detector events to the evaluator without importing answers into detection.

Heuristic scores are not exported as calibrated probabilities. Historical or
explicitly calibrated events can still populate reliability curves.
"""
from __future__ import annotations

from benchmark.eval.model import Event
from detection.model import DetectedEvent


def to_event(d: DetectedEvent) -> Event:
    return Event(
        sid=d.sid,
        country_code=d.country_code,
        channel=d.channel,
        event_type=d.event_type,
        start=d.start,
        end=d.end,
        multiplier=d.magnitude_ratio,
        components=d.components,
        tags=d.tags,
        # Reliability curves require probabilities, not raw evidence scores.
        detection_confidence=(None if d.evidence.get("confidence_kind") == "heuristic"
                              else d.detection_confidence),
        informativeness=d.informativeness,
    )


def for_detector(key: str):
    """The scoreable callable for any registered detector (detectors.registry).

    `run_dev` and `run_final` pass None for both frames unless --load-data is
    given. Returning [] there would look like a detector that found nothing
    rather than a misconfigured run -- and on the one-shot final gate that
    would burn the run and record a 0.0 audit line. So it raises instead.
    An unknown key fails here, before any scenario is scored.
    """
    from dataclasses import replace

    from detectors import registry

    detector = registry.get(key)

    def run(media_df, sales_df, sid: str) -> list[Event]:
        if media_df is None:
            raise ValueError(
                "detect() needs the media frame; pass --load-data to run_dev "
                "or run_final")
        return [to_event(replace(d, sid=sid)) for d in detector.detect(media_df, sales_df)]

    return run


def detect(media_df, sales_df, sid: str) -> list[Event]:
    """Our rule-based library, the harness's default detector:
    `--detector benchmark.eval.adapter:detect`."""
    return for_detector("rule_based")(media_df, sales_df, sid)
