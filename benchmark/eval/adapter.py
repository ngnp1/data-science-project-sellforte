"""Adapt detector events to the evaluator without importing answers into detection.

Heuristic scores are not exported as calibrated probabilities. Historical or
explicitly calibrated events can still populate reliability curves.
"""
from __future__ import annotations

from benchmark.eval.model import Event
from detection.model import DetectedEvent
from detection.pipeline import run_detection


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


def detect(media_df, sales_df, sid: str) -> list[Event]:
    """The scoreable detector callable.

    `run_dev` and `run_final` pass None for both frames unless --load-data is
    given. Returning [] there would look like a detector that found nothing
    rather than a misconfigured run -- and on the one-shot final gate that
    would burn the run and record a 0.0 audit line. So it raises instead.
    """
    if media_df is None:
        raise ValueError(
            "detect() needs the media frame; pass --load-data to run_dev "
            "or run_final")
    return [to_event(d) for d in run_detection(media_df, sales_df, sid)]
