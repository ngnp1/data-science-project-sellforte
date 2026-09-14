"""Convert the detector's output into the harness's Event type.

The dependency points THIS way on purpose. `detection/` is the deliverable and
must stay blind to the harness -- tests/eval/test_gating.py asserts no file
there even mentions the harness package or a truth path. So the detector emits
its own `DetectedEvent` and the harness, which is allowed to know everything,
adapts.

Two fields are deliberately dropped on the way across. `evidence` is a free-form
diagnostic dict with no place in the scored vocabulary, and the section 8 scores
(`detection_confidence`, `informativeness`) are left None because nothing
computes them yet; a placeholder would be indistinguishable from a real score
in the reliability curve.
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
