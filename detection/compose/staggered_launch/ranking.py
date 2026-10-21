"""Analysis priority and JSON summaries for already-detected launches."""
from __future__ import annotations

from collections.abc import Iterable
from math import isfinite

import pandas as pd

from detection.model import DetectedEvent


_READINESS_ORDER = {
    "analysis_ready": 0,
    "needs_review": 1,
    "not_analysis_ready": 2,
}


def _verification(event: DetectedEvent) -> dict:
    value = event.evidence.get("launch_verification")
    return value if isinstance(value, dict) else {}


def _readiness(event: DetectedEvent) -> str:
    status = _verification(event).get("status")
    return status if status in _READINESS_ORDER else "unassessed"


def _score(value: float | None) -> float | None:
    return float(value) if value is not None and isfinite(value) else None


def _descending_score(value: float | None) -> float:
    value = _score(value)
    return -value if value is not None else float("inf")


def rank_staggered_launches(events: Iterable[DetectedEvent]) -> list[DetectedEvent]:
    """Select all launches and order them by readiness, then existing scores.

    Ready launches precede review cases, known rejections, and unassessed
    launches. Within each group, prefer informativeness, then confidence;
    missing/non-finite scores come last at their respective sort position.
    Identifying fields break ties deterministically. Original event objects
    are returned without mutation. This is a triage policy, not a probability.
    """
    def key(event: DetectedEvent) -> tuple:
        return (
            _READINESS_ORDER.get(_readiness(event), 3),
            _descending_score(event.informativeness),
            _descending_score(event.detection_confidence),
            event.sid, event.country_code or "", event.channel or "",
            event.start, event.end,
        )

    return sorted((e for e in events if e.event_type == "staggered_launch"), key=key)


def _date(value) -> str | None:
    return None if value is None or pd.isna(value) else str(pd.Timestamp(value).date())


def summarize_staggered_launches(events: Iterable[DetectedEvent]) -> list[dict]:
    """Return ranked, JSON-compatible launch records for downstream consumers.

    Missing onset/windows remain null; they are never inferred from the inactive
    interval. Non-finite scores become null only in this export. Original event
    scores and nested readiness evidence remain unchanged.
    """
    rows = []
    for rank, event in enumerate(rank_staggered_launches(events), 1):
        verification = _verification(event)
        windows = verification.get("windows") or {}
        rows.append({
            "rank": rank,
            "sid": event.sid,
            "country_code": event.country_code,
            "channel": event.channel,
            "launch_date": _date(event.evidence.get("onset")),
            "inactive_start": _date(event.start),
            "inactive_end": _date(event.end),
            "readiness": _readiness(event),
            "reason_codes": list(verification.get("reason_codes") or []),
            "qualified_peers": list(verification.get("qualified_peers") or []),
            "windows": {name: _date(windows.get(name)) for name in
                        ("pre_start", "pre_end", "post_start", "post_end")},
            "detection_confidence": _score(event.detection_confidence),
            "informativeness": _score(event.informativeness),
        })
    return rows
