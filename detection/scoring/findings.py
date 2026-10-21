"""Scoring and data-quality checks for findings from any detector.

`score_findings(findings, panel)` takes findings that only say what was found
(market, channel, type, dates) and adds everything the app and the harness
read: cross-market comparison notes, edge-of-data and overlap flags, the
confidence and informativeness scores with their sub-scores, the data-quality
verdict, launch readiness checks and a plain-language explanation.

Our pipeline calls it after detection, and so can any other detector
(detectors/team.py does), so every detector's findings are ranked, checked
and explained the same way. The rules themselves live in score.py,
validity.py and explain.py beside this file, and in compose/.
"""
from __future__ import annotations

from dataclasses import replace

from detection.compose.cross_market import annotate
from detection.compose.staggered_launch.readiness import verify_staggered_launches
from detection.scoring.explain import explain
from detection.io.panel import Panel
from detection.model import DetectedEvent
from detection.scoring.score import confidence, informativeness
from detection.scoring.validity import assess


def score_findings(events: list[DetectedEvent], panel: Panel) -> list[DetectedEvent]:
    """Everything after detection: cross-market notes, edge and overlap flags,
    the scores, the validity gate, launch readiness and the explanation.

    Separate from run_detection so findings from another detector (the Team
    detector) can be ranked and explained the same way. `annotate` ignores
    staggered launches, so passing them in here changes nothing for them.
    """
    events = annotate(events, panel)

    annotated = []
    for e in events:
        overlapping = any(other is not e and other.country_code == e.country_code
                          and other.start <= e.end and e.start <= other.end for other in events)
        evidence = dict(e.evidence)
        evidence.update(censored_start=e.start == panel.dates[0],
                        censored_end=e.end == panel.dates[-1])
        tags = tuple(dict.fromkeys((*e.tags, *(("confounded",) if overlapping else ()))))
        annotated.append(replace(e, evidence=evidence, tags=tags))

    scored = []
    for e in annotated:
        raw, parts = confidence(e, panel)
        info, drivers = informativeness(e, panel)
        verdict, reasons = assess(e, panel)
        evidence = dict(e.evidence)
        evidence.update(confidence_sub_scores=parts,
                        informativeness_drivers=drivers,
                        raw_confidence=raw, confidence_kind="heuristic")
        e = replace(
            e,
            detection_confidence=raw,
            informativeness=info,
            validity=verdict,
            validity_reasons=reasons,
            evidence=evidence,
        )
        scored.append(e)
    events = [replace(e, explanation=explain(e, panel))
              for e in verify_staggered_launches(scored, panel)]

    events.sort(key=lambda e: (str(e.country_code), e.start, str(e.channel),
                               e.event_type))
    return events
