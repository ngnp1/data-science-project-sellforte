"""Greedy one-to-one matching of detections to ground truth by temporal IoU.

Spec section 9: matched greedily by descending temporal IoU, one-to-one,
requiring the same country, type and channel, with IoU at least 0.5.

Greedy rather than optimal (Hungarian) on purpose: with a 0.5 threshold no
truth interval can be claimed by two predictions that also overlap each other
enough to change the assignment, so greedy and optimal agree in practice, and
greedy is auditable by hand -- which matters when a reviewer has to explain why
a particular event was scored the way it was.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from benchmark.eval.model import Event


def overlap_days(a: Event, b: Event) -> int:
    """Inclusive-interval overlap, in whole days."""
    lo = max(a.start, b.start)
    hi = min(a.end, b.end)
    return max(0, int((hi - lo).days) + 1)


def iou(a: Event, b: Event) -> float:
    inter = overlap_days(a, b)
    if inter == 0:
        return 0.0
    union = a.n_days + b.n_days - inter
    return inter / union if union else 0.0


@dataclass
class Match:
    truth: Event
    pred: Event
    iou: float


@dataclass
class MatchResult:
    matches: list[Match] = field(default_factory=list)
    unmatched_truth: list[Event] = field(default_factory=list)
    unmatched_pred: list[Event] = field(default_factory=list)

    @property
    def n_tp(self) -> int:
        return len(self.matches)

    @property
    def n_fp(self) -> int:
        return len(self.unmatched_pred)

    @property
    def n_fn(self) -> int:
        return len(self.unmatched_truth)


def _compatible(t: Event, p: Event, strict: bool) -> bool:
    if t.sid != p.sid:
        return False
    if not strict:
        return True
    return (t.country_code == p.country_code
            and t.channel == p.channel
            and t.event_type == p.event_type)


def match_events(truth: list[Event], pred: list[Event], *,
                 min_iou: float = 0.5, strict: bool = True) -> MatchResult:
    """One-to-one assignment, best IoU first.

    Greedy matching by descending temporal IoU. Ties on IoU are broken by
    prediction event content (start, end, country_code, channel, event_type),
    then truth event content, ensuring deterministic results regardless of
    input order.

    `strict=False` ignores country, channel and type, which is what spec
    section 9 item 4 needs: under the strict match a channel mix-up vanishes
    into a false negative plus a false positive and channel accuracy reads a
    meaningless 100%.
    """
    candidates = []
    for ti, t in enumerate(truth):
        for pi, p in enumerate(pred):
            if not _compatible(t, p, strict):
                continue
            score = iou(t, p)
            if score >= min_iou:
                candidates.append((score, t, p, ti, pi))

    # Sort by descending IoU, then by prediction and truth content for
    # deterministic tie-breaking regardless of caller's input order.
    def sort_key(c):
        score, t, p, ti, pi = c
        return (-score, p.start, p.end, p.country_code or "", p.channel or "",
                p.event_type or "", t.start, t.end, t.country_code or "",
                t.channel or "", t.event_type or "")

    candidates.sort(key=sort_key)

    used_t: set[int] = set()
    used_p: set[int] = set()
    result = MatchResult()
    for score, t, p, ti, pi in candidates:
        if ti in used_t or pi in used_p:
            continue
        used_t.add(ti)
        used_p.add(pi)
        result.matches.append(Match(truth=t, pred=p, iou=score))

    result.unmatched_truth = [t for i, t in enumerate(truth) if i not in used_t]
    result.unmatched_pred = [p for i, p in enumerate(pred) if i not in used_p]
    return result
