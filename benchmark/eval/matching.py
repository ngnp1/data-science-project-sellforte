"""Maximum-cardinality one-to-one event matching above an IoU threshold.

Start with strongest overlaps, then reassign pairs when that allows more valid
matches. IoU and field agreement determine preferences, not a guarantee of
maximum total IoU among assignments with the same number of matches.
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


def _agreement(t: Event, p: Event) -> int:
    """How many of country, channel and type a candidate pair agrees on.

    Only ever non-uniform under `strict=False`, where the matcher is
    deliberately blind to those three fields. Blindness is required -- it is
    what makes a channel mix-up visible as a channel error rather than as
    FN+FP -- but it must not decide WHICH pair to credit when several tie on
    IoU. Two markets sharing one identical dark window (a `global_pause`, or
    any multi-market dark period) produce exactly that tie, and breaking it on
    event content alone pairs a missed market's truth with another market's
    correct detection whenever the alphabet happens to line up that way. Market
    accuracy then reads 0.000 or 1.000 for the same detector quality depending
    on which country code sorts first. Preferring agreement first makes the
    reading depend on the detector instead.
    """
    return ((t.country_code == p.country_code)
            + (t.channel == p.channel)
            + (t.event_type == p.event_type))


def match_events(truth: list[Event], pred: list[Event], *,
                 min_iou: float = 0.5, strict: bool = True) -> MatchResult:
    """Maximize valid match count with deterministic overlap preferences.

    Strict matching requires scenario, market, channel and type agreement.
    Relaxed matching requires only scenario agreement for label diagnostics.
    """
    candidates = []
    for ti, t in enumerate(truth):
        for pi, p in enumerate(pred):
            if not _compatible(t, p, strict):
                continue
            score = iou(t, p)
            if score >= min_iou:
                candidates.append((score, t, p, ti, pi))

    # Sort by descending IoU, then by descending country/channel/type
    # agreement, then by prediction and truth content for deterministic
    # tie-breaking regardless of caller's input order.
    def sort_key(c):
        score, t, p, ti, pi = c
        return (-score, -_agreement(t, p),
                p.start, p.end, p.country_code or "", p.channel or "",
                p.event_type or "", t.start, t.end, t.country_code or "",
                t.channel or "", t.event_type or "")

    candidates.sort(key=sort_key)

    result = MatchResult()
    # A greedy choice can block two valid pairs. Search alternating paths
    # from unmatched truths, preserving the initial assignment unless another
    # match can be added. Each augmentation increases cardinality by one.
    adjacency: dict[int, list[int]] = {}
    scores = {}
    for score, t, p, ti, pi in candidates:
        adjacency.setdefault(ti, []).append(pi)
        scores[ti, pi] = score
    # Use indices rather than object identity: duplicate event objects count
    # as separate predictions and still must obey one-to-one assignment.
    by_pred = {}
    occupied_truth = set()
    for _, _, _, ti, pi in candidates:
        if ti not in occupied_truth and pi not in by_pred:
            by_pred[pi] = ti
            occupied_truth.add(ti)

    for root in adjacency:
        if root in occupied_truth:
            continue
        queue = [root]
        seen_truth = {root}
        parent_pred = {}
        free = None
        for ti in queue:
            for pi in adjacency[ti]:
                if pi in parent_pred:
                    continue
                parent_pred[pi] = ti
                if pi not in by_pred:
                    free = pi
                    break
                owner = by_pred[pi]
                if owner not in seen_truth:
                    seen_truth.add(owner)
                    queue.append(owner)
            if free is not None:
                break
        if free is None:
            continue
        by_truth = {ti: pi for pi, ti in by_pred.items()}
        while True:
            ti = parent_pred[free]
            previous = by_truth.get(ti)
            by_pred[free] = ti
            if previous is None:
                break
            free = previous
        occupied_truth.add(root)

    used_t = set(by_pred.values())
    used_p = set(by_pred)
    result.matches = [Match(truth[ti], pred[pi], scores[ti, pi])
                      for _, _, _, ti, pi in candidates
                      if by_pred.get(pi) == ti]
    result.unmatched_truth = [t for i, t in enumerate(truth) if i not in used_t]
    result.unmatched_pred = [p for i, p in enumerate(pred) if i not in used_p]
    return result
