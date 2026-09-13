"""The ten metrics of spec section 9.

Two of them are easy to get quietly wrong and are worth naming here:

- Channel and market accuracy are computed under a RELAXED match that ignores
  type, country and channel. Under the strict match a channel mix-up becomes a
  false negative plus a false positive, the pair never appears as a matched
  couple, and the accuracy reads a meaningless 100%.
- False-positive rate is per country-year, not per scenario, so scenarios of
  different sizes are comparable. It is only measurable at all because the
  benchmark carries 9 null scenarios with no events.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from benchmark.eval.matching import MatchResult, match_events
from benchmark.eval.model import Event


def prf(n_tp: int, n_fp: int, n_fn: int) -> dict:
    precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) else 0.0
    recall = n_tp / (n_tp + n_fn) if (n_tp + n_fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return {"precision": precision, "recall": recall, "f1": f1}


def event_level(truth: list[Event], pred: list[Event]) -> dict:
    res = match_events(truth, pred)
    out = prf(res.n_tp, res.n_fp, res.n_fn)
    out.update(n_tp=res.n_tp, n_fp=res.n_fp, n_fn=res.n_fn)
    return out


def per_type(truth: list[Event], pred: list[Event]) -> dict[str, dict]:
    types = sorted({e.event_type for e in truth} | {e.event_type for e in pred})
    return {
        t: event_level([e for e in truth if e.event_type == t],
                       [e for e in pred if e.event_type == t])
        for t in types
    }


def iou_stats(result: MatchResult) -> dict:
    vals = [m.iou for m in result.matches]
    if not vals:
        return {"mean_iou": 0.0, "median_iou": 0.0, "n": 0}
    return {"mean_iou": float(np.mean(vals)),
            "median_iou": float(np.median(vals)),
            "n": len(vals)}


def boundary_error(result: MatchResult) -> dict:
    if not result.matches:
        return {"start_median": 0.0, "start_p90": 0.0,
                "end_median": 0.0, "end_p90": 0.0, "n": 0}
    starts = np.array([abs((m.pred.start - m.truth.start).days)
                       for m in result.matches], dtype=float)
    ends = np.array([abs((m.pred.end - m.truth.end).days)
                     for m in result.matches], dtype=float)
    return {
        "start_median": float(np.median(starts)),
        "start_p90": float(np.percentile(starts, 90)),
        "end_median": float(np.median(ends)),
        "end_p90": float(np.percentile(ends, 90)),
        "n": len(result.matches),
    }


def channel_and_market_accuracy(truth: list[Event], pred: list[Event]) -> dict:
    res = match_events(truth, pred, strict=False)
    if not res.matches:
        return {"channel_accuracy": 0.0, "market_accuracy": 0.0,
                "n_relaxed_matches": 0}
    same_channel = sum(m.truth.channel == m.pred.channel for m in res.matches)
    same_market = sum(m.truth.country_code == m.pred.country_code
                      for m in res.matches)
    n = len(res.matches)
    return {"channel_accuracy": same_channel / n,
            "market_accuracy": same_market / n,
            "n_relaxed_matches": n}


def _day_keys(events: list[Event]) -> set[tuple]:
    """Every (sid, country, channel, day) an event covers."""
    out: set[tuple] = set()
    for e in events:
        for day in range(e.n_days):
            out.add((e.sid, e.country_code, e.channel,
                     (e.start + np.timedelta64(day, "D"))))
    return out


def day_level(truth: list[Event], pred: list[Event]) -> dict:
    t_days = _day_keys(truth)
    p_days = _day_keys(pred)
    tp = len(t_days & p_days)
    return prf(tp, len(p_days - t_days), len(t_days - p_days))


def type_confusion(truth: list[Event], pred: list[Event]) -> dict:
    """Which truth type got called which detector label, over relaxed matches
    that agree on country and channel -- so this isolates TYPE errors rather
    than mixing them with localisation errors."""
    res = match_events(truth, pred, strict=False)
    counts: Counter = Counter()
    for m in res.matches:
        if (m.truth.country_code == m.pred.country_code
                and m.truth.channel == m.pred.channel):
            counts[(m.truth.event_type, m.pred.event_type)] += 1
    return dict(counts)


def reliability_curve(truth: list[Event], pred: list[Event],
                      n_bins: int = 10) -> list[dict]:
    """Spec section 9 item 8: confidence bin against EMPIRICAL precision.

    This is what converts `detection_confidence` from an assertion into a
    testable claim -- "events at confidence 0.9 are correct about 90% of the
    time". Detections without a confidence are skipped rather than assumed
    confident, so an unscored detector yields an empty curve instead of a
    misleadingly perfect one.
    """
    scored = [p for p in pred if p.detection_confidence is not None]
    if not scored:
        return []

    matched = {id(m.pred) for m in match_events(truth, pred).matches}
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    out = []
    for i in range(n_bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        # Last bin is closed on the right so confidence 1.0 is counted.
        in_bin = [p for p in scored
                  if lo <= p.detection_confidence < hi
                  or (i == n_bins - 1 and p.detection_confidence == 1.0)]
        if not in_bin:
            continue
        correct = sum(id(p) in matched for p in in_bin)
        out.append({
            "bin_lo": lo, "bin_hi": hi, "n": len(in_bin),
            "mean_confidence": float(np.mean([p.detection_confidence
                                              for p in in_bin])),
            "empirical_precision": correct / len(in_bin),
        })
    return out


def operating_curve(truth: list[Event], pred: list[Event],
                    cuts: list[float] | None = None) -> list[dict]:
    """Spec section 9 item 9: sweep the confidence cut and report P/R at each.

    One operating point is not a result. This is what lets the handover say
    "at cut 0.5 you get P=x R=y; at 0.8, P=x' R=y'" instead of implying the
    detector has a single fixed accuracy.
    """
    if cuts is None:
        cuts = [round(c, 2) for c in np.arange(0.0, 1.0, 0.05)]

    out = []
    for cut in cuts:
        kept = [p for p in pred
                if p.detection_confidence is None
                or p.detection_confidence >= cut]
        scores = event_level(truth, kept)
        out.append({"cut": float(cut), "n_pred": len(kept), **scores})
    return out


def false_positive_rate(pred: list[Event], country_years: float) -> float:
    """Events reported per country-year. Used on the null scenarios, where every
    prediction is by construction a false positive."""
    if country_years <= 0:
        return 0.0
    return len(pred) / country_years


def evaluate_scenario(truth: list[Event], pred: list[Event]) -> dict:
    strict = match_events(truth, pred)
    return {
        "event_level": event_level(truth, pred),
        "reliability": reliability_curve(truth, pred),
        "operating": operating_curve(truth, pred),
        "per_type": per_type(truth, pred),
        "iou": iou_stats(strict),
        "boundary": boundary_error(strict),
        "accuracy": channel_and_market_accuracy(truth, pred),
        "day_level": day_level(truth, pred),
        "confusion": type_confusion(truth, pred),
    }
