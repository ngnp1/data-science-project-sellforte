"""Per-axis slices of a split's results.

Only `noise_level` is stratified within family, so it is the only axis whose
breakdown is causally interpretable. `trend_p` and `market_spread` are
confounded with family on BOTH splits -- 8 of 10 dev families and 7 of 10 test
families sit at a single trend level, and dev's trend=1.0 bucket contains no
null scenarios at all, so precision there has no false-positive denominator.
Reporting those as axis effects would be reporting family difficulty under
another name, so they carry a warning that travels with the numbers.
"""
from __future__ import annotations

from collections import defaultdict

from benchmark.eval.metrics import prf

CONFOUNDED_AXES: frozenset[str] = frozenset({"trend_p", "market_spread"})

_WARNING = (
    "CONFOUNDED WITH FAMILY on both splits -- most families sit at a single "
    "value of this axis, so differences here measure family difficulty, not "
    "the axis. Do not report as an axis effect. See BENCHMARK.md."
)


def breakdown_by(per_scenario: dict, meta_by_sid: dict, axis: str) -> dict:
    buckets: dict = defaultdict(lambda: {"n_tp": 0, "n_fp": 0, "n_fn": 0,
                                         "n_scenarios": 0})
    for sid, result in per_scenario.items():
        meta = meta_by_sid.get(sid)
        if meta is None or axis not in meta:
            continue
        b = buckets[meta[axis]]
        ev = result["event_level"]
        b["n_tp"] += ev["n_tp"]
        b["n_fp"] += ev["n_fp"]
        b["n_fn"] += ev["n_fn"]
        b["n_scenarios"] += 1

    out = {}
    for value, b in buckets.items():
        entry = dict(b)
        entry.update(prf(b["n_tp"], b["n_fp"], b["n_fn"]))
        out[value] = entry
    out["_warning"] = _WARNING if axis in CONFOUNDED_AXES else ""
    return out


# --- Event-level breakdowns (spec section 9 item 10's remaining three axes) --
#
# `breakdown_by` above buckets whole SCENARIOS on scenario-level meta.json
# keys. meta.json carries no duration, no magnitude and no zero-kind, so three
# of spec section 9 item 10's axes are unreachable from it -- but `Event`
# already carries `n_days` and `multiplier` (BENCHMARK.md: magnitude must be
# read per-event from scenario.json, which truth.py already does), so they are
# reachable per EVENT.
#
# These are RECALL-only by construction. A truth event is either matched or it
# is not, which gives TP and FN; a false positive belongs to no truth bucket,
# so precision has no denominator here and is deliberately not reported rather
# than reported wrong.

DURATION_BANDS: tuple[tuple[int | None, str], ...] = (
    (14, "<14 days"),
    (42, "14-41 days"),
    (90, "42-89 days"),
    (None, "90+ days"),
)

# Holdout magnitudes are drawn from [0, 0, 0.02, 0.05, 0.08] and step changes
# from multipliers on either side of 1.0, so these bands separate "the signal
# is gone" from "the signal is faint" from "the level moved".
MAGNITUDE_BANDS: tuple[str, ...] = (
    "exact zero", "near zero (0 < m < 0.1)", "reduced (0.1 <= m < 1)",
    "amplified (m >= 1)",
)

EVENT_AXES: tuple[str, ...] = ("duration", "magnitude", "zero_kind")


def _duration_band(e) -> str:
    for upper, label in DURATION_BANDS:
        if upper is None or e.n_days < upper:
            return label
    return DURATION_BANDS[-1][1]


def _magnitude_band(e) -> str | None:
    m = e.multiplier
    if m is None:
        return None
    if m == 0.0:
        return MAGNITUDE_BANDS[0]
    if m < 0.1:
        return MAGNITUDE_BANDS[1]
    if m < 1.0:
        return MAGNITUDE_BANDS[2]
    return MAGNITUDE_BANDS[3]


def _zero_kind(e) -> str | None:
    """Exact zero versus near zero -- spec section 9 item 10's last axis, and
    the hardest distinction in the benchmark: a channel that went truly silent
    versus one running at 2-8% of normal. Events that are neither are excluded
    rather than lumped into a third bucket, because the axis is a comparison
    between those two cases and nothing else."""
    m = e.multiplier
    if m is None or m >= 0.1:
        return None
    return "exact zero" if m == 0.0 else "near zero"


_AXIS_FN = {"duration": _duration_band, "magnitude": _magnitude_band,
            "zero_kind": _zero_kind}
_AXIS_ORDER = {
    "duration": [label for _, label in DURATION_BANDS],
    "magnitude": list(MAGNITUDE_BANDS),
    "zero_kind": ["exact zero", "near zero"],
}


def event_breakdown(match_result, axis: str) -> list[dict]:
    """Truth events bucketed by a property of the EVENT, with how many of each
    bucket the detector found. Rows come back in the axis's natural order, not
    alphabetically, so `<14 days` is not filed between `14-41` and `90+`."""
    if axis not in _AXIS_FN:
        raise ValueError(f"unknown event axis {axis!r}")
    bucket_of = _AXIS_FN[axis]

    matched_ids = {id(m.truth) for m in match_result.matches}
    all_truth = [m.truth for m in match_result.matches] + \
        list(match_result.unmatched_truth)

    counts: dict[str, dict] = defaultdict(lambda: {"n_events": 0, "n_matched": 0})
    for e in all_truth:
        label = bucket_of(e)
        if label is None:
            continue
        counts[label]["n_events"] += 1
        counts[label]["n_matched"] += int(id(e) in matched_ids)

    order = _AXIS_ORDER[axis]
    labels = [l for l in order if l in counts] + \
        sorted(l for l in counts if l not in order)
    return [
        {"value": l,
         "n_events": counts[l]["n_events"],
         "n_matched": counts[l]["n_matched"],
         "recall": counts[l]["n_matched"] / counts[l]["n_events"]}
        for l in labels
    ]


def event_breakdowns(match_result) -> dict[str, list[dict]]:
    return {axis: event_breakdown(match_result, axis) for axis in EVENT_AXES}
