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
