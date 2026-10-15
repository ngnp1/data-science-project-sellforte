"""Stress test: synthetic spend series with one known difficulty each.

    python -m pulse.stress

Prints a markdown table: expected vs found pauses and the pulse probability.
Synthetic only -- never reads the benchmark's test split or spec.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pulse.onoff import off_mask, p_off
from pulse.pulse import find_pulse_trains, off_runs, pulse_probability

DAYS = pd.date_range("2024-01-01", periods=365)


def series(pauses, noise=0.2, pause_level=0.0, seed=0, base=None, spike=False, missing=()):
    """Daily spend ~100 with lognormal noise; each (start, length) pause set to pause_level.

    spike: one day of full spend in the middle of each pause.
    missing: day indices with no data (the detector treats them as off, like detect.py).
    """
    rng = np.random.default_rng(seed)
    s = (100.0 if base is None else base) * rng.lognormal(0, noise, len(DAYS))
    for start, n in pauses:
        lvl = pause_level(rng, n) if callable(pause_level) else pause_level
        s[start:start + n] = lvl
        if spike:
            s[start + n // 2] = 100.0
    s[list(missing)] = 0.0
    return pd.Series(s, index=DAYS)


def regular(n, length, first=60, gap=28):
    return [(first + i * (length + gap), length) for i in range(n)]


def step_base():
    b = np.full(len(DAYS), 100.0)
    b[180:] = 300.0  # budget triples mid-year
    return b


def dropouts(seed=1):
    """No pulse: random 1-5 day outages only."""
    rng = np.random.default_rng(seed)
    return [(int(d), int(rng.integers(1, 6))) for d in rng.choice(np.arange(10, 350), 12, replace=False)]


# name, pauses that are the true pulse (None = should find nothing), series kwargs
CASES = [
    ("clean: 3 x 14d, exact zero", regular(3, 14), {}),
    ("partial: spend at 10% in pauses", regular(3, 14), {"pause_level": 10.0}),
    ("partial: spend at 25% in pauses", regular(3, 14), {"pause_level": 25.0}),
    ("near-zero noise 0-12% in pauses", regular(3, 14),
     {"pause_level": lambda rng, n: rng.uniform(0, 12, n)}),
    ("uneven 7/11/9", [(60, 7), (95, 11), (134, 9)], {}),
    ("alternating 7/11/7/11", [(60, 7), (95, 11), (134, 7), (169, 11)], {}),
    ("irregular gaps 20/60/35", [(60, 14), (94, 14), (168, 14), (217, 14)], {}),
    ("1-day spend spike inside each pause", regular(3, 14), {"spike": True}),
    ("train starts on day 0", [(0, 14), (42, 14), (84, 14)], {}),
    ("train runs to the last day", [(281, 14), (323, 14), (351, 14)], {}),
    ("3 missing days scattered in on-periods", regular(3, 14), {"missing": (20, 100, 300)}),
    ("2 missing days inside a pause's flank", regular(3, 14), {"missing": (74, 75)}),
    ("only 2 pauses", regular(2, 14), {}),
    ("high noise (sigma 0.6)", regular(3, 14), {"noise": 0.6}),
    ("very high noise (sigma 1.0)", regular(3, 14), {"noise": 1.0}),
    ("pulse across a 3x budget step", [(150, 14), (192, 14), (234, 14)], {"base": step_base()}),
    ("short 5-day pauses (below MIN_DAYS)", regular(4, 5), {}),
]
NEGATIVES = [
    ("lone 40-day pause", [(150, 40)], {}),
    ("2 pauses far apart (gap > 6x)", [(40, 10), (250, 10)], {}),
    ("random 1-5 day dropouts", dropouts(), {}),
    ("very high noise, no pauses", [], {"noise": 1.0}),
]


def overlap(a, b):
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    inter = max(0, (hi - lo).days + 1)
    return inter / ((max(a[1], b[1]) - min(a[0], b[0])).days + 1)


def score(pauses, **kw):
    """(trains found, true pauses matched at IoU >= 0.5, max boundary error, probabilities)."""
    s = series(pauses, **kw)
    trains = find_pulse_trains(off_runs(off_mask(s)))
    p = p_off(s)
    probs = [pulse_probability(p, t.components) for t in trains]
    truth = [(DAYS[a], DAYS[a + n - 1]) for a, n in pauses]
    found = [c for t in trains for c in t.components]
    pairs = [(t, max(found, key=lambda f: overlap(t, f))) for t in truth if found]
    pairs = [(t, f) for t, f in pairs if overlap(t, f) >= 0.5]
    err = max((max(abs((t[0] - f[0]).days), abs((t[1] - f[1]).days)) for t, f in pairs), default=None)
    return len(trains), len(pairs), err, probs


def main():
    rows = ["| case | true pauses | trains found | pauses matched | max boundary error (days) | pulse probability |",
            "|---|---|---|---|---|---|"]
    for name, pauses, kw in CASES:
        n, matched, err, probs = score(pauses, **kw)
        rows.append(f"| {name} | {len(pauses)} | {n} | {matched}/{len(pauses)} | "
                    f"{'-' if err is None else err} | {', '.join(f'{x:.2f}' for x in probs) or '-'} |")
    for name, pauses, kw in NEGATIVES:
        n, _, _, probs = score(pauses, **kw)
        rows.append(f"| NO PULSE: {name} | 0 | {n} | - | - | {', '.join(f'{x:.2f}' for x in probs) or '-'} |")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
