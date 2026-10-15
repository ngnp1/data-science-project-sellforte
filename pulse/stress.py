"""Stress test: synthetic spend series with one known difficulty each.

    python -m pulse.stress

Prints a markdown table: expected vs found pauses and the pulse probability.
Synthetic only -- never reads the benchmark's test split or spec.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pulse.detect import run_series

DAYS = pd.date_range("2024-01-01", periods=365)


def series(pauses, noise=0.2, pause_level=0.0, seed=0, base=None, spike=False, missing=(), extra=(),
           extra_level=None):
    """Daily spend ~100 with lognormal noise; each (start, length) pause set to pause_level.

    spike: days of full spend in the middle of each pause (True = 1 day, or a count).
    missing: day indices with no data (spend 0 and present False, like detect.py).
    """
    rng = np.random.default_rng(seed)
    s = (100.0 if base is None else base) * rng.lognormal(0, noise, len(DAYS))
    for start, n in pauses:
        lvl = pause_level(rng, n) if callable(pause_level) else pause_level
        s[start:start + n] = lvl
        if spike:
            s[start + n // 2:start + n // 2 + int(spike)] = 100.0
    s[list(missing)] = 0.0
    for start, n in extra:  # off periods that are not part of the pulse
        s[start:start + n] = 0.0
    if extra_level:         # (spans, level): spend set to a level that is NOT a pause
        for start, n in extra_level[0]:
            s[start:start + n] = extra_level[1]
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
    ("near-zero noise 0-12% in pauses", regular(3, 14),
     {"pause_level": lambda rng, n: rng.uniform(0, 12, n)}),
    ("uneven 7/11/9", [(60, 7), (95, 11), (134, 9)], {}),
    ("alternating 7/11/7/11", [(60, 7), (95, 11), (134, 7), (169, 11)], {}),
    ("irregular gaps 20/60/35", [(60, 14), (94, 14), (168, 14), (217, 14)], {}),
    ("1-day spend spike inside each pause", regular(3, 14), {"spike": True}),
    ("2-day spend spike inside each pause", regular(3, 14), {"spike": 2}),
    ("3-day spend spike inside each pause", regular(3, 14), {"spike": 3}),
    # Longer than C's off-length prior (most mass on 7-30 days).
    ("long pauses: 3 x 35d", regular(3, 35, first=40), {}),
    ("long pauses: 3 x 42d", regular(3, 42, first=30), {}),
    ("long pauses: 3 x 56d", regular(3, 56, first=20), {}),
    ("1-day spike inside 3 x 42d pauses", regular(3, 42, first=30), {"spike": True}),
    ("slightly uneven 13/15/14/16", [(60, 13), (101, 15), (144, 14), (186, 16)], {}),
    ("train starts on day 0", [(0, 14), (42, 14), (84, 14)], {}),
    ("train runs to the last day", [(281, 14), (323, 14), (351, 14)], {}),
    ("3 missing days scattered in on-periods", regular(3, 14), {"missing": (20, 100, 300)}),
    ("2 missing days inside a pause's flank", regular(3, 14), {"missing": (74, 75)}),
    ("only 2 pauses", regular(2, 14), {}),
    ("high noise (sigma 0.6)", regular(3, 14), {"noise": 0.6}),
    ("very high noise (sigma 1.0)", regular(3, 14), {"noise": 1.0}),
    # The 30-day pause (110-139) is not part of the 14-day train but sits inside its window.
    ("long 30-day pause inside the train's window", [(60, 14), (88, 14), (150, 14), (178, 14)],
     {"extra": [(110, 30)]}),
    ("pulse across a 3x budget step", [(150, 14), (192, 14), (234, 14)], {"base": step_base()}),
    ("short 5-day pauses (below MIN_DAYS)", regular(4, 5), {}),
]
NEGATIVES = [
    ("lone 40-day pause", [(150, 40)], {}),
    ("2 pauses far apart (gap > 6x)", [(40, 10), (250, 10)], {}),
    ("random 1-5 day dropouts", dropouts(), {}),
    ("very high noise, no pauses", [], {"noise": 1.0}),
    ("short 1-4 day outages every ~3 weeks", [], {"extra": [(30 + 21 * i, 1 + i % 4) for i in range(15)]}),
    # Above the agreed 15% cutoff: a budget cut (step change), not a pause.
    ("spend drops to 25% for 3 x 14d", [], {"extra_level": (regular(3, 14), 25.0)}),
    ("spend drops to 30% for 3 x 14d", [], {"extra_level": (regular(3, 14), 30.0)}),
    ("spend drops to 40% for 3 x 14d", [], {"extra_level": (regular(3, 14), 40.0)}),
    # Half the days at full spend: not a pause (B does not absorb this much on its own).
    ("every other day off, 3 x 14d", [], {"extra_level": ([(a + i, 1) for a, n in regular(3, 14) for i in range(0, n, 2)], 0.0)}),
]


def overlap(a, b):
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    inter = max(0, (hi - lo).days + 1)
    return inter / ((max(a[1], b[1]) - min(a[0], b[0])).days + 1)


def score(pauses, layer="A", **kw):
    """(trains found, true pauses matched at IoU >= 0.5, max boundary error, probabilities)."""
    s = series(pauses, **kw)
    present = pd.Series(True, index=DAYS)
    present.iloc[list(kw.get("missing", ()))] = False
    found_trains = run_series(s, present, layer)
    trains, probs = [t for t, _, _ in found_trains], [p for _, p, _ in found_trains]
    truth = [(DAYS[a], DAYS[a + n - 1]) for a, n in pauses]
    found = [c for t in trains for c in t.components]
    pairs = [(t, max(found, key=lambda f: overlap(t, f))) for t in truth if found]
    pairs = [(t, f) for t, f in pairs if overlap(t, f) >= 0.5]
    err = max((max(abs((t[0] - f[0]).days), abs((t[1] - f[1]).days)) for t, f in pairs), default=None)
    return len(trains), len(pairs), err, probs


def _cell(pauses, layer, kw):
    n, matched, err, probs = score(pauses, layer, **kw)
    if not pauses:
        return f"{n} trains"
    return (f"{matched}/{len(pauses)}, err {'-' if err is None else err}, "
            f"p {', '.join(f'{x:.2f}' for x in probs) or '-'}")


def main():
    """Per case and layer: pauses matched, max boundary error (days), pulse probability."""
    rows = ["| case | layer A | layer B | layer C |", "|---|---|---|---|"]
    for name, pauses, kw in CASES:
        rows.append(f"| {name} | " + " | ".join(_cell(pauses, layer, kw) for layer in "ABC") + " |")
    for name, pauses, kw in NEGATIVES:
        rows.append(f"| NO PULSE: {name} | " + " | ".join(
            f"{score(pauses, layer, **kw)[0]} trains" for layer in "ABC") + " |")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
