"""Natural holdouts: one channel stops while the market's other channels keep running.

Built on layer C (pulse/hsmm.py) and the same 15 % definition of "off", with a broader
off-spell prior than pulses: holdout tests typically run from a couple of weeks to a few
months, so the prior is centred on 6 weeks and wide (log-spread 0.8, ~2.5 to 20 weeks).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pulse.hsmm import p_off_hsmm
from pulse.onoff import RHO, active_level
from pulse.pulse import MIN_DAYS, off_spells

HOLDOUT_MEDIAN = 42.0   # days
HOLDOUT_SIGMA = 0.8


def find_holdouts(market: dict, pulse_spans: dict) -> list[tuple]:
    """market: {channel: (spend, present)}; pulse_spans: {channel: [(start, end), ...]}.

    Returns [(channel, start, end, confidence)].
    """
    active = {ch: sp for ch, sp in market.items() if np.isfinite(active_level(sp[0]))}
    p = {ch: p_off_hsmm(s, pr, HOLDOUT_MEDIAN, HOLDOUT_SIGMA) for ch, (s, pr) in active.items()}
    out = []
    for ch, (s, present) in active.items():
        others = [o for o in active if o != ch]
        for a, b, _ in off_spells(p[ch], s, present, RHO * active_level(s)):
            if ((b - a).days + 1 < MIN_DAYS or a == s.index[0]   # too short / from day one: a launch
                    or not others
                    or any(a <= e and pa <= b for pa, e in pulse_spans.get(ch, ()))):
                continue
            on = [1 - p[o].loc[a:b].mean() for o in others]   # how "on" each other channel was
            # One channel stops while the others keep running: most others must stay on.
            if np.mean([x > 0.5 for x in on]) > 0.5:
                out.append((ch, a, b, float(p[ch].loc[a:b].mean() * np.mean(on))))
    return out
