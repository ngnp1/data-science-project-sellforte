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


def market_p_off(market: dict) -> dict:
    """{channel: P(off) per day} for the market's active channels, layer C, holdout prior."""
    return {ch: p_off_hsmm(s, pr, HOLDOUT_MEDIAN, HOLDOUT_SIGMA)
            for ch, (s, pr) in market.items() if np.isfinite(active_level(s))}


def find_holdouts(market: dict, pulse_spans: dict, p: dict | None = None) -> list[tuple]:
    """market: {channel: (spend, present)}; pulse_spans: {channel: [(start, end), ...]}.

    Returns [(channel, start, end, confidence)].
    """
    p = market_p_off(market) if p is None else p
    active = {ch: market[ch] for ch in p}
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


def find_single_channel(market: dict, p: dict | None = None) -> list[tuple]:
    """All channels except one stop: runs of >= MIN_DAYS where exactly one active channel is on.

    Needs 3+ active channels: with 2, "all but one stop" and "one stops while the other runs"
    are the same spend pattern, and it is reported as a natural holdout (see README).
    Returns [(surviving channel, start, end, confidence)].
    """
    p = market_p_off(market) if p is None else p
    if len(p) < 3:
        return []
    off = _off_frame(market, p)
    # The one channel still on, on days when all the others are off; None otherwise.
    survivor = (~off).idxmax(axis=1).where(off.sum(axis=1) == len(p) - 1)
    out = []
    for _, run in survivor.groupby((survivor != survivor.shift()).cumsum()):
        ch, a, b = run.iloc[0], run.index[0], run.index[-1]
        if isinstance(ch, str) and len(run) >= MIN_DAYS:
            stopped = np.mean([p[o].loc[a:b].mean() for o in p if o != ch])
            out.append((ch, a, b, float(stopped * (1 - p[ch].loc[a:b].mean()))))
    return out


def _off_frame(market: dict, p: dict) -> pd.DataFrame:
    """Day x channel: True inside the channel's 15 %-checked off spells."""
    off = pd.DataFrame({ch: False for ch in p}, index=next(iter(market.values()))[0].index)
    for ch in p:
        s, present = market[ch]
        for a, b, _ in off_spells(p[ch], s, present, RHO * active_level(s)):
            off.loc[a:b, ch] = True
    return off


def find_dark_periods(market: dict, p: dict | None = None) -> list[tuple]:
    """All channels in a market stop at the same time: runs of >= MIN_DAYS with every active
    channel off (a channel that never spends is already stopped). Needs 2+ channels in the
    market: with one, it is just that channel's pause. Returns [(start, end, confidence)]."""
    p = market_p_off(market) if p is None else p
    if len(market) < 2 or not p:
        return []
    dark = _off_frame(market, p).all(axis=1)
    out = []
    for _, run in dark[dark].groupby((~dark).cumsum()):
        a, b = run.index[0], run.index[-1]
        if len(run) >= MIN_DAYS:
            out.append((a, b, float(np.mean([p[ch].loc[a:b].mean() for ch in p]))))
    return out
