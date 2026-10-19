# Market-level periods (holdout, single channel, dark, launch) on the on/off model's P(off).
from __future__ import annotations

import numpy as np
import pandas as pd

from pulse.hsmm import p_off_hsmm
from pulse.onoff import RHO, active_level
from pulse.pulse import MIN_DAYS, off_spells

HOLDOUT_MEDIAN = 42.0   # days
HOLDOUT_SIGMA = 0.8


def market_p_off(market: dict) -> dict:
    # {channel: P(off) per day} for the market's active channels (on/off model, holdout prior).
    return {ch: p_off_hsmm(s, pr, HOLDOUT_MEDIAN, HOLDOUT_SIGMA)
            for ch, (s, pr) in market.items() if np.isfinite(active_level(s))}


def find_holdouts(market: dict, pulse_spans: dict, p: dict | None = None) -> list[tuple]:
    # [(channel, start, end, conf)]: one channel off while most others stay on.
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
    # [(survivor, start, end, conf)]: all channels off but one; needs 3+ channels.
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
    # Day x channel: True inside the channel's 15 %-checked off spells.
    off = pd.DataFrame({ch: False for ch in p}, index=next(iter(market.values()))[0].index)
    for ch in p:
        s, present = market[ch]
        for a, b, _ in off_spells(p[ch], s, present, RHO * active_level(s)):
            off.loc[a:b, ch] = True
    return off


def find_dark_periods(market: dict, p: dict | None = None) -> list[tuple]:
    # [(start, end, conf)]: every channel off; needs 2+ channels.
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


def find_staggered_launches(markets: dict, ps: dict) -> list[tuple]:
    # [(country, channel, start, end, conf)]: channel starts >= MIN_DAYS later than in another market.
    delay, spell = {}, {}
    for country, p in ps.items():
        for ch, pc in p.items():
            s, present = markets[country][ch]
            first = [r for r in off_spells(pc, s, present, RHO * active_level(s)) if r[0] == s.index[0]]
            n = (first[0][1] - first[0][0]).days + 1 if first else 0
            delay[country, ch] = n if n >= MIN_DAYS else 0
            if delay[country, ch]:
                spell[country, ch] = first[0]
    out = []
    for (country, ch), (a, b, _) in spell.items():
        n = delay[country, ch]
        if any(o != country and c == ch and d <= n - MIN_DAYS for (o, c), d in delay.items()):
            out.append((country, ch, a, b, float(ps[country][ch].loc[a:b].mean())))
    return out
