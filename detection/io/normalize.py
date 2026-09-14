"""The three scales of spec section 5.

Raw euros are never compared across markets: the benchmark spans a 15x range
between its largest and smallest, so a threshold tuned on Germany would be
meaningless in Finland.

- within-series : spend / the series' own active level, then log1p. Scale-free,
  and multiplicative spend noise becomes additive.
- within-market : each channel's share of the market's daily total, so
  composition changes are visible independently of budget swings.
- cross-market  : spend / a market-scale proxy, the long-run median of that
  market's total daily spend.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from detection.io.panel import Panel


def active_level(s: pd.Series) -> float:
    """Median spend over days the channel was actually running.

    Conditioning on positive days is what keeps a long holdout from dragging
    the baseline down and concealing itself.
    """
    positive = s[s > 0]
    if positive.empty:
        return float("nan")
    return float(positive.median())


def scale_free(s: pd.Series) -> pd.Series:
    level = active_level(s)
    if not np.isfinite(level) or level <= 0:
        return pd.Series(0.0, index=s.index)
    return np.log1p(s / level)


def channel_share(panel: Panel, country: str) -> pd.DataFrame:
    cols = [(c, ch) for c, ch in panel.spend.columns if c == country]
    block = panel.spend[cols]
    block.columns = [ch for _, ch in cols]
    total = block.sum(axis=1)
    # A dark day has no total; share is 0 rather than NaN so downstream
    # composition logic never has to special-case it.
    return block.div(total.where(total > 0), axis=0).fillna(0.0)


def market_scale(panel: Panel, country: str) -> float:
    cols = [(c, ch) for c, ch in panel.spend.columns if c == country]
    if not cols:
        return 0.0
    total = panel.spend[cols].sum(axis=1)
    positive = total[total > 0]
    if positive.empty:
        return 0.0
    return float(positive.median())


def cross_market_series(panel: Panel, country: str, channel: str) -> pd.Series:
    scale = market_scale(panel, country)
    s = panel.series(country, channel)
    if scale <= 0:
        return pd.Series(0.0, index=s.index)
    return s / scale
