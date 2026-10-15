"""The on/off layer: each day's probability that a channel was off.

A soft cutoff around a hard threshold: spend at or below RHO x the channel's
usual level counts as off. At p = ONOFF_CUT it is exactly that hard cutoff.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RHO = 0.15          # off = spend at most 15% of the channel's usual level
EPS_ABS = 1e-6      # floor, so float noise around zero is never spend
ONOFF_WIDTH = 0.03  # softness of the cutoff, as a fraction of the usual level
ONOFF_CUT = 0.5     # P(off) at or above which a day counts as off


def active_level(s: pd.Series) -> float:
    """Median spend over days the channel actually ran, so long pauses can't hide themselves."""
    positive = s[s > 0]
    return float(positive.median()) if not positive.empty else float("nan")


def p_off(s: pd.Series, present: pd.Series | None = None) -> pd.Series:
    level = active_level(s)
    if not np.isfinite(level) or level <= 0:
        return pd.Series(0.0, index=s.index)  # never active: nothing to switch off
    threshold = max(EPS_ABS, RHO * level)
    z = np.clip((threshold - s.to_numpy(float)) / (ONOFF_WIDTH * level), -500, 500)
    p = pd.Series(1.0 / (1.0 + np.exp(-z)), index=s.index)
    if present is not None:
        p[~present.astype(bool).to_numpy()] = 1.0  # missing day = off
    return p


def off_mask(s: pd.Series, present: pd.Series | None = None) -> pd.Series:
    return p_off(s, present) >= ONOFF_CUT
