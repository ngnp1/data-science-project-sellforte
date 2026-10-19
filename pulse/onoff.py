"""Per-day P(off) from the 15 % cutoff: the evidence the on/off model reads."""
from __future__ import annotations

import numpy as np
import pandas as pd

RHO = 0.15          # off = spend at most 15% of the channel's usual level
EPS_ABS = 1e-6      # floor, so float noise around zero is never spend
ONOFF_WIDTH = 0.03  # softness of the cutoff, as a fraction of the usual level


def active_level(s: pd.Series) -> float:
    """Median spend over days the channel actually ran, so long pauses can't hide themselves."""
    positive = s[s > 0]
    return float(positive.median()) if not positive.empty else float("nan")


def p_off(s: pd.Series) -> pd.Series:
    """Logistic in spend, exactly 0.5 at the cutoff; 0 everywhere for a channel that never ran."""
    level = active_level(s)
    if not np.isfinite(level) or level <= 0:
        return pd.Series(0.0, index=s.index)
    threshold = max(EPS_ABS, RHO * level)
    z = np.clip((threshold - s.to_numpy(float)) / (ONOFF_WIDTH * level), -500, 500)
    return pd.Series(1.0 / (1.0 + np.exp(-z)), index=s.index)
