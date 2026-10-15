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


# --- Layer B: a sticky two-state HMM over layer A's verdicts ---------------
# Layer B does not redefine "off": its evidence for each day is layer A's P(off)
# (spend <= RHO x usual), read as a noisy observation. It only decides each day
# together with its neighbours. Fixed priors, not fitted to any benchmark data.
HMM_STAY = 0.95       # P(state persists to the next day): on/off spells usually last a week+
HMM_EPS = 0.05        # P(a day's layer-A verdict is wrong): one odd day must not force a switch


def p_off_hmm(s: pd.Series, present: pd.Series | None = None) -> pd.Series:
    """P(off) per day from forward-backward on a sticky on/off HMM.

    Missing days (present False) are uninformative: their neighbours decide.
    """
    q = p_off(s).to_numpy()                       # layer A, without the missing-day rule
    off = (1 - HMM_EPS) * q + HMM_EPS * (1 - q)   # P(day's evidence | off)
    on = HMM_EPS * q + (1 - HMM_EPS) * (1 - q)    # P(day's evidence | on)
    e = np.stack([on, off], axis=1)
    if present is not None:
        e[~present.astype(bool).to_numpy()] = 1.0
    T = np.array([[HMM_STAY, 1 - HMM_STAY], [1 - HMM_STAY, HMM_STAY]])
    n = len(q)
    a, b = np.empty((n, 2)), np.ones((n, 2))
    a[0] = 0.5 * e[0]
    a[0] /= a[0].sum()
    for t in range(1, n):              # scaled forward pass
        a[t] = (a[t - 1] @ T) * e[t]
        a[t] /= a[t].sum()
    for t in range(n - 2, -1, -1):     # scaled backward pass
        b[t] = T @ (e[t + 1] * b[t + 1])
        b[t] /= b[t].sum()
    post = a * b
    return pd.Series(post[:, 1] / post.sum(axis=1), index=s.index)
