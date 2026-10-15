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


# --- Layer B: a sticky two-state HMM over x = spend / usual level -----------
# Fixed domain priors, not fitted to any benchmark data.
HMM_STAY = 0.95       # P(state persists to the next day): on/off spells usually last a week+
OFF_SCALE = 0.05      # off-state spend ~ exponential with mean 5% of the usual level
EPS_OUT = 0.05        # outlier share in both states: uniform over [0, max x]
SIGMA_FLOOR = 0.1     # floor on the on-state spread, so a near-constant series isn't brittle


def p_off_hmm(s: pd.Series, present: pd.Series | None = None) -> pd.Series:
    """P(off) per day from forward-backward on a sticky on/off HMM.

    Missing days (present False) are uninformative: their neighbours decide.
    """
    level = active_level(s)
    if not np.isfinite(level) or level <= 0:
        return pd.Series(0.0, index=s.index)
    x = s.to_numpy(float) / level
    ok = np.isfinite(x) & (True if present is None else present.astype(bool).to_numpy())
    active = x[ok & (x > RHO)]
    sigma = max(SIGMA_FLOOR, 1.4826 * float(np.median(np.abs(active - np.median(active))))) if active.size else SIGMA_FLOOR
    u = EPS_OUT / max(1.0, float(x[ok].max())) if ok.any() else 0.0
    xs = np.where(ok, x, 0.0)
    # On: Gaussian around the usual level, flat above it (spending more is still on).
    on = (1 - EPS_OUT) * np.exp(-0.5 * ((np.minimum(xs, 1.0) - 1.0) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi)) + u
    off = (1 - EPS_OUT) * np.exp(-xs / OFF_SCALE) / OFF_SCALE + u
    e = np.where(ok[:, None], np.stack([on, off], axis=1), 1.0)
    T = np.array([[HMM_STAY, 1 - HMM_STAY], [1 - HMM_STAY, HMM_STAY]])
    n = len(x)
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
