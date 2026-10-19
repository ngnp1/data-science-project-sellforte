"""On/off model: two-state hidden semi-Markov model over the per-day 15 % verdicts."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pulse.onoff import p_off

HMM_EPS = 0.05      # P(a day's verdict is wrong): one odd day must not force a switch
OFF_MEDIAN = 14.0   # typical pause length in days
OFF_SIGMA = 0.4     # spread in log-length: ~92 % of off spells between 7 and 30 days


def _log_pmf_and_survival(pmf: np.ndarray):
    pmf = pmf / pmf.sum()
    surv = np.clip(pmf[::-1].cumsum()[::-1], 1e-300, None)        # P(length >= d)
    return np.log(np.r_[1e-300, np.clip(pmf, 1e-300, None)]), np.log(np.r_[1.0, surv])  # by length d


def p_off_hsmm(s: pd.Series, present: pd.Series | None = None) -> pd.Series:
    q = p_off(s).to_numpy()
    n = len(q)
    le = np.log(np.stack([HMM_EPS * q + (1 - HMM_EPS) * (1 - q),        # on
                          (1 - HMM_EPS) * q + HMM_EPS * (1 - q)]))      # off
    if present is not None:
        le[:, ~present.astype(bool).to_numpy()] = 0.0                  # missing: uninformative
    cum = np.concatenate([np.zeros((2, 1)), le.cumsum(axis=1)], axis=1)  # cum[k, t]: days 1..t
    d = np.arange(1, n + 1, dtype=float)
    lp, ls = zip(_log_pmf_and_survival(np.ones(n)),                                          # on
                 _log_pmf_and_survival(np.exp(-0.5 * (np.log(d / OFF_MEDIAN) / OFF_SIGMA) ** 2) / d))  # off

    # log P(spell (a, b] of state k); ponytail: O(n^2) spells, fine at benchmark size.
    def spell(a, b, k, fwd):
        start = np.where(a == 0, np.log(0.5), fwd[a, 1 - k])
        edge = (a == 0) | (b == n)
        return start + np.where(edge, ls[k][b - a], lp[k][b - a]) + cum[k, b] - cum[k, a]

    fwd = np.full((n + 1, 2), -np.inf)   # fwd[t, k]: days 1..t, a spell of k ends exactly at t (t < n)
    for t in range(1, n):
        a = np.arange(t)
        for k in range(2):
            fwd[t, k] = np.logaddexp.reduce(spell(a, t, k, fwd))
    z = np.logaddexp.reduce([np.logaddexp.reduce(spell(np.arange(n), n, k, fwd)) for k in range(2)])
    bwd = np.zeros((n + 1, 2))           # bwd[t, k]: days t+1..n, given a spell of k ended at t
    for t in range(n - 1, 0, -1):
        b, o = np.arange(t + 1, n + 1), None
        for k in range(2):
            o = 1 - k
            dur = np.where(b == n, ls[o][b - t], lp[o][b - t])
            bwd[t, k] = np.logaddexp.reduce(dur + cum[o, b] - cum[o, t] + np.where(b == n, 0.0, bwd[b, o]))
    # Day posterior: every spell (a, b] adds its probability to days a+1..b (difference array).
    post = np.zeros((2, n + 1))
    for b in range(1, n + 1):
        a = np.arange(b)
        for k in range(2):
            w = np.exp(spell(a, b, k, fwd) + (0.0 if b == n else bwd[b, k]) - z)
            post[k, a] += w
            post[k, b] -= w.sum()
    post = post.cumsum(axis=1)[:, :n]
    return pd.Series(np.clip(post[1] / post.sum(axis=0), 0.0, 1.0), index=s.index)
