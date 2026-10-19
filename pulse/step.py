# Step change: Bayesian online changepoint detection on log spend.
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from pulse.pulse import MIN_DAYS

LEVEL_DAYS = 100.0      # prior: a budget level typically lasts about a quarter
STEP_MIN_RATIO = 1.5    # smallest change called a new level: 1.5x up or down
PRIOR_SD = 2.0          # prior spread of a segment's mean log spend (very broad)


def _changepoints(x: np.ndarray, sigma: float) -> tuple[list[int], np.ndarray]:
    # BOCPD change indices and score[s] = P(run started within +-3 days of s), read LAG days later.
    n, h, v0 = len(x), 1.0 / LEVEL_DAYS, PRIOR_SD ** 2
    mu0, var = float(np.median(x)), sigma ** 2
    r = np.array([1.0])                       # P(run length = k)
    cnt, tot = np.zeros(1), np.zeros(1)       # per run length: points and their sum
    score, lag = np.zeros(n), 2 * MIN_DAYS
    for t in range(n):
        post_var = 1 / (1 / v0 + cnt / var)
        mean = post_var * (mu0 / v0 + tot / var)
        pred = np.exp(-0.5 * (x[t] - mean) ** 2 / (var + post_var)) / np.sqrt(var + post_var)
        grow = r * pred * (1 - h)
        r = np.r_[(r * pred * h).sum(), grow]
        r /= r.sum()
        cnt, tot = np.r_[0.0, cnt + 1], np.r_[0.0, tot + x[t]]
        if t >= lag:                        # run length lag +- 3 now = a run began near t - lag
            score[t - lag] = r[lag - 3:lag + 4].sum()
    peaks = [s for s in range(MIN_DAYS, n - MIN_DAYS) if score[s] > 0.5 and score[s] == score[s - MIN_DAYS:s + MIN_DAYS].max()]
    out = []
    for s in peaks:                            # best mean-shift split within +-MIN_DAYS
        k = np.arange(max(MIN_DAYS, s - MIN_DAYS), min(n - MIN_DAYS, s + MIN_DAYS) + 1)
        out.append(int(k[np.argmax([abs(x[max(0, j - 2 * MIN_DAYS):j].mean() - x[j:j + 2 * MIN_DAYS].mean()) for j in k])]))
    return sorted(set(out)), score


def _jumps(x: np.ndarray, a: int, lo: int, hi: int) -> bool:
    # True if the best-fitting transition width at a is <= MIN_DAYS (a jump, not a ramp).
    lo, hi = max(lo, a - 8 * MIN_DAYS), min(hi, a + 8 * MIN_DAYS)
    y, t = x[lo:hi], np.arange(lo, hi)
    sse = {}
    for w in (1, MIN_DAYS, 2 * MIN_DAYS, 4 * MIN_DAYS, 8 * MIN_DAYS):
        for start in range(a - w, a + 1):                  # any placement of the transition containing a
            f = np.clip((t - start) / w, 0, 1)             # 0 before, 1 after, linear in between
            A = np.stack([1 - f, f], axis=1)               # the two levels, fitted by least squares
            e = ((y - A @ np.linalg.lstsq(A, y, rcond=None)[0]) ** 2).sum()
            sse[w] = min(sse.get(w, np.inf), e)
    return min(sse, key=sse.get) <= MIN_DAYS


def find_steps(s: pd.Series, on: pd.Series) -> list[tuple]:
    # s: daily spend; on: days the channel is on. Returns [(start, end, ratio, confidence)].
    days = s.index[on.to_numpy(bool)]
    x = np.log(s.loc[days].to_numpy(float))
    if len(x) < 2 * MIN_DAYS:
        return []
    sigma = max(float(np.median(np.abs(np.diff(x) - np.median(np.diff(x))))) / 0.6745 / math.sqrt(2), 1e-3)
    cps, score = _changepoints(x, sigma)
    bounds = [0, *cps, len(x)]
    level = [x[a:b].mean() for a, b in zip(bounds, bounds[1:])]
    out, i, big = [], 1, math.log(STEP_MIN_RATIO)
    while i < len(level):
        a = bounds[i]
        # A return to the pre-excursion level is not a new level.
        back = i >= 2 and abs(level[i] - level[i - 2]) < big
        if abs(level[i] - level[i - 1]) < big or back or not _jumps(x, a, bounds[i - 1], bounds[i + 1]):
            i += 1
            continue
        j = i + 1                              # extend until spend is back near the old level
        while j < len(level) and abs(level[j] - level[i - 1]) >= big:
            j += 1
        b = bounds[j]
        conf = float(score[max(0, a - MIN_DAYS):a + MIN_DAYS + 1].max())
        out.append((days[a], days[b - 1], float(math.exp(x[a:b].mean() - level[i - 1])), conf))
        i = j + 1
    return out
