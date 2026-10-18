# Informative-period detection

Detects the six informative-period types in daily media spend (sales are not used).
A day is **off** when spend is at most **15 % of the channel's usual level** (median of its
active days). Run it with:

```bash
python -m benchmark.eval.run_dev --detector pulse.detect:detect --load-data
```

## Shared base: on/off layer C (`hsmm.py`)

Each market × channel gets a per-day P(off). Each day's evidence is the 15 % rule; a two-state
hidden semi-Markov model then reads it with explicit spell lengths, so one-off spikes and
missing days don't flip a channel on or off. Off spells must also have median spend ≤ 15 %.
Earlier layers A (per-day cutoff) and B (sticky HMM) are still available as `detect_a`/`detect_b`.

## Method per period

| period | method |
|---|---|
| **Channel pulse** (`pulse.py`) | A channel's off spells of 7+ days, grouped by similar length (each within 1.5× of the group median) and closeness (gap ≤ 6× the pause); 2+ pauses form a train. Pauses cut off by the data edge can extend a train, never start one. |
| **Natural holdout** (`holdout.py`) | One channel off for 7+ days while more than half of the market's other channels stay on. Excludes pulse pauses and spells starting on day 1 (launch). |
| **Single channel** (`holdout.py`) | 7+ days where every channel but one is off. Needs 3+ channels; with 2 it is the same pattern as a holdout and is reported as one. |
| **Dark period** (`holdout.py`) | 7+ days where every channel in the market is off (a channel that never spends counts as off). |
| **Staggered launch** (`holdout.py`) | A channel is off from day 1 for 7+ days in one market, while the same channel ran at least 7 days earlier in another market. One event per late market. |
| **Step change** (`step.py`) | Bayesian online changepoint detection on log spend over "on" days. A step is a jump of at least 1.5× up or down that is abrupt (completes within a week, so ramps don't count), not a return to an earlier level, and lasts until spend returns or the data ends. |

Confidence on every event is a heuristic score in [0, 1] (mainly the mean P(off), or the
changepoint posterior for steps), not a calibrated probability.

## Scores

Precision / recall / F1 (found / false / missed).

| | dev (45 scenarios) | test (55 scenarios) | teammate, test |
|---|---|---|---|
| **Overall** | **0.945 / 0.920 / 0.932** (69 / 4 / 6) | **0.989 / 0.939 / 0.963** (92 / 1 / 6) | 0.978 / 0.918 / 0.947 (90 / 2 / 8) |
| channel pulse | 1.000 (10 / 0 / 0) | 1.000 (13 / 0 / 0) | 1.000 |
| dark period | 1.000 (12 / 0 / 0) | 1.000 (15 / 0 / 0) | 1.000 |
| natural holdout | 0.850 (17 / 3 / 3) | 0.939 (23 / 0 / 3) | 0.898 |
| single channel | 0.824 (7 / 0 / 3) | 1.000 (12 / 0 / 0) | 1.000 |
| staggered launch | 0.952 (10 / 1 / 0) | 0.970 (16 / 1 / 0) | 0.970 |
| step change | 1.000 (13 / 0 / 0) | 0.897 (13 / 0 / 3) | 0.857 |
| false alarms on no-event scenarios | 0.000 | 0.000 | 0.000 |

Boundaries are exact (median and p90 error 0 days) on both splits.

The test score is from `pulse-extended` @ 0528a12. It is the project's **second** final run on the
sealed test split; the team's first is commit 34cd15e on `detection-library`.
