# Informative-period detection

Spend-only detector for the six period types. Entry point: `team.pulse.detect:detect`
(`python -m benchmark.eval.run_dev --detector team.pulse.detect:detect --load-data`).

## On/off model (shared by every detector)

A channel's **usual level** is the median of its spend on days it spent anything. A day is
**off** when spend is at most 15 % of that level.

1. Each day's spend is turned into a probability of being off (`onoff.p_off`): a smooth
   logistic curve that is exactly 0.5 at the 15 % cutoff.
2. A hidden semi-Markov model (`hsmm.p_off_hsmm`) reads those per-day verdicts as noisy
   evidence and decides each day together with its neighbours, with explicit priors on how long
   on and off periods last (pauses typically about two weeks for pulses, about six weeks for the
   other types). One-off spikes of up to 2–3 days inside a pause and missing days don't flip a
   channel on or off.

An off period only counts if its median spend is still at most 15 % of the usual level.

## Method per period

| period | method |
|---|---|
| **Channel pulse** (`pulse.py`) | A channel's off periods of 7+ days, grouped by similar length and closeness in time; 2+ pauses form a pulse train. |
| **Natural holdout** (`holdout.py`) | One channel off for 7+ days while most of the market's other channels stay on. |
| **Single channel** (`holdout.py`) | 7+ days where every channel in the market is off except one (markets with 3+ channels). |
| **Dark period** (`holdout.py`) | 7+ days where every channel in the market is off. |
| **Staggered launch** (`holdout.py`) | A channel that starts at least a week later in one market than in another. |
| **Step change** (`step.py`) | Bayesian online changepoint detection on log spend finds where the spend level shifts; a step is an abrupt shift of at least 1.5× up or down that is not just a return to an earlier level. |

Each event carries a confidence score between 0 and 1 (a heuristic, not a calibrated probability).

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

The test score is from `pulse-extended` @ 0528a12, the project's second final run on the sealed
test split (the first is 34cd15e on `detection-library`).
