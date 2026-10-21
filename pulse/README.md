# Pulse detection

Spend-only detector for **channel pulses**: a channel switched off repeatedly, for pauses of
similar length. Entry point: `pulse.detect:detect`
(`python -m benchmark.eval.run_dev --detector pulse.detect:detect --load-data`).

## On/off model

A channel's **usual level** is the median of its spend on days it spent anything. A day is
**off** when spend is at most 15 % of that level.

1. Each day's spend is turned into a probability of being off (`onoff.p_off`): a smooth
   logistic curve that is exactly 0.5 at the 15 % cutoff.
2. A hidden semi-Markov model (`hsmm.p_off_hsmm`) reads those per-day verdicts as noisy
   evidence and decides each day together with its neighbours, with explicit priors on how long
   on and off periods last (pauses typically about two weeks). One-off spikes of up to 2–3 days
   inside a pause and missing days don't flip a channel on or off.

An off period only counts if its median spend is still at most 15 % of the usual level.

## Pulse trains (`pulse.py`)

Off periods of 7+ days are grouped by similar length (each within 1.5× of the group's median) and
closeness in time; 2+ pauses form a pulse train. Pauses cut off by the start or end of the data
can extend a train but never start one. Each train gets a confidence score between 0 and 1 from
how clearly off its pauses are, how alike their lengths are, and how much of its window they
cover (a heuristic, not a calibrated probability).

## Scores

Dev split (45 scenarios), channel pulse only:

| metric | value |
|---|---|
| channel pulse P / R / F1 | 1.000 / 1.000 / 1.000 (10 found / 0 false / 0 missed) |
| individual pauses | 27 / 27 |
| boundary error (median / p90) | 0 / 0 days |
| false alarms on no-event scenarios | 0.000 per country-year |

The same pulse code on `pulse-extended` scores 1.000 (13 / 0 / 0) on the sealed test split.
