# Pulse detection

Spend-only detector for **channel pulses**: a channel switched off repeatedly, for pauses of
similar length. Entry point: `pulse.detect:detect` (layer C); `detect_b` and `detect_a` use
layers B and A (`python -m benchmark.eval.run_dev --detector pulse.detect:detect --load-data`).

## On/off layer

A channel's **usual level** is the median of its spend on days it spent anything. A day is
**off** when spend is at most 15 % of that level.

- **Layer A** (`onoff.p_off`): turns each day's spend into a probability of being off, using a
  smooth logistic curve that is exactly 0.5 at the 15 % cutoff.
- **Layer B** (`onoff.p_off_hmm`): a two-state (on/off) hidden Markov model that treats layer A's
  verdict as a noisy observation and smooths it with its neighbouring days, so a single odd day
  or a missing day does not flip the state.
- **Layer C** (`hsmm.p_off_hsmm`, default): a hidden semi-Markov model, like B but with explicit
  priors on how long on and off periods last (pauses typically about two weeks). This also
  absorbs 2–3 day spikes inside a pause.

An off period only counts if its median spend is still at most 15 % of the usual level.

## Pulse trains (`pulse.py`)

Off periods of 7+ days are grouped by similar length (each within 1.5× of the group's median) and
closeness in time; 2+ pauses form a pulse train. Pauses cut off by the start or end of the data
can extend a train but never start one. Each train gets a confidence score between 0 and 1 from
how clearly off its pauses are, how alike their lengths are, and how much of its window they
cover (a heuristic, not a calibrated probability).

## Scores

Dev split (45 scenarios), channel pulse only; layers A, B and C score the same:

| metric | value |
|---|---|
| channel pulse P / R / F1 | 1.000 / 1.000 / 1.000 (10 found / 0 false / 0 missed) |
| individual pauses | 27 / 27 |
| boundary error (median / p90) | 0 / 0 days |
| false alarms on no-event scenarios | 0.000 per country-year |

The same pulse code on `pulse-extended` scores 1.000 (13 / 0 / 0) on the sealed test split.
