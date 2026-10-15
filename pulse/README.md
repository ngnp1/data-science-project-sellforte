# Pulse detection

Finds **channel pulses**: a channel in one market that is switched off repeatedly,
for pauses of similar length, with spend back on in between. Spend-only; sales are ignored.

## Method

1. **Daily series.** For each market × channel, sum `media_investment` per day over
   the full date range. A day with no rows counts as spend 0.
2. **On/off layer** (`onoff.py`). Each day gets a probability of being *off*:
   - usual level = median spend over days the channel actually ran;
   - cutoff = max(1e-6, 0.15 × usual level);
   - `P(off) = logistic((cutoff − spend) / (0.03 × usual level))`.
   At `P(off) = 0.5` this is exactly the hard rule "spend ≤ 15 % of usual = off";
   spend well above the cutoff is ≈ 0, true zeros are ≈ 1. A channel that never ran is never off.
3. **Off runs** (`pulse.py: off_runs`). Maximal stretches of days with `P(off) ≥ 0.5`.
   Runs touching the first or last day are dropped (we can't see where they start or end).
4. **Grouping into trains** (`pulse.py: find_pulse_trains`).
   - keep runs of at least 7 days;
   - sort by length; a run joins the current group if it is at most 1.5 × the group's
     **median** length, otherwise it starts a new group (the median stops one short pause
     from capping the rest of the train);
   - within a group, split wherever the gap between two pauses exceeds 6 × the shorter pause;
   - a chunk with at least 2 pauses is a pulse train. Its window runs from the first pause's
     start to the last pause's end; the pauses are kept as `components`.
5. **Pulse probability** (`pulse.py: pulse_probability`) = mean `P(off)` over the train's
   off-days × (1 − coefficient of variation of the pause lengths, floored at 0).
   Clean, equal-length pauses score near 1; partial or uneven pauses score lower.
   It is exported as the event's `detection_confidence`.

| constant | value | file |
|---|---|---|
| `RHO` | 0.15 | onoff.py |
| `ONOFF_WIDTH` | 0.03 | onoff.py |
| `ONOFF_CUT` | 0.5 | onoff.py |
| `MIN_DAYS` | 7 | pulse.py |
| `MIN_RUNS` | 2 | pulse.py |
| `LEN_RATIO` | 0.5 | pulse.py |
| `MAX_GAP_MULT` | 6 | pulse.py |

## Run it

```bash
python -m pytest tests/pulse -q
python -m benchmark.eval.run_dev --detector pulse.detect:detect --load-data
```

On Windows set `PYTHONUTF8=1` first, or printing the report fails on the console encoding.

## Dev scores (45 scenarios, test split untouched)

This detector only emits `channel_pulse`, so read the pulse rows. The overall
F1 (0.235) only reflects that the other five event types are not attempted.

| metric | value |
|---|---|
| channel_pulse precision / recall / F1 | 1.000 / 1.000 / 1.000 (10 TP, 0 FP, 0 FN) |
| individual pauses (components) P / R / F1 | 1.000 / 1.000 / 1.000 (27 of 27) |
| off-day coverage F1 | 1.000 |
| mean IoU / boundary error | 1.000 / 0 days |
| false positives on null scenarios | 0.000 per country-year |
| confidence of the 10 detections | mean 0.993, all correct |

### Against the detection-library baseline (same 45 dev scenarios)

| | detection-library (all types) | this (pulse only) |
|---|---|---|
| channel_pulse P / R / F1 | 1.000 / 1.000 / 1.000 (10/0/0) | 1.000 / 1.000 / 1.000 (10/0/0) |
| overall P / R / F1 | 0.868 / 0.880 / 0.874 | 1.000 / 0.133 / 0.235 |
| null FP per country-year | 0.038 | 0.000 |

Overall recall is low by design: dark period, holdout, single channel, step change and
staggered launch are not detected here.

### Caveats

The 10 dev pulses are the easy case: every pause is an exact zero, pauses within a
train have equal lengths (10, 14 or 21 days), spacing is regular, 2–4 pauses per train.
So the scores above do not yet show behaviour on:

- uneven pauses: 7 and 11 days still split (11 > 1.5 × 7);
- partial pauses (spend dropped but above 15 % of usual): treated as on;
- trains touching the start or end of the data: those pauses are dropped;
- a single long pause: never a pulse (needs 2);
- confidence on wrong detections: there were none to check against.

## Next steps

1. Stress test on harder pulses: synthetic series with partial pauses, uneven lengths,
   trains at the data edges and lone long pauses, scored for detection and confidence.
2. Layer B: a sticky two-state HMM for P(off), so a day's state also depends on its neighbours.
3. Layer C: a hidden semi-Markov model, which models pause and on-period durations directly.
