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
python -m pulse.stress
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

## Stress test (harder pulse cases)

The dev pulses are easy, so `stress.py` builds synthetic series where we know the answer
and each case adds one difficulty. Detector code and constants are unchanged; the point is
to see what breaks.

How the cases are made: 365 days of spend around 100/day with lognormal day-to-day noise
(sigma 0.2 unless stated), fixed seed. A pulse is 3 pauses of 14 days with 28 on-days between
them unless stated. A pause is spend 0 unless stated. "Pauses matched" means a found pause
overlaps a true pause with IoU >= 0.5; boundary error is the worst start/end miss in days.

```bash
python -m pulse.stress              # prints the table below
python -m pytest tests/pulse -q     # pins it: known failures are xfail(strict)
```

| case | true pauses | trains found | pauses matched | max boundary error (days) | pulse probability |
|---|---|---|---|---|---|
| clean: 3 x 14d, exact zero | 3 | 1 | 3/3 | 0 | 0.99 |
| partial: spend at 10% in pauses | 3 | 1 | 3/3 | 0 | 0.82 |
| partial: spend at 25% in pauses | 3 | 0 | 0/3 | - | - |
| near-zero noise 0-12% in pauses | 3 | 1 | 3/3 | 0 | 0.89 |
| uneven 7/11/9 | 3 | 1 | 3/3 | 0 | 0.81 |
| alternating 7/11/7/11 | 4 | 1 | 2/4 | 0 | 0.99 |
| irregular gaps 20/60/35 | 4 | 1 | 4/4 | 0 | 0.99 |
| 1-day spend spike inside each pause | 3 | 1 | 3/3 | 7 | 0.99 |
| train starts on day 0 | 3 | 1 | 2/3 | 0 | 0.99 |
| train runs to the last day | 3 | 1 | 2/3 | 0 | 0.99 |
| 3 missing days scattered in on-periods | 3 | 1 | 3/3 | 0 | 0.99 |
| 2 missing days inside a pause's flank | 3 | 1 | 3/3 | 2 | 0.93 |
| only 2 pauses | 2 | 1 | 2/2 | 0 | 0.99 |
| high noise (sigma 0.6) | 3 | 1 | 3/3 | 0 | 0.99 |
| very high noise (sigma 1.0) | 3 | 1 | 3/3 | 0 | 0.99 |
| pulse across a 3x budget step | 3 | 1 | 3/3 | 0 | 0.99 |
| short 5-day pauses (below MIN_DAYS) | 4 | 0 | 0/4 | - | - |
| NO PULSE: lone 40-day pause | 0 | 0 | - | - | - |
| NO PULSE: 2 pauses far apart (gap > 6x) | 0 | 0 | - | - | - |
| NO PULSE: random 1-5 day dropouts | 0 | 0 | - | - | - |
| NO PULSE: very high noise, no pauses | 0 | 0 | - | - | - |

What breaks:
- **Partial pauses above the cutoff**: spend at 25% of usual is "on", so the pulse is missed.
  At 10% it is found, and the probability drops to 0.82, as it should.
- **One spike inside a pause** splits it into a 7-day and a 6-day run; only the 7-day half
  survives, so each pause is found with a 7-day boundary error.
- **Pauses touching the data edge** are dropped, so 2 of 3 pauses are found.
- **Alternating 7/11-day pauses** end up in different length groups: only one pair forms a train.
- **Missing days next to a pause** count as off and extend it (2-day error).
- **5-day pauses** are below `MIN_DAYS` by design.
- **The probability is over-confident** when pauses are lost: 0.99 for a train that found only
  2 of 4 pauses, or that lost one to the data edge, because it only looks at the pauses it kept.

What holds: day-to-day noise up to sigma 1.0, irregular gaps, a 3x budget step under the
pulse, scattered missing days, and all four no-pulse controls (lone long pause, two far-apart
pauses, random 1-5 day dropouts, pure noise) produce no false pulse.

For layer B: the spike, missing-day and partial-pause failures all come from deciding each
day on its own. A sticky HMM decides each day's state together with its neighbours, so one
on-day inside a pause or a missing day at its edge would not flip the state. That also gives a
probability that reflects uncertain days. The edge, alternating-length and over-confidence
issues are grouping rules, not the on/off layer, and need their own fixes.

## Next steps

1. Layer B: a sticky two-state HMM for P(off), so a day's state also depends on its neighbours.
2. Layer C: a hidden semi-Markov model, which models pause and on-period durations directly.
