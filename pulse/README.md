# Pulse detection

Finds **channel pulses**: a channel in one market that is switched off repeatedly,
for pauses of similar length, with spend back on in between. Spend-only; sales are ignored.

**Definition (agreed):** a day is *off* when spend is at most **15 % of the channel's usual
level**. A drop that stays above 15 % (say to 25 %) is a budget cut / step change, not a pause.

## Method

1. **Daily series.** For each market × channel, sum `media_investment` per day over
   the full date range. A day with no rows counts as spend 0.
2. **On/off layer** (`onoff.py`). Each day gets a probability of being *off*. Two layers
   exist; the rest of the method is identical for both. **Layer A** (`detect_a`):
   - usual level = median spend over days the channel actually ran;
   - cutoff = max(1e-6, 0.15 × usual level);
   - `P(off) = logistic((cutoff − spend) / (0.03 × usual level))`.
   At `P(off) = 0.5` this is exactly the hard rule "spend ≤ 15 % of usual = off";
   spend well above the cutoff is ≈ 0, true zeros are ≈ 1. A channel that never ran is never off.

   **Layer B** (default; `p_off_hmm`, `detect`): a sticky two-state (on/off) hidden Markov
   model, solved with forward-backward; `P(off)` is the posterior of the off state. B does **not**
   redefine off: each day's evidence is layer A's 15 % verdict, read as a noisy observation, and
   B decides each day together with its neighbours. Fixed priors, not fitted to benchmark data:

   | prior | value | why |
   |---|---|---|
   | P(stay in state) | 0.95 | on/off spells usually last a week or more |
   | start | 0.5 / 0.5 | no prior on the first day |
   | P(layer-A verdict wrong) | 0.05 | one odd day must not force a state change |
   | missing day | likelihood 1 for both states | no data: the neighbours decide |

   Spike check: one full-spend day inside a pause (layer A says *on*). Staying *off* costs one
   wrong verdict, 0.05 × 0.95² ≈ 0.045; turning *on* costs two switches, 0.95 × 0.05² ≈ 0.0024.
   Staying off is ≈ 19× more likely, so the spike is absorbed (posterior P(off) 0.94 in the
   stress case). Days inside a pause with spend above the 15 % cutoff are listed on the event as
   `spike_absorbed:<date>` tags.
3. **Off runs** (`pulse.py: off_runs`). Maximal stretches of days with `P(off) ≥ 0.5`.
   A run touching the first or last day is marked *censored*: we can't see its true start
   (or end), so its length is only a lower bound. Under layer B, days with no data are
   trimmed off a run's ends: a pause can't be shown to start or end on a day we can't see.
   **Definition check** (`detect.py: run_series`): a run is kept only if its median spend
   (over days with data) is at most 15 % of usual. Layer A satisfies this by construction; for
   B it guarantees smoothing can absorb odd days but never turn a budget cut into a pause.
   No stress case currently needs it (B doesn't absorb that much on its own); it is a guard.
4. **Grouping into trains** (`pulse.py: find_pulse_trains`).
   - keep runs of at least 7 days;
   - sort the uncensored runs by length; starting from the shortest, take the longest stretch
     in which **every** length is within 1.5 × the stretch's **median** (either way). The whole
     group is checked at once, so alternating 7/11-day pauses (median 9) stay together;
   - within a group, split wherever the gap between two pauses exceeds 6 × the shorter pause;
   - a chunk with at least 2 pauses is a pulse train;
   - a censored edge run joins a train if it is not longer than 1.5 × the train's median and
     sits within the gap limit of the train's first or last pause. It can extend a train but
     never create one;
   - the train's window runs from the first pause's start to the last pause's end; the pauses
     are kept as `components`.
5. **Pulse probability** (`pulse.py: pulse_probability`) = mean `P(off)` over the train's
   off-days × (1 − coefficient of variation of the pause lengths, floored at 0) × coverage,
   where coverage = the train's pause days ÷ all off-run days (runs ≥ 7 days) inside its window.
   Clean, equal-length pauses score near 1; partial or uneven pauses, or a train whose window
   contains an off period it didn't claim, score lower.
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
python -m benchmark.eval.run_dev --detector pulse.detect:detect --load-data     # layer B (default)
python -m benchmark.eval.run_dev --detector pulse.detect:detect_a --load-data   # layer A
```

On Windows set `PYTHONUTF8=1` first, or printing the report fails on the console encoding.

## Dev scores (45 scenarios, test split untouched)

This detector only emits `channel_pulse`, so read the pulse rows. The overall
F1 (0.235) only reflects that the other five event types are not attempted. Scores below are for the default (layer B); layer A is identical on dev except confidence 0.993.

| metric | value |
|---|---|
| channel_pulse precision / recall / F1 | 1.000 / 1.000 / 1.000 (10 TP, 0 FP, 0 FN) |
| individual pauses (components) P / R / F1 | 1.000 / 1.000 / 1.000 (27 of 27) |
| off-day coverage F1 | 1.000 |
| mean IoU / boundary error | 1.000 / 0 days |
| false positives on null scenarios | 0.000 per country-year |
| confidence of the 10 detections | mean 0.992, all correct |

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
The stress test below covers harder cases; still open:

- a single long pause is never a pulse (needs 2);
- pauses shorter than 7 days are never found;
- confidence on wrong detections: dev had none to check against.

### Layer A vs layer B on dev

| | layer A | layer B |
|---|---|---|
| channel_pulse P / R / F1 | 1.000 / 1.000 / 1.000 (10/0/0) | 1.000 / 1.000 / 1.000 (10/0/0) |
| pauses matched | 27 / 27 | 27 / 27 |
| boundary error (median / p90 days) | 0 / 0 | 0 / 0 |
| off-day coverage F1 | 1.000 | 1.000 |
| null FP per country-year | 0.000 | 0.000 |
| confidence (10 detections) | mean 0.993, all correct | mean 0.992, all correct |

Identical on dev: every dev pause is a clean zero, which both layers read the same way.

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

Each cell: pauses matched, max boundary error in days, pulse probability.

| case | layer A | layer B |
|---|---|---|
| clean: 3 x 14d, exact zero | 3/3, err 0, p 0.99 | 3/3, err 0, p 0.99 |
| partial: spend at 10% in pauses | 3/3, err 0, p 0.82 | 3/3, err 0, p 0.95 |
| near-zero noise 0-12% in pauses | 3/3, err 0, p 0.89 | 3/3, err 0, p 0.97 |
| uneven 7/11/9 | 3/3, err 0, p 0.81 | 3/3, err 0, p 0.81 |
| alternating 7/11/7/11 | 4/4, err 0, p 0.77 | 4/4, err 0, p 0.77 |
| irregular gaps 20/60/35 | 4/4, err 0, p 0.99 | 4/4, err 0, p 0.99 |
| 1-day spend spike inside each pause | 3/3, err 7, p 0.99 | 3/3, err 0, p 0.99 |
| train starts on day 0 | 3/3, err 0, p 0.99 | 3/3, err 0, p 0.99 |
| train runs to the last day | 3/3, err 0, p 0.99 | 3/3, err 0, p 0.99 |
| 3 missing days scattered in on-periods | 3/3, err 0, p 0.99 | 3/3, err 0, p 0.99 |
| 2 missing days inside a pause's flank | 3/3, err 2, p 0.93 | 3/3, err 0, p 0.99 |
| only 2 pauses | 2/2, err 0, p 0.99 | 2/2, err 0, p 0.99 |
| high noise (sigma 0.6) | 3/3, err 0, p 0.99 | 3/3, err 0, p 0.99 |
| very high noise (sigma 1.0) | 3/3, err 0, p 0.99 | 3/3, err 0, p 0.99 |
| long 30-day pause inside the train's window | 4/4, err 0, p 0.65 | 4/4, err 0, p 0.65 |
| pulse across a 3x budget step | 3/3, err 0, p 0.99 | 3/3, err 0, p 0.99 |
| short 5-day pauses (below MIN_DAYS) | 0/4, err -, p - | 0/4, err -, p - |
| NO PULSE: lone 40-day pause | 0 trains | 0 trains |
| NO PULSE: 2 pauses far apart (gap > 6x) | 0 trains | 0 trains |
| NO PULSE: random 1-5 day dropouts | 0 trains | 0 trains |
| NO PULSE: very high noise, no pauses | 0 trains | 0 trains |
| NO PULSE: spend drops to 25% for 3 x 14d | 0 trains | 0 trains |
| NO PULSE: spend drops to 30% for 3 x 14d | 0 trains | 0 trains |
| NO PULSE: spend drops to 40% for 3 x 14d | 0 trains | 0 trains |
| NO PULSE: every other day off, 3 x 14d | 0 trains | 0 trains |

Fixed after the first run (grouping and probability only; the on/off layer is unchanged):

| case | before | after | root cause |
|---|---|---|---|
| alternating 7/11/7/11 | 2/4 pauses, probability 0.99 | 4/4, 0.77 | grouping checked each new pause against the median of the group *so far* ([7, 7] → 7, so 11 > 10.5 was rejected) instead of the whole group (median 9) |
| train starts on day 0 | 2/3, 0.99 | 3/3, 0.99 | runs touching the data edge were dropped outright |
| train runs to the last day | 2/3, 0.99 | 3/3, 0.99 | same |
| over-confidence | 0.99 regardless of what was left out | coverage term; the new "long 30-day pause inside the window" case scores 0.65 | the probability only looked at the pauses it kept |

Dev scores are unchanged after the fix (pulses 10/0/0, 27/27 pauses, 0 null false
positives, same confidences), and the four no-pulse controls stay silent.

Layer A vs layer B:
- **B fixes** the spike inside a pause (7 → 0 days error; the spike day is tagged
  `spike_absorbed`) and missing days next to a pause (2 → 0).
- **Both** respect the 15 % definition: drops to 25 %, 30 % and 40 % of usual are reported as no
  pulse by either layer (these are step changes by definition), and 10 % is still a pulse.
- **Both** miss 5-day pauses (below `MIN_DAYS`, by design).
- B has **no new misses**. An earlier version of B modelled spend itself instead of layer A's
  verdict; it caught the 25 % drop (wrong by definition) and stretched a pause by 2 days under
  very heavy noise. Building B on layer A's verdicts fixed both.
- Confidence: B scores clearly-low partial pauses higher (10 %: 0.95 vs 0.82), because a run of
  consistent days is stronger evidence together than one day at a time.

What holds for both: irregular gaps, a 3x budget step under the pulse, scattered missing days,
edge pauses, alternating lengths, heavy noise, and every no-pulse control stays silent.

**Default: layer B** (`pulse.detect:detect`). It passes every dev gate and fixes the spike and
missing-day cases without new misses. Layer A stays available as `pulse.detect:detect_a`.

## Next steps

1. Layer C: a hidden semi-Markov model, which models pause and on-period durations directly.
