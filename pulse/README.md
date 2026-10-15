# Pulse detection

Finds **channel pulses**: a channel in one market that is switched off repeatedly,
for pauses of similar length, with spend back on in between. Spend-only; sales are ignored.

**Definition (agreed):** a day is *off* when spend is at most **15 % of the channel's usual
level**. A drop that stays above 15 % (say to 25 %) is a budget cut / step change, not a pause.

## Method

1. **Daily series.** For each market × channel, sum `media_investment` per day over
   the full date range. A day with no rows counts as spend 0.
2. **On/off layer** (`onoff.py`). Each day gets a probability of being *off*. Three layers
   exist; the rest of the method is identical for both. **Layer A** (`detect_a`):
   - usual level = median spend over days the channel actually ran;
   - cutoff = max(1e-6, 0.15 × usual level);
   - `P(off) = logistic((cutoff − spend) / (0.03 × usual level))`.
   At `P(off) = 0.5` this is exactly the hard rule "spend ≤ 15 % of usual = off";
   spend well above the cutoff is ≈ 0, true zeros are ≈ 1. A channel that never ran is never off.

   **Layer B** (`p_off_hmm`, `detect_b`): a sticky two-state (on/off) hidden Markov
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

   **Layer C** (default; `hsmm.py: p_off_hsmm`, `detect`): a two-state hidden *semi*-Markov
   model. Same evidence as B (layer A's 15 % verdict, wrong 5 % of the time; missing days
   uninformative), so C cannot redefine off either. The difference: instead of B's daily
   P(stay) = 0.95, which makes every spell length geometric (1-day spells most likely), C gives
   whole spells explicit length priors and is solved with explicit-duration forward-backward:

   | prior | value | why |
   |---|---|---|
   | off-spell length | lognormal, median 14 days, log-spread 0.4 (≈ 92 % between 7 and 30 days) | pauses are typically one to four weeks; 1-3 day off spells are unlikely |
   | on-spell length | flat over every length | no view on how long a channel runs |
   | first / last spell | survival P(length ≥ d) instead of P(length = d) | the data edges cut them off (same idea as censored runs) |
   | start | 0.5 / 0.5 | no prior on the first day |

   Effect: a 2- or 3-day spend spike inside a pause would turn one 14-day pause into two short
   off spells, which the off-length prior makes unlikely, so C absorbs it; B only absorbs
   1-day spikes (at 2 days, two switches already cost about as much as two wrong verdicts).
   A short 1–4 day outage is likewise read as *on*. Exact: every spell length up to the series
   length, no cap. Runtime: about 0.1 s per 2-year series; the full dev run (1,111 series) takes
   132 s, against 8 s for layer B.
3. **Off runs** (`pulse.py: off_runs`). Maximal stretches of days with `P(off) ≥ 0.5`.
   A run touching the first or last day is marked *censored*: we can't see its true start
   (or end), so its length is only a lower bound. Under layers B and C, days with no data are
   trimmed off a run's ends: a pause can't be shown to start or end on a day we can't see.
   **Definition check** (`detect.py: run_series`): a run is kept only if its median spend
   (over days with data) is at most 15 % of usual. Layer A satisfies this by construction; for
   B and C it guarantees smoothing can absorb odd days but never turn a budget cut into a pause.
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
python -m benchmark.eval.run_dev --detector pulse.detect:detect --load-data     # layer C (default)
python -m benchmark.eval.run_dev --detector pulse.detect:detect_b --load-data   # layer B
python -m benchmark.eval.run_dev --detector pulse.detect:detect_a --load-data   # layer A
```

On Windows set `PYTHONUTF8=1` first, or printing the report fails on the console encoding.

## Dev scores (45 scenarios, test split untouched)

Only pulse metrics are reported. This detector emits `channel_pulse` events only, so the
evaluator's overall precision/recall/F1 (which counts all six event types) says nothing about
it and is left out. Scores are for the default (layer C); layers A and B are identical on dev
(A's mean confidence is 0.993).

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
| false pulses | 0 | 0 |

### How confidence is used

Every detected pulse is **reported, with a confidence score** (`detection_confidence` = the
pulse probability). How the evaluator (`benchmark/eval`, unchanged) uses it:

- **Precision/recall/F1 ignore confidence.** Every emitted event is matched and counted,
  i.e. treated as authoritative: `runner.py:59` matches all predictions, and
  `metrics.py:32-37` (`event_level`) and `metrics.py:39-45` (`per_type`) compute P/R/F1 from
  those matches. There is no confidence threshold in the headline numbers.
- **Two things are computed from confidence** (`runner.py:80-81`):
  - *reliability curve* (`metrics.py:140-173`): confidence in 10 bins vs the share that matched
    truth. Layer C on dev (same for B): one bin, 0.9–1.0, n = 10, mean confidence 0.992, empirical precision 1.000;
  - *operating curve* (`metrics.py:176-202`): P/R after dropping detections below a cut,
    0.00 to 1.00 in steps of 0.05. Layer C's 10 dev pulses range 0.988–0.994, so all 10 are kept
    (pulse P = R = 1.000) up to cut 0.95 and none at 1.00.
- Nothing else: no Brier score, AUC or calibration fit. Matching (`matching.py`) never reads
  confidence.

The confidence is a **heuristic score in [0, 1], not a calibrated probability**: it combines
P(off), pause-length regularity and coverage, and nothing has shown that 0.8 means "right 80 %
of the time". Dev has no false or low-confidence pulses, so calibration can't be judged yet;
treat the score as a ranking (higher = cleaner pulse), not as odds.

### Caveats

The 10 dev pulses are the easy case: every pause is an exact zero, pauses within a
train have equal lengths (10, 14 or 21 days), spacing is regular, 2–4 pauses per train.
The stress test below covers harder cases; still open:

- a single long pause is never a pulse (needs 2);
- pauses shorter than 7 days are never found;
- confidence on wrong detections: dev had none to check against.

### Layers A, B and C on dev

| | layer A | layer B | layer C |
|---|---|---|---|
| channel_pulse P / R / F1 | 1.000 / 1.000 / 1.000 (10/0/0) | same | same |
| pauses matched | 27 / 27 | 27 / 27 | 27 / 27 |
| boundary error (median / p90 days) | 0 / 0 | 0 / 0 | 0 / 0 |
| off-day coverage F1 | 1.000 | 1.000 | 1.000 |
| null FP per country-year | 0.000 | 0.000 | 0.000 |
| confidence (10 detections) | mean 0.993, all correct | mean 0.992, all correct | mean 0.992, all correct |
| full dev run time | not timed (≈ B) | 8 s | 132 s |

Identical on dev: every dev pause is a clean zero, which all three layers read the same way.

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

| case | layer A | layer B | layer C |
|---|---|---|---|
| clean: 3 x 14d, exact zero | 3/3, err 0, p 0.99 | 3/3, err 0, p 0.99 | 3/3, err 0, p 0.99 |
| partial: spend at 10% in pauses | 3/3, err 0, p 0.82 | 3/3, err 0, p 0.95 | 3/3, err 0, p 0.95 |
| near-zero noise 0-12% in pauses | 3/3, err 0, p 0.89 | 3/3, err 0, p 0.97 | 3/3, err 0, p 0.97 |
| uneven 7/11/9 | 3/3, err 0, p 0.81 | 3/3, err 0, p 0.81 | 3/3, err 0, p 0.81 |
| alternating 7/11/7/11 | 4/4, err 0, p 0.77 | 4/4, err 0, p 0.77 | 4/4, err 0, p 0.77 |
| irregular gaps 20/60/35 | 4/4, err 0, p 0.99 | 4/4, err 0, p 0.99 | 4/4, err 0, p 0.99 |
| 1-day spend spike inside each pause | 3/3, err 7, p 0.99 | 3/3, err 0, p 0.99 | 3/3, err 0, p 0.99 |
| 2-day spend spike inside each pause | 3/3, err 7, p 0.99 | 3/3, err 7, p 0.99 | 3/3, err 0, p 0.99 |
| 3-day spend spike inside each pause | 3/3, err 7, p 0.99 | 3/3, err 7, p 0.98 | 3/3, err 0, p 0.97 |
| long pauses: 3 x 35d | 3/3, err 0, p 0.99 | 3/3, err 0, p 1.00 | 3/3, err 0, p 1.00 |
| long pauses: 3 x 42d | 3/3, err 0, p 0.99 | 3/3, err 0, p 1.00 | 3/3, err 0, p 1.00 |
| long pauses: 3 x 56d | 3/3, err 0, p 0.99 | 3/3, err 0, p 1.00 | 3/3, err 0, p 1.00 |
| 1-day spike inside 3 x 42d pauses | 3/3, err 21, p 0.97 | 3/3, err 0, p 1.00 | 3/3, err 0, p 0.99 |
| slightly uneven 13/15/14/16 | 4/4, err 0, p 0.92 | 4/4, err 0, p 0.92 | 4/4, err 0, p 0.91 |
| train starts on day 0 | 3/3, err 0, p 0.99 | 3/3, err 0, p 0.99 | 3/3, err 0, p 0.99 |
| train runs to the last day | 3/3, err 0, p 0.99 | 3/3, err 0, p 0.99 | 3/3, err 0, p 0.99 |
| 3 missing days scattered in on-periods | 3/3, err 0, p 0.99 | 3/3, err 0, p 0.99 | 3/3, err 0, p 0.99 |
| 2 missing days inside a pause's flank | 3/3, err 2, p 0.93 | 3/3, err 0, p 0.99 | 3/3, err 0, p 0.99 |
| only 2 pauses | 2/2, err 0, p 0.99 | 2/2, err 0, p 0.99 | 2/2, err 0, p 0.99 |
| high noise (sigma 0.6) | 3/3, err 0, p 0.99 | 3/3, err 0, p 0.99 | 3/3, err 0, p 0.99 |
| very high noise (sigma 1.0) | 3/3, err 0, p 0.99 | 3/3, err 0, p 0.99 | 3/3, err 0, p 0.99 |
| long 30-day pause inside the train's window | 4/4, err 0, p 0.65 | 4/4, err 0, p 0.65 | 4/4, err 0, p 0.65 |
| pulse across a 3x budget step | 3/3, err 0, p 0.99 | 3/3, err 0, p 0.99 | 3/3, err 0, p 0.99 |
| short 5-day pauses (below MIN_DAYS) | 0/4, err -, p - | 0/4, err -, p - | 0/4, err -, p - |
| NO PULSE: lone 40-day pause | 0 trains | 0 trains | 0 trains |
| NO PULSE: 2 pauses far apart (gap > 6x) | 0 trains | 0 trains | 0 trains |
| NO PULSE: random 1-5 day dropouts | 0 trains | 0 trains | 0 trains |
| NO PULSE: very high noise, no pauses | 0 trains | 0 trains | 0 trains |
| NO PULSE: short 1-4 day outages every ~3 weeks | 0 trains | 0 trains | 0 trains |
| NO PULSE: spend drops to 25% for 3 x 14d | 0 trains | 0 trains | 0 trains |
| NO PULSE: spend drops to 30% for 3 x 14d | 0 trains | 0 trains | 0 trains |
| NO PULSE: spend drops to 40% for 3 x 14d | 0 trains | 0 trains | 0 trains |
| NO PULSE: every other day off, 3 x 14d | 0 trains | 0 trains | 0 trains |

Fixed after the first run (grouping and probability only; the on/off layer is unchanged):

| case | before | after | root cause |
|---|---|---|---|
| alternating 7/11/7/11 | 2/4 pauses, probability 0.99 | 4/4, 0.77 | grouping checked each new pause against the median of the group *so far* ([7, 7] → 7, so 11 > 10.5 was rejected) instead of the whole group (median 9) |
| train starts on day 0 | 2/3, 0.99 | 3/3, 0.99 | runs touching the data edge were dropped outright |
| train runs to the last day | 2/3, 0.99 | 3/3, 0.99 | same |
| over-confidence | 0.99 regardless of what was left out | coverage term; the new "long 30-day pause inside the window" case scores 0.65 | the probability only looked at the pauses it kept |

Dev scores are unchanged after the fix (pulses 10/0/0, 27/27 pauses, 0 null false
positives, same confidences), and the four no-pulse controls stay silent.

Layers A, B and C:
- **B fixes** a 1-day spike inside a pause (7 → 0 days error; the spike day is tagged
  `spike_absorbed`) and missing days next to a pause (2 → 0).
- **C also fixes 2- and 3-day spikes** (7 → 0), which B does not: for B, two switches cost about
  as much as two wrong verdicts. C's off-length prior makes splitting a pause into short pieces
  unlikely.
- **All three** respect the 15 % definition: drops to 25 %, 30 % and 40 % of usual are no pulse
  (step changes by definition), and 10 % is still a pulse.
- **Long pauses** (3 × 35, 42 and 56 days), outside C's 7–30 day prior: all three layers find
  every pause exactly; C's prior does not split or miss them. With a 1-day spike inside 42-day
  pauses, A is 21 days off while B and C are exact.
- **All three** miss 5-day pauses (below `MIN_DAYS`, by design) and stay silent on short 1–4 day
  outages and every other no-pulse control.
- **No new misses** for B or C. An earlier version of B modelled spend itself instead of layer
  A's verdict; it caught the 25 % drop (wrong by definition) and stretched a pause by 2 days under
  very heavy noise. Building B and C on layer A's verdicts fixed both.
- Confidence: B and C score clearly-low partial pauses higher than A (10 %: 0.95 vs 0.82),
  because a run of consistent days is stronger evidence together than one day at a time.

**Default: layer C** (`pulse.detect:detect`). It passes every dev gate, is strictly better than
B on the stress test (2- and 3-day spikes) and has no new misses. The cost is runtime: 132 s for
the full dev run instead of 8 s. Layers B and A stay available as `pulse.detect:detect_b` and
`pulse.detect:detect_a`.

## Natural holdout

**Definition** (`benchmark/BENCHMARK.md:41`): one channel stops while the market's other channels
keep running. "Stops" uses the same agreed 15 % definition as pulses.

**Method** (`holdout.py`, built on layer C):
1. P(off) per day for every active channel in the market, from layer C with a broader off-spell
   prior than pulses: holdout tests typically run from a couple of weeks to a few months, so
   lognormal, median **42 days**, log-spread **0.8** (≈ 2.5 to 20 weeks). Fixed, not fitted to dev.
2. Off spells from it, checked against the 15 % definition (median spend ≤ 15 % of usual).
3. Keep a spell if it is at least 7 days, does not start on the first day of the data (from
   spend alone that looks like a late launch), is not one of the channel's pulse pauses, and
   **more than half of the market's other active channels stay on** through it (mean P(off) < 0.5).
4. Confidence = mean P(off) of the channel over the spell × mean "on-ness" (1 − mean P(off)) of
   the other channels. Heuristic, in [0, 1], same caveats as the pulse confidence.

`pulse.detect:detect` now emits both pulse and holdout events (per market, so holdouts can see
the other channels). Full dev run: 170 s.

**Dev scores** (45 scenarios, test split untouched):

| | teammate detection-library | this (layer C) |
|---|---|---|
| natural_holdout P / R / F1 | 0.842 / 0.800 / 0.821 (16 TP / 3 FP / 4 FN) | **0.850 / 0.850 / 0.850 (17 / 3 / 3)** |
| boundary error (median / p90 days) | – | 0 / 0 |
| channel_pulse | 10 / 0 / 0 | 10 / 0 / 0 (unchanged) |
| null FP per country-year | – | 0.000 |

Confidence: all 20 holdout detections score 0.99+, including the 3 false positives, so it does
not separate them (see below).

Every miss, from the series:
- **3 FP — dev_008 NL, dev_009 FR, dev_010 US**: `single_channel` periods in markets with only
  2 channels. "All channels but one stop" and "one channel stops while the other runs" are the
  same spend pattern there; the label can't be recovered from spend. Will be revisited with the
  single-channel detector.
- **FN dev_036 US Google Discovery** (60 days from the first day): excluded on purpose, since a
  channel off from day one looks like a late launch.
- **FN dev_040 PL Cinema** (5 days): below `MIN_DAYS` = 7, by design.
- **FN dev_042 US YouTube** (all 730 days): the channel never spent, so there is no "on" level
  to stop from.

## Single channel

**Definition** (`benchmark/BENCHMARK.md:40`): all channels in a market stop spending except one.
The event names the channel that stayed on. "Stop" uses the same 15 % definition.

**Method** (`holdout.py: find_single_channel`, reusing the holdout's layer-C P(off), same prior):
1. Mark each active channel's off spells (15 %-checked).
2. Keep runs of at least 7 days in which exactly one active channel is on and all others are off;
   the survivor must stay the same through the run.
3. Confidence = mean P(off) of the stopped channels × (1 − mean P(off) of the survivor).

**2-channel markets.** With two channels, "all but one stop" (`BENCHMARK.md:40`) and "one channel
stops while the others run" (`BENCHMARK.md:41`) are the same spend pattern, and the docs give no
tie-break. We take the literal reading and report it as a **natural holdout**; single channel
needs 3+ active channels. On dev this costs 3 single-channel FNs, which are also the 3 holdout FPs.
The opposite rule would fix those 3 but break dev's 2 detectable 2-channel holdouts
(dev_011, dev_027): 4 errors instead of 6. It is not adopted because that choice would be made
from dev's mix, not from the definitions.

`pulse.detect:detect` now emits pulse, holdout and single-channel events; P(off) is computed once
per market and shared by holdout and single channel. Full dev run: 178 s.

**Dev scores** (test split untouched):

| | teammate detection-library | this (layer C) |
|---|---|---|
| single_channel P / R / F1 | 1.000 / 0.700 / 0.824 (7 TP / 0 FP / 3 FN) | 1.000 / 0.700 / 0.824 (7 / 0 / 3) |
| boundary error (median / p90 days) | – | 0 / 0 |
| confidence (7 detections) | – | 0.992–0.997, all correct |
| natural_holdout | 16 / 3 / 4 | 17 / 3 / 3 (unchanged) |
| channel_pulse | 10 / 0 / 0 | 10 / 0 / 0 (unchanged) |
| null FP per country-year | – | 0.000 |

Every miss: the 3 FNs (dev_008 NL TikTok, dev_009 FR Affiliate, dev_010 US Print) are all in
2-channel markets, reported as holdouts by the rule above.

## Next steps

1. Get pulses with wrong or low-confidence detections (a harder benchmark or real data) so the
   confidence score can be checked for calibration.
2. Speed up layer C if needed: cap spell lengths (e.g. 120 days, with a tail term) to make it
   O(n·D) instead of O(n²).
