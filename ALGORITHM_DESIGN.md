# How each algorithm works

This document teaches the mechanism of the informative-period detector. Each algorithm gets
a short description, the essential lines of code, a small dataset, a walk through the steps,
and the output.

**Every number in this document came from a run.** Each example is a script that calls the
real functions in `detection/`. The printed result goes into this document unchanged. No
number here is a prediction, a rounding or a guess.

For results, parameter sensitivity and production readiness, read
[REPORT.md](REPORT.md). This document does not repeat any of that.

## How to run the examples

Run every example from the repository root:

```
PYTHONPATH=. synthetic_data_generator/.venv/bin/python your_script.py
```

Every example imports the real function from `detection/`. No example reimplements an
algorithm.

## Reading the diagrams

Several examples print a series as one character per day:

- `.` means the channel spent money that day.
- `X` means the channel was off that day.
- `?` means the export carried no row for that day.

## The pipeline in one list

1. **Preprocessing** turns raw spend into a scale-free series (`detection/io/normalize.py`).
2. **P1** finds runs of off-days (`detection/primitives/zero_runs.py`).
3. **P2** finds level shifts and pairs them into episodes
   (`detection/primitives/level_shift.py`).
4. **P3** groups repeated off-runs into a pulse train (`detection/primitives/pulse.py`).
5. **P4** finds dormant starts and ends (`detection/primitives/onset.py`).
6. **Regime composition** cuts each market's timeline and names each window
   (`detection/compose/label.py`).
7. **The cross-market layer** asks the peer markets what they did
   (`detection/compose/cross_market.py`).

---

## Shared inputs

Two definitions from preprocessing apply to every section below
(`detection/io/normalize.py`).

- The **active level** `L` is the median spend over the days a channel actually ran.
- The **scale-free series** is `y = log1p(spend / L)`.

Dividing by `L` removes market size. The benchmark panel spans a 15x range between its
largest and smallest market. A log ratio keeps one noise threshold correct across that
whole range.

P2 reads a **7-day centred rolling median** of `y` for its level comparisons. P1 reads the
raw daily series instead, so each run starts and ends on an exact date.

---

## 1. P1, off-runs

### What it does and why

P1 finds maximal runs of days where a channel was effectively off. It then decides which
runs are worth reporting.

The rule has three parts:

1. **The off-day rule.** A day is off when `spend <= max(EPS_ABS, RHO * L)`. `RHO` is 0.15,
   so the threshold is 15% of the channel's own active level. The threshold is a fraction of
   the level and not a constant euro amount, so market size cancels.
2. **Run finding.** `_spans` walks the boolean mask once and returns each maximal block of
   `True`.
3. **The notability rule.** A run is notable when it is at least `MIN_DAYS` long, and also at
   least `RUN_RATIO` times the 90th percentile of the series' **other** off-runs.

The notability rule is relative to the series' own history, and that is the point. A
flighting channel whose normal gaps are three days needs a much longer run before anything is
reported. A channel that is otherwise never off needs only `MIN_DAYS`. One global threshold
fails one of those two cases whichever value you pick.

The ratio part switches on only when the series has at least `MIN_RUNS_FOR_RATIO` other runs.
Below that there is no usable distribution, so `MIN_DAYS` alone applies.

### The code

```python
from detection.primitives.zero_runs import off_mask, find_off_runs

mask = off_mask(s)                  # s <= max(EPS_ABS, RHO * active_level(s))
runs = find_off_runs(s)             # each run carries n_days, kind, depth, notable
```

The notability decision inside `find_off_runs`:

```python
others = np.delete(lengths, idx)
if others.size >= params.MIN_RUNS_FOR_RATIO:
    floor = max(params.MIN_DAYS, params.RUN_RATIO * float(np.percentile(others, 90)))
else:
    floor = params.MIN_DAYS
notable = n_days >= floor
```

### The dataset

One channel, 56 days, active level 200. The channel takes a 3-day break every eight days.
On days 23 to 25 it cuts spend to 25 rather than to zero. On days 46 to 54 it stops for nine
days.

```
index 0..55, one character per day, '.' = on, 'X' = off
spend   .......XXX.....XXX.....XXX.....XXX.....XXX....XXXXXXXXX.
```

### The walk through

1. `active_level` returns 200.
2. The off threshold is `max(1e-06, 0.15 * 200) = 30`.
3. The 25-euro days fall under 30, so they are off days. This is a near-zero event.
4. `_spans` returns six blocks.
5. Each run is then judged against the other five.

### The output

```
active_level          = 200.0
RHO * active_level    = 30.0
off threshold         = 30.0
spend on days 23..25  = [25.0, 25.0, 25.0]
spans (lo, hi)        = [(7, 9), (15, 17), (23, 25), (31, 33), (39, 41), (46, 54)]

run lengths           = [3, 3, 3, 3, 3, 9]
run 0: n_days=3 others=[3, 3, 3, 3, 9] p90=6.60 floor=max(7, 3*6.60)=19.80 notable=False
run 5: n_days=9 others=[3, 3, 3, 3, 3] p90=3.00 floor=max(7, 3*3.00)=9.00 notable=True

find_off_runs:
  2024-01-08..2024-01-10 n_days= 3 kind=exact_zero depth=1.000 edge=1.000 notable=False
  2024-01-16..2024-01-18 n_days= 3 kind=exact_zero depth=1.000 edge=1.000 notable=False
  2024-01-24..2024-01-26 n_days= 3 kind=near_zero  depth=0.875 edge=1.000 notable=False
  2024-02-01..2024-02-03 n_days= 3 kind=exact_zero depth=1.000 edge=1.000 notable=False
  2024-02-09..2024-02-11 n_days= 3 kind=exact_zero depth=1.000 edge=1.000 notable=False
  2024-02-16..2024-02-24 n_days= 9 kind=exact_zero depth=1.000 edge=1.000 notable=True
```

Read the two floor lines. P1 judges the 9-day run against five 3-day runs, so its floor is
9.00 and it clears the floor exactly. P1 judges a 3-day run against four 3-day runs and the
9-day run, so its p90 is 6.60 and its floor is 19.80. The same series gives two very different
floors, because the rule asks what is unusual **for this channel**.

Three other fields come out of the run:

- `kind` is `near_zero` for the 25-euro window and `exact_zero` for the rest. `missing` marks
  a window where the export carried no rows at all.
- `depth` is `1 - mean(window) / L`. An exact stop scores 1.000. The 25-euro window scores
  0.875, because 25 is one eighth of 200.
- `edge_sharpness` compares the flanking days against the level. A clean stop and a clean
  restart score 1.000.

---

## 2. P2, level shifts

### What it does and why

P2 finds the days where a channel's spend level changed and stayed changed. It works on the
smoothed scale-free series `y` defined in Shared inputs.

At each candidate day `t` it compares a 21-day window before `t` against a 21-day window from
`t`. It takes the median of each window, subtracts them, and divides by a noise scale built from
the median absolute deviation. That ratio is the **z**. Medians and the median absolute
deviation both ignore outliers, so one extreme day cannot move the z.

Three gates then run in order:

1. **The z gate.** `abs(z)` must reach `Z_THRESH`, which is 3.5.
2. **The persistence gate.** The new level must still hold `PERSIST` days later, at half the
   original delta and in the same direction. This rejects spikes.
3. **The sharpness gate.** At least `SHARPNESS` of the total change must land inside
   `SHARPNESS_WINDOW` days. This rejects gradual ramps. Persistence alone does not reject a
   ramp, because a ramp's new level genuinely does hold.

A final pass keeps only the strongest candidate in each 21-day neighbourhood, so one step
does not report as a cluster.

### Why the scale must be two-sided

P2 measures the noise scale on each window separately and takes the larger of the two. It
never measures the scale across the pair.

```python
def _noise_sigma(before, after):
    return max(_robust_sigma(before), _robust_sigma(after))
```

A pooled scale fails, and it fails worst at the one place that matters. At a true change
point the pooled sample holds an even mixture of both levels. Its median absolute deviation
is therefore about half the step. The z collapses to about `2 / MAD_TO_SIGMA`, which is
1.349, whatever the step size. The z gate then had a notch exactly where a step change sits.
The example below measures that number.

### The code

```python
from detection.primitives.level_shift import find_level_shifts, find_step_episodes

shifts = find_level_shifts(s)          # one LevelShift per surviving candidate
episodes = find_step_episodes(s)       # opposing shifts paired into StepEpisode
```

### The dataset

One channel, 130 days, 5% Gaussian noise, seed 7. Spend sits at 100, rises to 300 on day 45,
and returns to 100 on day 90. P2 needs at least `2 * W + 1 = 43` days, so this example cannot
be as short as the others.

```
n days = 130  2*W+1 = 43
spend, one number per day, rounded to the nearest 10:
  day   0:  100  102   99   96   98   95  100  107   98   97  102  102  100   95  100  104   93   98   90   94
  day  20:   91   99   94  101  101   99   87   97  100  101   92   98   95   96  105   96  100  104   97   99
  day  40:  101  100   94  100  107  277  313  302  290  330  311  282  301  309  297  310  299  310  322  290
  day  60:  303  293  302  282  291  297  314  317  280  288  310  270  293  298  319  310  295  294  296  323
  day  80:  294  295  305  298  297  283  300  293  318  310  100  103   98  105  100  103   94  102   92   90
  day 100:   98   96  101  111   96   97  101  102   99   99  104  103   95  100  100   95  101   96  105  101
  day 120:  100   97   99   90   94  102   89  104   91  104
```

### The walk through: the two scales at day 45

```
At t = 45, the true change point:
  median(before) = 0.6815   median(after) = 1.3800
  delta          = 0.6985
  sigma(before)  = 0.0500
  sigma(after)   = 0.0500
  two-sided sigma = max of the two = 0.0500  ->  z = 13.970
  pooled sigma    = one MAD over both = 0.5116  ->  z = 1.365
  2 / MAD_TO_SIGMA = 1.349
```

The two-sided scale gives `z = 13.970`, which is almost four times the gate of 3.5.
The pooled scale gives `z = 1.365`, which fails the gate. The predicted collapse value is
1.349 and the measured value is 1.365. A pooled scale would miss this step completely.

### The walk through: the three gates, day by day

```
 t   delta      sigma    z       persist_held  sharpness
 40  +0.7060   0.0500   +14.120  +0.6943      0.004
 41  +0.7000   0.0500   +14.000  +0.6883      0.004
 42  +0.7000   0.0500   +14.000  +0.6883      0.002
 43  +0.7000   0.0500   +14.000  +0.6883      0.045
 44  +0.6985   0.0500   +13.970  +0.6868      0.899
 45  +0.6985   0.0500   +13.970  +0.6845      0.948
 46  +0.6970   0.0500   +13.940  +0.6830      0.948
 47  +0.6950   0.0500   +13.900  +0.6810      0.092
 48  +0.6933   0.0500   +13.865  +0.6810      0.041
 49  +0.6933   0.0500   +13.865  +0.6830      0.000
 50  +0.6933   0.0500   +13.865  +0.6830      0.000
```

Read the columns in order:

1. Every row clears the z gate. A 21-day window that straddles the step still sees the full
   delta from five days either side of it.
2. Every row clears persistence. `persist_held` is close to `delta` and has the same sign.
3. Only days 44, 45 and 46 clear the sharpness gate of 0.6. Everywhere else the change is
   already inside one of the two 3-day windows, so the near-difference is near zero.

Sharpness is what turns a wide band of high-z days into a dated change point. Suppression
then keeps one of the three survivors.

### The output

```
find_level_shifts:
  at=2024-02-14 index=44 z=+13.970 delta=+0.6985 ratio=3.029 sharpness=0.899
  at=2024-03-30 index=89 z=-13.641 delta=-0.6820 ratio=0.337 sharpness=0.962

find_step_episodes:
  2024-02-14..2024-03-29 ratio=3.029 z=+13.970 open_ended=False
```

The example planted the rise on day 45, and P2 reports it at index 44. The rolling median
resolves a shift's date only to within its half-width, which is three days. `ratio=3.029` is the raw median
after the shift divided by the raw median before it, and the planted ratio was 3.0.

`find_step_episodes` then pairs the two shifts. They have opposite signs and comparable
magnitude, so they close one episode that runs from the rise to the day before the fall.

### What happens when a step never reverts

```
A series that rises on day 65 and never comes back:
  shift at=2024-03-05 z=+11.284
  episode 2024-03-05..2024-05-09 open_ended=True
```

The episode has no closing shift. Its end date is the last observed day by construction, not
by measurement.

**Known limitation.** `detection/compose/label.py` drops every open-ended episode, so a real
budget change that never reverts is never reported. On the development split this rule holds
back 187 open-ended episodes against 11 bounded ones. You must relax it for production, and
you must add a market-wide collapse rule first. See REPORT.md section 11, "The step pass is
held together by a rule that must be relaxed for production".

---

## 3. P3, pulse grouping

### What it does and why

A flighting channel produces many off-windows. P3 groups them into **one** event that spans
the first start to the last end, and attaches the individual windows as components.

The grouping is not cosmetic. The benchmark's truth groups the windows the same way. A
detector that emitted one event per off-window would score an overlap too low to match the
grouped truth. A pulse detector working perfectly would then match nothing.

Two conditions must hold:

1. At least `PULSE_MIN_RUNS` notable off-runs, which is 2. Runs that touch the start or the
   end of the series do not count, because those are dormant starts and discontinuations.
2. The run lengths must be similar. The interquartile range of the lengths, divided by their
   median, must not exceed `PULSE_LEN_IQR_RATIO`, which is 0.5. A 7-day gap and a 90-day
   shutdown are not one phenomenon.

### The code

```python
from detection.primitives.zero_runs import find_off_runs
from detection.primitives.pulse import find_pulse_trains

trains = find_pulse_trains(find_off_runs(s))
```

### The dataset

One channel, 64 days, spend 500 on active days. The channel runs 10 days and stops 8 days,
three times.

```
--- 3 off-windows, series length 64 days
    ...............XXXXXXXX..........XXXXXXXX..........XXXXXXXX.....
```

### The walk through

1. `find_off_runs` returns three runs of 8 days each.
2. Each run has two other runs, which is below `MIN_RUNS_FOR_RATIO` of 5. The floor is
   therefore `MIN_DAYS`, which is 7, and all three runs are notable.
3. No run touches the start or the end, so all three are eligible.
4. Three eligible runs reach `PULSE_MIN_RUNS`.
5. The lengths are all 8. The median is 8 and the interquartile range is 0, so the ratio is 0
   and the shape check passes.
6. P3 emits one train, from the first start to the last end.

### The output

```
    run lengths : [8, 8, 8]  notable: [True, True, True]
    train 2024-01-16..2024-02-28 n_pulses=3
    components = [('2024-01-16', '2024-01-23'), ('2024-02-03', '2024-02-10'), ('2024-02-21', '2024-02-28')]
```

The train spans 44 days. Only 24 of those are off-days. The other 20 are active days between
pulses. REPORT.md section 11 explains why the validity gate treats them separately.

### The shape check, on two runs of different length

```
--- two notable runs of very different length
    notable run lengths: [8, 40]
    median=24.0 iqr=16.0 iqr/median=0.667 limit=0.5
    find_pulse_trains -> []
```

0.667 exceeds 0.5, so P3 emits no train.

### The same channel, at five windows and at six

```
--- 5 off-windows, series length 100 days
    ...............XXXXXXXX..........XXXXXXXX..........XXXXXXXX..........XXXXXXXX..........XXXXXXXX.....
    run lengths : [8, 8, 8, 8, 8]  notable: [True, True, True, True, True]
    train 2024-01-16..2024-04-04 n_pulses=5

--- 6 off-windows, series length 118 days
    ...............XXXXXXXX..........XXXXXXXX..........XXXXXXXX..........XXXXXXXX..........XXXXXXXX..........XXXXXXXX.....
    run lengths : [8, 8, 8, 8, 8, 8]  notable: [False, False, False, False, False, False]
    find_pulse_trains -> [] (no train)
```

At six windows every run has five other runs, so P1's ratio rule switches on. The p90 of the
other runs is 8, the floor becomes 24, and no 8-day run clears it. Every run loses its
notability at once and P3 groups nothing.

**Known limitation.** A regular pulse train of six or more windows is never grouped. The
windows are still found, because regime composition never consults notability. They are
emitted as one event per window, typed `natural_holdout` or `dark_period` rather than
`channel_pulse`. Expect type error and fragmentation on real flighting data, not silence. See
REPORT.md section 11, "A regular pulse train of six or more windows is mis-typed and
fragmented".

---

## 4. P4, onsets

### What it does and why

P4 looks at the two edges of a series. An off-run that touches the start is a dormant start.
An off-run that touches the end is a discontinuation. Both must be at least `MIN_DAYS` long.

A dormant start is a **candidate** only. It becomes a `staggered_launch` event when the
cross-market layer finds that the channel was already live in another market during the
dormancy. On its own it is a censored holdout, and naming it a `staggered_launch` without
peers would be a guess. Section 6 finishes the job.

### The code

```python
from detection.primitives.onset import find_onset, find_discontinuation

onset = find_onset(s)                  # Onset(first_active, dormant_days) or None
end = find_discontinuation(s)          # the trailing OffRun, or None
```

### The dataset

Three series, one channel each, 40 days.

```
late   XXXXXXXXXXXXXXXXXX......................
early  ........................................
stops  ............................XXXXXXXXXXXX
```

### The walk through

1. `find_off_runs` runs first. It returns an empty list when the channel never spent, so an
   all-off series never reaches the edge checks.
2. `find_onset` looks for a run with `touches_start` true and `n_days >= MIN_DAYS`. It
   returns the first active day, which is the day after the run ends.
3. `find_discontinuation` looks for a run with `touches_end` true and the same length floor.

### The output

```
late: off_runs = [('2024-01-01', '2024-01-18', 18, True, False)]
      find_onset = Onset(first_active=2024-01-19, dormant_days=18)
      find_discontinuation = None
early: off_runs = []
      find_onset = None
      find_discontinuation = None
stops: off_runs = [('2024-01-29', '2024-02-09', 12, False, True)]
      find_onset = None
      find_discontinuation = OffRun(2024-01-29..2024-02-09, n_days=12)
```

The `early` series has no off-runs at all, so both functions return `None`. A channel that
was live from the first day has no onset to report.

---

## 5. Regime composition

### What it does and why

This layer turns primitives into named events. It does not cluster intervals, because
clustering would merge unrelated concurrent events.

Instead, per market, it computes the **active-channel set** for each day and cuts the
timeline wherever that set changes. Each maximal run of a constant set is a **regime**.
Regimes then name themselves from the set:

| active set `A`, over the market's channels `K` | event |
|---|---|
| `A` is empty | `dark_period` |
| `A` has one channel and two or more went off | `single_channel`, naming the live channel |
| `A` is missing some but not all of `K` | `natural_holdout` per off channel |
| `A` is all of `K`, and a step episode spans it | `step_change` |

This resolves the nesting of event types structurally. A dark period simply **is** the regime
where nothing is active. Its per-channel holdouts ride along as components rather than
competing with the event that contains them.

The composer settles three claims before the regime pass. Each of them owns windows that the
regime pass would otherwise report a second time:

1. A pulse train claims every off-window it groups.
2. A channel that was never live on one side of a regime is a dormant start or a
   discontinuation, and P4 plus the cross-market layer own it.
3. A step episode overlapping a notable off-run of its own channel is the level shift that the
   off-run already explains. The composer drops it.

The composer skips a regime shorter than `MIN_DAYS`.

### The code

```python
from detection.io.panel import build_panel
from detection.compose.label import active_matrix, segment_regimes, label_market

panel = build_panel(media_df, sales_df, sid="mini")
regimes = segment_regimes(panel, "DE")
events = label_market(panel, "DE", "mini")
```

### The dataset

Two markets, three channels, 70 days. Sections 7, 8 and 9 use this same panel.

```
70 days. '.' = spend, 'X' = zero spend, '?' = no row in the export.
            0000000000111111111122222222223333333333444444444455555555556666666666
DE/Search  XXXXXXXXXX..............................................?????XXXX.....
DE/Social  XXXXXXXXXX....................XXXXXXXXXXXX............................
DE/Video   XXXXXXXXXX....................XXXXXXXXXXXX............................
FR/Search  XXXXXXXXXX............................................................
FR/Social  XXXXXXXXXX............................................................
FR/Video   XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX........................................
```

The header row gives the tens digit of the day index.

### The walk through

1. Every market stops for the first ten days.
2. DE drops Social and Video on days 30 to 41 and keeps Search alive.
3. DE drops Search alone on days 56 to 64.
4. The export omits the DE/Search rows for days 56 to 60 entirely.
5. FR never ran Video before day 30.

### The output: the regimes

```
channels_in('DE') = ['Search', 'Social', 'Video']
channels_in('FR') = ['Search', 'Social', 'Video']

Regimes for DE (active-channel set per day, cut where the set changes):
  2024-01-01..2024-01-10   10 days  active=[]
  2024-01-11..2024-01-30   20 days  active=['Search', 'Social', 'Video']
  2024-01-31..2024-02-11   12 days  active=['Search']
  2024-02-12..2024-02-25   14 days  active=['Search', 'Social', 'Video']
  2024-02-26..2024-03-05    9 days  active=['Social', 'Video']
  2024-03-06..2024-03-10    5 days  active=['Search', 'Social', 'Video']
```

Note the `?` days. Days 56 to 60 carry no rows, so `off_mask` marks them off through the
`present` mask. The regime therefore starts on 26 February, not on 2 March.

### The output: the events

```
label_market(panel, 'DE', 'mini'):
  dark_period      2024-01-01..2024-01-10 channel=None evidence={'n_channels_off': 3}
  single_channel   2024-01-31..2024-02-11 channel=Search evidence={'n_channels_off': 2}
  natural_holdout  2024-02-26..2024-03-05 channel=Search evidence={}

label_market(panel, 'FR', 'mini'):
  dark_period      2024-01-01..2024-01-10 channel=None evidence={'n_channels_off': 3}
```

Three regimes became events. Two did not, because their active set is the full channel list
and no step episode spans them. The last regime is five days, which is below `MIN_DAYS`.

Read the `single_channel` line carefully. Its `channel` field names `Search`, the channel
that is still **running**. The channels that stopped are Social and Video. That inversion has
already produced two real defects in this codebase, and sections 7 and 8 both depend on
getting it right.

FR shows only a dark period. FR/Video was never live before day 30, so the regime pass does
not call it a holdout. P4 and the cross-market layer own it instead.

### The step_change branch

The 70-day panel above produces no step episode. Here is the same composer on a one-market
one-channel panel that carries the P2 series from section 2:

```
label_market on a one-channel market carrying the P2 step series:
  step_change      2024-02-14..2024-03-29 channel=Search magnitude_ratio=3.029 evidence={'z': 13.969862495785128, 'rose_from_zero': False}
```

The channel never went off, so no off-window suppresses the episode, and the episode is
bounded rather than open-ended.

---

## 6. The cross-market layer

### What it does and why

This layer decides an event's worth more than anything about the event's own shape. A holdout
whose peers kept running has a ready-made control group. The same holdout with every peer
also off has no control at all, and it looks exactly like a pipeline outage.

Two rules govern which peers get a vote:

1. A market that never bought a channel abstains. Its series is all zeros. Reading that as
   "off during the window" would manufacture a global pause out of markets that simply do not
   use the channel.
2. The question is always asked about the channels that **stopped**. For a holdout or a pulse
   train that is the event's own channel. For a dark period it is every channel the market
   runs. For a single-channel period the event names the channel still live, so the subject is
   every **other** channel.

A peer counts as off only when every subject channel it runs was off for the whole window.
One live day on one channel makes it a control.

The verdict goes into `evidence["control_available"]`:

- `peers` when at least one peer kept running.
- `none` when no peer ran and at least one peer was also off. The event also gets the
  `global_pause` tag.
- `sibling_channels` when no peer carries these channels at all.

The comparison runs on `off_mask` and not on a rescaled series. `off_mask` already thresholds
each series at a fraction of that series' own level, so market size cancels before the
comparison starts. Dividing by a market-scale proxy first would be a no-op.

This layer also owns `staggered_launch` events. It collects each market's onset for a
channel. If
the spread between the earliest and the latest onset reaches `ONSET_SPREAD`, which is 14
days, every market except the earliest gets a `staggered_launch`. The earliest market has no
witness that the channel was runnable, so it is the comparison group and not an event.

### The code

```python
from detection.compose.cross_market import annotate, find_staggered_launches

events = annotate(events, panel)                 # adds control_available and tags
events.extend(find_staggered_launches(panel, sid))
```

### The dataset

The same two-market panel from section 5.

### The walk through

1. `_subject_channels` picks the channels each event makes a claim about.
2. `_peer_status` counts the peers that kept running and the peers that were also off.
3. `annotate` writes the verdict into the evidence and adds the tag.

### The output

```
For each event: which channels the peers are asked about, and the answer.
  DE dark_period      2024-01-01..2024-01-10 channel=None subjects=['Search', 'Social', 'Video'] peers_running=0 peers_off=1
  DE single_channel   2024-01-31..2024-02-11 channel=Search subjects=['Social', 'Video'] peers_running=1 peers_off=0
  DE natural_holdout  2024-02-26..2024-03-05 channel=Search subjects=['Search'] peers_running=1 peers_off=0
  FR dark_period      2024-01-01..2024-01-10 channel=None subjects=['Search', 'Social', 'Video'] peers_running=0 peers_off=1

annotate(events, panel):
  DE dark_period      2024-01-01..2024-01-10 tags=('global_pause',) control=none
  DE single_channel   2024-01-31..2024-02-11 tags=() control=peers
  DE natural_holdout  2024-02-26..2024-03-05 tags=('cross_market_holdout',) control=peers
  FR dark_period      2024-01-01..2024-01-10 tags=('global_pause',) control=none
```

Read the `single_channel` row. Its subjects are `['Social', 'Video']` and not `Search`.
Asking the peers about `Search` would ask the opposite question. It would answer "peers"
whenever one channel keeps running everywhere.

Both dark periods get `control=none` and the `global_pause` tag, because the other market
stopped on the same ten days.

### The output: `staggered_launch` events

```
Onsets per market for each channel:
  Search  DE: first_active=2024-01-11 (dormant 10 days)
  Search  FR: first_active=2024-01-11 (dormant 10 days)
  Social  DE: first_active=2024-01-11 (dormant 10 days)
  Social  FR: first_active=2024-01-11 (dormant 10 days)
  Video   DE: first_active=2024-01-11 (dormant 10 days)
  Video   FR: first_active=2024-01-31 (dormant 30 days)

find_staggered_launches(panel, 'mini'):
  FR Video 2024-01-01..2024-01-30 tags=('cross_market_control',) evidence={'onset': '2024-01-31', 'earliest_market_onset': '2024-01-11', 'n_markets': 2}
```

Search and Social have a spread of zero days, so they produce nothing. Video has a spread of
20 days, which reaches `ONSET_SPREAD`. DE started Video first, so DE is the control and FR
gets the event.

---

## Where to read more

- [REPORT.md](REPORT.md) section 2 covers the final design per event type, with test scores.
- [REPORT.md](REPORT.md) section 5 covers parameter sensitivity.
- [REPORT.md](REPORT.md) section 8 covers production readiness per detector.
- [REPORT.md](REPORT.md) section 11 covers what this benchmark cannot tell you.
- [REPORT.md](REPORT.md) section 4 covers scoring and calibration.
- `detection/params.py` holds every threshold, with the failure mode of getting it wrong.
