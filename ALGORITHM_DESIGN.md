# How each algorithm works

Each section gives one algorithm, one small dataset and the real output. **Every number here
came from a run** of the real functions in `detection/`. For results, parameter sensitivity
and production readiness, read [REPORT.md](REPORT.md).

## How to run the examples

Run every example from the repository root:

```
PYTHONPATH=. synthetic_data_generator/.venv/bin/python your_script.py
```

## Reading the diagrams

In a series diagram, `.` means the channel spent money, `X` means the channel was off, and
`?` means the export carried no row for that day.

## The pipeline in one list

1. **Preprocessing** turns raw spend into a scale-free series (`detection/io/normalize.py`).
2. **P1** finds runs of off-days (`detection/primitives/zero_runs.py`).
3. **P2** finds level shifts and pairs them (`detection/primitives/level_shift.py`).
4. **P3** groups repeated off-runs into a pulse train (`detection/primitives/pulse.py`).
5. **P4** finds dormant starts and ends (`detection/primitives/onset.py`).
6. **Regime composition** names each window (`detection/compose/label.py`).
7. **The cross-market layer** asks the peers (`detection/compose/cross_market.py`).

## Shared inputs

The **active level** `L` is the median spend over the days a channel actually ran. The
**scale-free series** is `y = log1p(spend / L)`. Division by `L` removes market size, and the
panel spans a 15x range between its largest and smallest market. P2 reads a 7-day centred
rolling median of `y`. P1 reads the raw daily series, so each run lands on an exact date.

## 1. P1, off-runs

P1 finds maximal runs of days where a channel was effectively off, then judges each run
against the series' own history. A day is off when `spend <= max(EPS_ABS, RHO * L)`, and
`RHO` is 0.15, so market size cancels. A run is notable when it is at least `MIN_DAYS` long,
and also at least `RUN_RATIO` times the 90th percentile of the series' **other** off-runs.
That ratio part applies only above `MIN_RUNS_FOR_RATIO` other runs. A flighting channel with
three-day gaps therefore needs a much longer run than a channel that is otherwise never off.

The dataset is one channel, 56 days, active level 200. It takes a 3-day break every eight
days, cuts spend to 25 on days 23 to 25, and stops for nine days on days 46 to 54.

```
index 0..55, one character per day, '.' = on, 'X' = off
spend   .......XXX.....XXX.....XXX.....XXX.....XXX....XXXXXXXXX.
active_level          = 200.0
off threshold         = 30.0
spend on days 23..25  = [25.0, 25.0, 25.0]

run lengths           = [3, 3, 3, 3, 3, 9]
run 0: n_days=3 others=[3, 3, 3, 3, 9] p90=6.60 floor=max(7, 3*6.60)=19.80 notable=False
run 5: n_days=9 others=[3, 3, 3, 3, 3] p90=3.00 floor=max(7, 3*3.00)=9.00 notable=True

find_off_runs:
  2024-01-08..2024-01-10 n_days= 3 kind=exact_zero depth=1.000 edge=1.000 notable=False
  2024-01-24..2024-01-26 n_days= 3 kind=near_zero  depth=0.875 edge=1.000 notable=False
  2024-02-16..2024-02-24 n_days= 9 kind=exact_zero depth=1.000 edge=1.000 notable=True
```

One series gives two very different floors, because the rule asks what is unusual **for this
channel**. Each run also carries `kind`, which separates `exact_zero`, `near_zero` and
`missing` windows, `depth`, which is `1 - mean(window) / L`, and `edge_sharpness`, which
reads the flanking days.

## 2. P2, level shifts

P2 finds the days where a channel's spend level changed and stayed changed. At each candidate
day `t` it compares a 21-day window before `t` against a 21-day window from `t` on the
smoothed series `y`. It subtracts the two medians and divides by a noise scale built from the
median absolute deviation. That ratio is the **z**. Three gates then run in order:

1. **The z gate.** `abs(z)` must reach `Z_THRESH`, which is 3.5.
2. **The persistence gate.** The new level must still hold `PERSIST` days later, at half the
   original delta and in the same direction. This rejects spikes.
3. **The sharpness gate.** At least `SHARPNESS` of the total change must land inside
   `SHARPNESS_WINDOW` days. This rejects gradual ramps. Persistence alone does not reject a
   ramp, because a ramp's new level genuinely does hold.

A final pass keeps only the strongest candidate in each 21-day neighbourhood. The dataset is
one channel, 130 days, 5% Gaussian noise, seed 7. Spend sits at 100, rises to 300 on day 45,
and returns to 100 on day 90.

```
n days = 130  2*W+1 = 43
spend, one number per day, rounded to the nearest 10:
  day  40:  101  100   94  100  107  277  313  302  290  330  311  282  301  309  297  310  299  310  322  290
  day  80:  294  295  305  298  297  283  300  293  318  310  100  103   98  105  100  103   94  102   92   90
```

### Why the scale must be two-sided

P2 measures the noise scale on each window separately and takes the larger of the two.

```python
def _noise_sigma(before, after):
    return max(_robust_sigma(before), _robust_sigma(after))
```

A pooled scale fails, and it fails worst at the one place that matters. At a true change
point the pooled sample holds an even mixture of both levels. Its median absolute deviation
is therefore about half the step. The z collapses to about `2 / MAD_TO_SIGMA`, which is
1.349, whatever the step size. The z gate then had a notch exactly where a step change sits.

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

The two-sided scale gives `z = 13.970`, almost four times the gate of 3.5. The pooled scale
gives `z = 1.365`, which fails the gate. The predicted collapse value is 1.349 and the
measured value is 1.365. A pooled scale would miss this step completely.

### The three gates, day by day

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

Every row clears the z gate, because a 21-day window that straddles the step still sees the
full delta from five days either side. Every row clears persistence, because `persist_held`
stays close to `delta` and keeps its sign. Only days 44, 45 and 46 clear the sharpness gate
of 0.6, because everywhere else the change already sits inside one of the two 3-day windows.
Sharpness dates the change point, and the z does not. Suppression then keeps one survivor.

```
find_level_shifts:
  at=2024-02-14 index=44 z=+13.970 delta=+0.6985 ratio=3.029 sharpness=0.899
  at=2024-03-30 index=89 z=-13.641 delta=-0.6820 ratio=0.337 sharpness=0.962

find_step_episodes:
  2024-02-14..2024-03-29 ratio=3.029 z=+13.970 open_ended=False
```

P2 dates the planted day-45 rise at index 44, because the rolling median resolves a date only
to within its half-width of three days. `ratio=3.029` is the raw median after the shift over
the raw median before it, against a planted ratio of 3.0. The two shifts have opposite signs
and comparable magnitude, so `find_step_episodes` pairs them. A step that never reverts has
no closing shift:

```
A series that rises on day 65 and never comes back:
  shift at=2024-03-05 z=+11.284
  episode 2024-03-05..2024-05-09 open_ended=True
```

**Known limitation.** `detection/compose/label.py` drops every open-ended episode, so a real
budget change that never reverts never reaches the output. On the development split this rule
holds back 187 open-ended episodes against 11 bounded ones. You must relax it for production,
and you must add a market-wide collapse rule first. See REPORT.md section 11, "The step pass is
held together by a rule that must be relaxed for production".

## 3. P3, pulse grouping

A flighting channel produces many off-windows. P3 groups them into **one** event from the
first start to the last end, and attaches the windows as components. The benchmark's truth
groups them the same way, so one event per off-window would score an overlap too low to
match. Two conditions must hold. The series needs at least `PULSE_MIN_RUNS` notable off-runs,
which is 2, and runs that touch either end do not count. The interquartile range of the run
lengths, divided by their median, must not exceed `PULSE_LEN_IQR_RATIO`, which is 0.5.

```
--- 3 off-windows, series length 64 days
    ...............XXXXXXXX..........XXXXXXXX..........XXXXXXXX.....
    run lengths : [8, 8, 8]  notable: [True, True, True]
    train 2024-01-16..2024-02-28 n_pulses=3
    components = [('2024-01-16', '2024-01-23'), ('2024-02-03', '2024-02-10'), ('2024-02-21', '2024-02-28')]

--- two notable runs of very different length
    notable run lengths: [8, 40]
    median=24.0 iqr=16.0 iqr/median=0.667 limit=0.5
    find_pulse_trains -> []

--- 5 off-windows, series length 100 days
    run lengths : [8, 8, 8, 8, 8]  notable: [True, True, True, True, True]
    train 2024-01-16..2024-04-04 n_pulses=5

--- 6 off-windows, series length 118 days
    run lengths : [8, 8, 8, 8, 8, 8]  notable: [False, False, False, False, False, False]
    find_pulse_trains -> [] (no train)
```

The train spans 44 days and only 24 of those are off-days. REPORT.md section 11 explains why
the validity gate treats them separately. At six windows every run has five other runs, so
P1's ratio rule applies. The p90 is 8, the floor becomes 24, and no 8-day run clears it.

**Known limitation.** A regular pulse train of six or more windows is never grouped. The
windows are still found, because regime composition never consults notability. They are
emitted as one event per window, typed `natural_holdout` or `dark_period` rather than
`channel_pulse`. Expect type error and fragmentation on real flighting data, not silence. See
REPORT.md section 11, "A regular pulse train of six or more windows is mis-typed and
fragmented".

## 4. P4, onsets

P4 looks at the two edges of a series. An off-run that touches the start is a dormant start,
and an off-run that touches the end is a discontinuation. Both must be at least `MIN_DAYS`
long. A dormant start is a **candidate** only. It becomes a `staggered_launch` event when the
cross-market layer finds that the channel was already live in another market.

```
late   XXXXXXXXXXXXXXXXXX......................
early  ........................................
stops  ............................XXXXXXXXXXXX

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

The `early` series has no off-runs at all, so both functions return `None`. A channel that was
live from the first day has no onset to report.

## 5. Regime composition

This layer turns primitives into named events. It does not cluster intervals, because
clustering would merge unrelated concurrent events. Per market it computes the
**active-channel set** for each day and cuts the timeline wherever that set changes. Each
maximal run of a constant set is a **regime**, and each regime names itself from the set:

| active set `A`, over the market's channels `K` | event |
|---|---|
| `A` is empty | `dark_period` |
| `A` has one channel and two or more went off | `single_channel`, naming the live channel |
| `A` is missing some but not all of `K` | `natural_holdout` per off channel |
| `A` is all of `K`, and a step episode spans it | `step_change` |

A dark period simply **is** the regime where nothing is active, so the nesting of event types
resolves structurally. Three claims settle first, because each owns windows that the regime
pass would otherwise report twice. A pulse train claims every off-window it groups. A channel
that was never live on one side of a regime belongs to P4 and the cross-market layer. A step
episode that overlaps a notable off-run of its own channel is the level shift that the
off-run already explains. The composer also skips a regime shorter than `MIN_DAYS`.

```
70 days. '.' = spend, 'X' = zero spend, '?' = no row in the export.
            0000000000111111111122222222223333333333444444444455555555556666666666
DE/Search  XXXXXXXXXX..............................................?????XXXX.....
DE/Social  XXXXXXXXXX....................XXXXXXXXXXXX............................
DE/Video   XXXXXXXXXX....................XXXXXXXXXXXX............................
FR/Search  XXXXXXXXXX............................................................
FR/Social  XXXXXXXXXX............................................................
FR/Video   XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX........................................

Regimes for DE (active-channel set per day, cut where the set changes):
  2024-01-01..2024-01-10   10 days  active=[]
  2024-01-11..2024-01-30   20 days  active=['Search', 'Social', 'Video']
  2024-01-31..2024-02-11   12 days  active=['Search']
  2024-02-12..2024-02-25   14 days  active=['Search', 'Social', 'Video']
  2024-02-26..2024-03-05    9 days  active=['Social', 'Video']
  2024-03-06..2024-03-10    5 days  active=['Search', 'Social', 'Video']

label_market(panel, 'DE', 'mini'):
  dark_period      2024-01-01..2024-01-10 channel=None evidence={'n_channels_off': 3}
  single_channel   2024-01-31..2024-02-11 channel=Search evidence={'n_channels_off': 2}
  natural_holdout  2024-02-26..2024-03-05 channel=Search evidence={}
```

Days 56 to 60 carry no rows, so `off_mask` marks them off through the `present` mask, and the
fifth regime starts on 26 February rather than 2 March. Three of the six regimes became
events. The three with the full channel list did not, because no step episode spans them.
Read the `single_channel` line carefully. Its `channel` field names `Search`, the channel
that is still **running**, while the channels that stopped are Social and Video. That
inversion has already produced two real defects in this codebase.

## 6. The cross-market layer

A holdout whose peers kept running has a ready-made control group. The same holdout with
every peer also off has no control at all, and it looks exactly like a pipeline outage. Two
rules govern which peers get a vote. A market that never bought a channel abstains, because
its series is all zeros. The question is always asked about the channels that **stopped**.
For a holdout or a pulse train that is the event's own channel. For a dark period it is every
channel the market runs. For a single-channel period the subject is every **other** channel.
A peer counts as off only when every subject channel it runs was off for the whole window.
The verdict goes into `evidence["control_available"]`:

- `peers` when at least one peer kept running.
- `none` when no peer ran and at least one peer was also off, plus a `global_pause` tag.
- `sibling_channels` when no peer carries these channels at all.

The comparison reads `off_mask` and not a rescaled series, because `off_mask` already
thresholds each series at a fraction of that series' own level. This layer also owns
`staggered_launch` events. When the onset spread across markets reaches `ONSET_SPREAD`, which
is 14 days, every market except the earliest gets one. The panel is the panel from section 5.

```
For each event: which channels the peers are asked about, and the answer.
  DE dark_period      2024-01-01..2024-01-10 channel=None subjects=['Search', 'Social', 'Video'] peers_running=0 peers_off=1
  DE single_channel   2024-01-31..2024-02-11 channel=Search subjects=['Social', 'Video'] peers_running=1 peers_off=0
  DE natural_holdout  2024-02-26..2024-03-05 channel=Search subjects=['Search'] peers_running=1 peers_off=0

annotate(events, panel):
  DE dark_period      2024-01-01..2024-01-10 tags=('global_pause',) control=none
  DE single_channel   2024-01-31..2024-02-11 tags=() control=peers
  DE natural_holdout  2024-02-26..2024-03-05 tags=('cross_market_holdout',) control=peers
  FR dark_period      2024-01-01..2024-01-10 tags=('global_pause',) control=none

Onsets per market for each channel:
  Video   DE: first_active=2024-01-11 (dormant 10 days)
  Video   FR: first_active=2024-01-31 (dormant 30 days)

find_staggered_launches(panel, 'mini'):
  FR Video 2024-01-01..2024-01-30 tags=('cross_market_control',) evidence={'onset': '2024-01-31', 'earliest_market_onset': '2024-01-11', 'n_markets': 2}
```

Read the `single_channel` row. Its subjects are `['Social', 'Video']` and not `Search`, and
asking the peers about `Search` would ask the opposite question. Both dark periods get
`control=none`, because the other market stopped on the same ten days. Video has an onset
spread of 20 days, so DE is the control and FR gets the event.

## Where to read more

- [REPORT.md](REPORT.md) section 2 covers the final design per event type, with test scores.
- [REPORT.md](REPORT.md) section 5 covers parameter sensitivity.
- [REPORT.md](REPORT.md) section 8 covers production readiness per detector.
- [REPORT.md](REPORT.md) section 11 covers what this benchmark cannot tell you.
- `detection/params.py` holds every threshold, with the failure mode of getting it wrong.
