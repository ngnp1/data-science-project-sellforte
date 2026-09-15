# Informative-period detection — handover

**Branch:** `detection-library` · **Deliverable:** `detection/` · **Evaluation:** `benchmark/`
**Final test-split run:** 2026-09-15T13:24:55Z, run #1, `detection/` source hash `aef7282c…`

This detector finds windows in marketing spend panel data where spend behaviour makes the
data unusually informative for marketing-mix modelling: periods when everything stopped,
when only one channel ran, when one channel paused while its peers kept going, when a budget
stepped to a new level, when a channel flighted on and off, and when a channel launched into
different markets on different dates.

It was evaluated **black-box**. A 100-scenario synthetic benchmark was generated and the
55-scenario test split was **sealed before any detector code existed** (seal timestamp
2026-09-13T16:23:57Z, `spec_hash d0eeba79…`). All development happened on the 45 dev
scenarios. The sealed split was read **exactly once**, at the end.
`benchmark/eval/final_runs.jsonl` contains **exactly one line**. There is no repeat-run
banner on this report, and a second run would produce one permanently.

---

## 1. What this detects, and how to run it

### The pipeline, end to end

```
media.csv  ─┐
            ├─► build_panel ──► daily panel: spend, presence, impressions, clicks, sales
sales.csv  ─┘   (campaigns aggregated to channel; complete date grid; presence mask
                 recorded BEFORE any fill, so a missing row ≠ a zero-spend row)
                     │
                     ▼
        ┌──────── primitives, per country × channel series ────────┐
        │  P1 zero_runs   off-runs, notable relative to the series'│
        │                 own gap distribution                     │
        │  P2 level_shift robust MAD-scaled z on log1p(spend/level)│
        │                 → shifts → paired bounded step episodes  │
        │  P3 pulse       ≥2 notable runs of similar length        │
        │  P4 onset       runs anchored at series start / end      │
        └──────────────────────────────────────────────────────────┘
                     │
                     ▼
        compose/label.py — per market, cut the timeline wherever the ACTIVE-CHANNEL SET
        changes; each maximal constant-set run is a regime and labels itself
        ({} → dark_period, |A|=1 → single_channel, partial → natural_holdout,
         full set + a step episode → step_change)
                     │
                     ▼
        compose/cross_market.py — peer-market check per event (control_available:
        peers / sibling_channels / none; global_pause tag) + staggered launches,
        fanned out to one event per launching market
                     │
                     ▼
        score.py (confidence, informativeness) → calibrate.py (frozen PAV knots)
        → validity.py → explain.py
                     │
                     ▼
        list[DetectedEvent]  — type, market, channel, dates, magnitude_ratio,
        detection_confidence, informativeness, validity + reasons, components,
        evidence dict, natural-language explanation
```

Detection reads **spend only**. Sales is carried on the panel and is read by exactly one
scoring driver (`sales_snr`, an informativeness input); it never influences which events are
found. `tests/detection/test_pipeline.py` runs the whole pipeline with and without the sales
frame and requires identical event identities; that invariant was independently re-verified
across all 45 dev scenarios during review.

### One command

From the repository root. Python is always `synthetic_data_generator/.venv/bin/python`.

```bash
synthetic_data_generator/.venv/bin/python - <<'PY'
import pandas as pd
from detection.pipeline import run_detection

media = pd.read_csv("media.csv", parse_dates=["date"])
sales = pd.read_csv("sales.csv", parse_dates=["date"])   # optional, pass None if absent

for e in run_detection(media, sales, sid="prod"):
    print(e.event_type, e.country_code, e.channel,
          e.start.date(), e.end.date(),
          round(e.detection_confidence, 3), round(e.informativeness, 3), e.validity)
    print("   ", e.explanation)
PY
```

**Required `media.csv` columns:** `date`, `country_code`, `advertising_channel`,
`media_investment`, `impressions`, `clicks`. **`sales.csv`:** `date`, `country_code`,
`turnover`. Campaign-grained rows are summed to channel level inside `build_panel`.

Real output, dev_005:

> `dark_period FR None 2024-12-18 2024-12-31 0.943 0.863 ok`
> FR: every channel stopped together for 14 consecutive days (2024-12-18 to 2024-12-31).
> Total spend across 2 channels in FR fell from a typical 15,103 per day to exactly 0 per
> day over this window. 4 other markets kept at least one of these 2 channels running during
> the window, so they are available as a control. Label: dark_period. Confidence 0.94,
> informativeness 0.86, validity ok.

To re-score against the development split (this **appends a line** to
`benchmark/eval/dev_history.jsonl`):

```bash
synthetic_data_generator/.venv/bin/python -m benchmark.eval.run_dev \
  --detector benchmark.eval.adapter:detect --load-data --label "your note"
```

`benchmark/eval/run_final.py` evaluates the sealed test split. **It has already been run,
once.** Do not run it again: it appends to `final_runs.jsonl` and every subsequent report is
permanently banner-flagged as a repeat run.

---

## 2. Final algorithm design, per event type

Shared preprocessing: each `(country, channel)` series has an **active level** `L_s` = median
of its positive-spend days; the working series is `y_t = log1p(spend_t / L_s)`, which is
scale-free across the benchmark's 15× market-size spread. Level work runs on a 7-day centred
rolling median (`ROLLING`) so day-of-week structure cancels; run detection stays on raw daily
values so boundaries land on exact dates.

A day is **off** when `spend ≤ max(EPS_ABS, RHO · L_s)`. A run of off-days is **notable**
when it is at least `MIN_DAYS` long *and*, where the series has at least
`MIN_RUNS_FOR_RATIO` other runs, at least `RUN_RATIO ×` the p90 of those other runs. That one
rule serves two opposite cases: a channel that is never otherwise off needs 7 days; a
flighting channel with routine 3-day gaps needs 9.

### dark_period — test F1 **1.000** (P 1.000 / R 1.000, 15 TP, 0 FP, 0 FN)

**Primitives:** P1 off-runs on every channel in the market → regime segmentation. The regime
where the active-channel set is empty *is* the dark period; the constituent per-channel
holdouts ride along as `components` rather than competing as separate events. The
cross-market layer adds the `global_pause` tag when every peer market is also off, and the
validity gate raises suspicion for the same shape.
**Parameters:** `RHO`, `EPS_ABS`, `MIN_DAYS`, `RUN_RATIO`, `MIN_RUNS_FOR_RATIO`.
Confidence weights lean on `consistency` (0.25) — a dark period's whole claim is that every
channel stopped together.

### single_channel — test F1 **1.000** (P 1.000 / R 1.000, 12 TP, 0 FP, 0 FN)

**Primitives:** the regime where exactly one channel is active, **more than one** channel went
off, and the off channels were genuinely active in adjacent regimes. The event's `channel`
field names the channel still **running**, which is the benchmark's convention; every
downstream consumer resolves the actual subjects through the shared `subject_channels()`
helper. That inversion has caused three separate real defects in this codebase and is now
single-sourced.
**Parameters:** as dark_period, plus the `len(off) > 1` discriminator (deliberately *not* the
spec's `len(all_channels) ≥ 2`, which would label a two-channel holdout as single_channel).

### natural_holdout — test F1 **0.898** (P 0.957 / R 0.846, 22 TP, 1 FP, 4 FN)

**Primitives:** the regime where some but not all channels are off, emitted per off-channel.
The cross-market layer then answers the question that determines the event's worth: did peer
markets keep running this channel? `control_available` is `peers` (a ready-made control
group — the most MMM-valuable finding here), `sibling_channels`, or `none`.
**Parameters:** `RHO` is decisive — see §5. `MIN_DAYS`, `RUN_RATIO` govern which runs qualify.
The single test-split false positive is a label substitution: one holdout reported as a
staggered launch. Mean IoU 0.991 across all matched pairs says the windows are right and the
residual errors are naming errors, not location errors.

### step_change — test F1 **0.857** (P 1.000 / R 0.750, 12 TP, 0 FP, 4 FN)

**Primitives:** P2. At each candidate boundary, `delta` is the difference of the two 21-day
(`W`) window medians of `y`; **`sigma` is measured on each window separately and the larger
taken** — never pooled across the pair. That detail is the single biggest result of the
project: the pooled form, which the spec describes, puts a *notch* in `|z|` exactly at a true
change point (a pooled sample straddling both levels has MAD ≈ delta/2, so `|z|` collapses to
≈1.35 regardless of step size). Fixing it took dev step recall from 0.154 to 0.846. Shifts
clear `Z_THRESH`, then two filters: **persistence** (`PERSIST`, `PERSIST_FRACTION`) rejects
spikes; **sharpness** (`SHARPNESS`, `SHARPNESS_WINDOW`) rejects gradual ramps. Opposite-sign
shifts of comparable magnitude (`EPISODE_MATCH_BAND`) pair into a bounded episode. Episodes
overlapping a notable off-run of their own channel are dropped — that level shift is the
holdout, already reported. **Episodes with no matching reversal are dropped entirely**; see
§11, this is the most consequential open item in the system.
**Parameters:** `W`, `Z_THRESH`, `SIGMA_FLOOR`, `PERSIST`, `PERSIST_FRACTION`, `SHARPNESS`,
`SHARPNESS_WINDOW`, `EPISODE_MATCH_BAND`, `MAD_TO_SIGMA`.
Reported `magnitude_ratio` is `median(spend after) / median(spend before)` in original units,
directly comparable to a ground-truth multiplier.

### channel_pulse — test F1 **1.000** (P 1.000 / R 1.000, 13 TP, 0 FP, 0 FN)

**Primitives:** P3. At least `PULSE_MIN_RUNS` notable off-runs on one series with similar
lengths (`IQR / median ≤ PULSE_LEN_IQR_RATIO`), emitted as **one** grouped event spanning
first start to last end, with the individual windows attached as `components`. The grouping
is load-bearing, not cosmetic: benchmark truth groups pulses the same way, and a detector
emitting one event per window would score an IoU around 0.12 against grouped truth and match
nothing.
**Parameters:** `PULSE_MIN_RUNS`, `PULSE_LEN_IQR_RATIO`, plus all of P1's.
**Caveat that no score reveals:** a regular train of **six or more** windows is suppressed
entirely. See §11.

### staggered_launch — test F1 **0.970** (P 0.941 / R 1.000, 16 TP, 1 FP, 0 FN)

**Primitives:** P4 finds a run anchored at series start followed by sustained activity — on
its own only a censored holdout. The cross-market layer collects onset dates per channel
across markets; a spread greater than `ONSET_SPREAD` days makes it a staggered launch, then
**fans it out to one event per market**, because truth is per-market and a country-less panel
event would match nothing.
**Parameters:** `ONSET_SPREAD`, `MIN_DAYS`.
The one test-split false positive is the known holdout/launch ambiguity of §11.

---

## 3. Benchmark results on the unseen test set

### Read this number first

| | |
|---|---|
| **False positives on null scenarios** | **0.000 per country-year** |

The 55-scenario test split contains **5 null scenarios with no events at all**. The detector
reported **nothing** in any of them.

This is the headline, not F1, and the harness prints the warning itself: *precision, recall
and F1 cannot tell a silent detector from an indiscriminate one — both land at 0.000 on all
three.* The dev-split baselines make that concrete: `never_detect` scored F1 0.000 at a null
FP rate of 0.000, and `detect_everything` scored F1 0.000 at a null FP rate of **0.654 per
country-year**. Only the FP rate separates them. Quote F1 from this report only alongside the
null rate.

### Headline

| metric | test split (55 scenarios) |
|---|---|
| Precision | 0.978 |
| Recall | 0.918 |
| F1 | 0.947 |
| TP / FP / FN | 90 / 2 / 8 |
| Mean IoU | 0.991 |
| Median IoU | 1.000 |
| Null-scenario FP rate | 0.000 per country-year |
| Channel accuracy (relaxed match) | 1.000 |
| Market accuracy (relaxed match) | 1.000 |
| Day-level F1 | 0.920 |

**Boundary error, matched pairs (n = 90):** start median 0.0 d, p90 1.0 d; end median 0.0 d,
p90 0.0 d. When the detector finds an event, it dates it to the day.

### Per event type

| type | precision | recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| channel_pulse | 1.000 | 1.000 | 1.000 | 13 | 0 | 0 |
| dark_period | 1.000 | 1.000 | 1.000 | 15 | 0 | 0 |
| single_channel | 1.000 | 1.000 | 1.000 | 12 | 0 | 0 |
| staggered_launch | 0.941 | 1.000 | 0.970 | 16 | 1 | 0 |
| natural_holdout | 0.957 | 0.846 | 0.898 | 22 | 1 | 4 |
| step_change | 1.000 | 0.750 | 0.857 | 12 | 0 | 4 |

**Type confusion:** one substitution in the entire split — one `natural_holdout` predicted as
`staggered_launch`. Every other prediction that matched a truth event named the correct type.
Both false positives are label substitutions on correctly located windows; combined with mean
IoU 0.991, the residual error is *naming*, not *finding*.

### Breakdowns (spec §9 item 10)

**Noise level** — the only axis stratified within family, and therefore the only one whose
breakdown is causally interpretable:

| noise | scenarios | P | R | F1 |
|---|---|---|---|---|
| low | 24 | 0.975 | 0.907 | 0.940 |
| med | 16 | 1.000 | 0.929 | 0.963 |
| high | 15 | 0.962 | 0.926 | 0.943 |

Noise does not degrade the detector meaningfully. That is expected: every threshold is a
robust median/MAD statistic on a scale-free series.

**Family:**

| family | scenarios | P | R | F1 |
|---|---|---|---|---|
| cross_market | 4 | 1.000 | 1.000 | 1.000 |
| dark | 4 | 1.000 | 1.000 | 1.000 |
| holdout | 4 | 1.000 | 1.000 | 1.000 |
| launch | 4 | 1.000 | 1.000 | 1.000 |
| pulse | 4 | 1.000 | 1.000 | 1.000 |
| single_channel | 4 | 1.000 | 1.000 | 1.000 |
| step | 6 | 1.000 | 1.000 | 1.000 |
| mixed | 10 | 0.975 | 0.929 | 0.951 |
| **edge** | **10** | **0.900** | **0.643** | **0.750** |
| null | 5 | n/a | n/a | n/a |

**Every clean single-event family scores 1.000, including step.** All of the loss is
concentrated in the 10 edge-case scenarios and, to a lesser degree, the mixed scenarios. The
step family scoring 1.000 while the step *type* scores 0.857 tells you exactly where step
recall is lost: not on clean steps, but on steps embedded in edge and mixed scenarios.

**Structure:**

| n_channels | scenarios | P | R | F1 |
|---|---|---|---|---|
| **2** | **7** | **1.000** | **0.429** | **0.600** |
| 4 | 17 | 0.971 | 0.944 | 0.958 |
| 6 | 11 | 1.000 | 1.000 | 1.000 |
| 9 | 10 | 1.000 | 0.944 | 0.971 |
| 12 | 10 | 0.947 | 0.947 | 0.947 |

| n_countries | scenarios | P | R | F1 |
|---|---|---|---|---|
| 1 | 8 | 1.000 | 1.000 | 1.000 |
| 2 | 17 | 0.952 | 0.800 | 0.870 |
| 3 | 13 | 0.960 | 0.923 | 0.941 |
| 5 | 9 | 1.000 | 0.962 | 0.980 |
| 8 | 8 | 1.000 | 1.000 | 1.000 |

**Two-channel markets are the weakest structural case in the benchmark: recall 0.429.** This
is not a tuning failure — it is a labelling ambiguity present in the truth itself (§11).
Precision stays at 1.000 there, so the detector is silent rather than wrong.

**Years:** 1 year (8 scenarios) 1.000/1.000/1.000; 2 years (47) 0.974/0.904/0.938.

**`trend_p` and `market_spread` are confounded with family on both splits** — most families
sit at a single value of each — so the numbers in those tables measure family difficulty, not
an axis effect. The harness prints that warning and this report repeats it rather than
quoting the tables as findings. In particular, test's `trend_p = 1.0` bucket holds 9 of the
10 edge cases.

**Event-level axes** (`meta.json` carries none of these, so they bucket individual truth
events and are therefore **recall-only** — a false positive belongs to no truth bucket):

| duration | truth events | matched | recall |
|---|---|---|---|
| < 14 days | 1 | 0 | **0.000** |
| 14–41 days | 23 | 23 | 1.000 |
| 42–89 days | 39 | 34 | 0.872 |
| 90+ days | 35 | 33 | 0.943 |

| magnitude | truth events | matched | recall |
|---|---|---|---|
| exact zero | 72 | 68 | 0.944 |
| near zero (0 < m < 0.1) | 10 | 10 | **1.000** |
| reduced (0.1 ≤ m < 1) | 4 | 3 | 0.750 |
| amplified (m ≥ 1) | 12 | 9 | 0.750 |

The single sub-14-day event is the benchmark's deliberate 5-day case, and missing it is the
designed behaviour of `MIN_DAYS = 7` (§7). Near-zero recall of 1.000 is the direct payoff of
the `RHO` change (§5); on dev, before that change, this row read 0.000.

### Development versus test, and why test scored higher

| | dev (45 scenarios, at freeze) | test (55 scenarios, sealed) |
|---|---|---|
| Precision | 0.943 | 0.978 |
| Recall | 0.880 | 0.918 |
| **F1** | **0.910** | **0.947** |
| Null FP / country-year | 0.000 | 0.000 |
| Mean IoU | 0.996 | 0.991 |
| Day-level F1 | 0.883 | 0.920 |
| Channel accuracy | 0.957 | 1.000 |

| type | dev F1 | test F1 |
|---|---|---|
| channel_pulse | 1.000 | 1.000 |
| dark_period | 1.000 | 1.000 |
| single_channel | 0.824 | 1.000 |
| staggered_launch | 0.952 | 0.970 |
| step_change | 0.917 | 0.857 |
| natural_holdout | 0.821 | 0.898 |

**The sealed split scored higher than the split the detector was developed on.** That is the
point of the whole exercise. The normal failure mode of a project like this is a detector
tuned until the development number looks good, which then drops on unseen data — the gap
between the two numbers *is* the overfitting. Here the gap runs the other way: +0.037 F1 in
favour of data the algorithm had never seen. No amount of process documentation proves
absence of overfitting the way that does.

Read it as *"the dev number was not inflated by tuning"*, **not** as *"the detector is better
than 0.910"*. Two honest reasons the test number is higher, neither of them capability:

1. The dev split carries two known **truth-label ambiguities** that cost it real events and
   that the detector deliberately was not fitted to (`dev_036`/`dev_022`; the two-channel
   markets of `dev_008/009/010` versus `dev_011/013` — §11). Those specific scenarios are
   dev-side.
2. 45 and 55 scenarios are small samples of a random draw. A ±0.04 F1 difference between two
   draws of this size is not a measurement of anything.

The defensible claim is: **F1 in the low 0.9s, with no false positives on event-free data, on
a split that was never tuned against.**

---

## 4. Reliability and operating curves — read this section carefully

### The calibrated confidence is a constant

Spec §8 asks confidence to become a testable claim: *events at confidence X are right about
X% of the time on data never seen*. Confidence was binned into deciles on the **development
split only**, empirical precision measured per bin, and a monotone pool-adjacent-violators
mapping fitted and frozen before the final run.

**The fit collapsed to a single knot.** `detection/calibration_fit.py` contains, in full:

```python
KNOTS = [(1.0, 0.9428571428571428)]
```

Every event the detector emits carries `detection_confidence = 0.943`. Exactly. Always.

The test split confirms the claim is honest, and that is all it confirms:

| confidence bin | n | mean confidence | empirical precision |
|---|---|---|---|
| 0.9–1.0 | 92 | 0.943 | 0.978 |

**One point, one bin.** The claim "events at 0.943 are right about 94% of the time on unseen
data" holds — measured 0.978 on the sealed split, slightly *better* than claimed. The claim
that cannot be made is any comparative one. The calibrated operating curve is flat at
P 0.978 / R 0.918 / F1 0.947 for every cut from 0.00 to 0.90, then falls to 0.000/0.000/0.000
at 0.95 because all 92 detections sit below it. **There is no calibrated operating point to
choose.**

### Why it collapsed

PAV is not malfunctioning; it is reporting something true about the score. Measured on the
70 dev detections at freeze, raw confidence is **bimodal, not saturated** (min 0.625, 44 of
70 at ≥ 0.9), and it separates by **event type**, not by correctness:

| type | n | median raw confidence | dev precision |
|---|---|---|---|
| dark_period | 12 | 1.000 | 1.000 |
| natural_holdout | 19 | 1.000 | 0.842 |
| single_channel | 7 | 1.000 | 1.000 |
| staggered_launch | 11 | 0.996 | 0.909 |
| step_change | 11 | 0.710 | 1.000 |
| channel_pulse | 10 | 0.640 | 1.000 |

On dev, confidence is **mildly anti-correlated with correctness**. Every false positive sits
in the high-confidence group (holdout 0.842, launch 0.909), while the two *lowest*-confidence
types are perfectly precise. The raw reliability curve shows the same inversion directly:

| raw bin | n | mean raw | empirical precision |
|---|---|---|---|
| 0.6–0.7 | 13 | 0.644 | 1.000 |
| 0.7–0.8 | 8 | 0.721 | 1.000 |
| 0.8–0.9 | 5 | 0.822 | 1.000 |
| 0.9–1.0 | 44 | 0.998 | **0.909** |

The most confident bin is the least precise. PAV's job is to enforce monotonicity, and an
inversion at the top forces a leftward pooling cascade that swallows the entire range. The
sub-score weights reward the signature of a *clean exact-zero event* (depth 1.0,
corroboration 1.0, consistency 1.0) rather than whatever actually predicts being right.

### The raw operating curve — and it is not a usable trade-off either

Measured on the 45 dev scenarios (70 detections), sweeping the **uncalibrated** raw score.
**This is dev-split, uncalibrated, and is not a test-split result.** Raw confidences on the
sealed split were not extracted; doing so would mean a second final run.

| raw cut | kept | precision | recall | F1 |
|---|---|---|---|---|
| 0.00 – 0.60 | 70 | 0.943 | 0.880 | 0.910 |
| 0.65 | 59 | 0.932 | 0.733 | 0.821 |
| 0.70 | 57 | 0.930 | 0.707 | 0.803 |
| 0.75 | 50 | 0.920 | 0.613 | 0.736 |
| 0.80 | 49 | 0.918 | 0.600 | 0.726 |
| 0.85 – 0.95 | 44 | 0.909 | 0.533 | 0.672 |
| 1.00 | 26 | 0.885 | 0.307 | 0.455 |

Raising the cut **loses recall and loses precision at the same time**. There is no cut
anywhere on this curve that buys precision. That is the anti-correlation above, priced out:
the events you discard first are the ones most likely to be right.

### What to do with this

The honest framing, and the one to carry into any conversation about this detector:

> **We are right about 94% of the time and we cannot tell you which ones.**

Concretely:

- **Do not gate on `detection_confidence`.** It is a constant. A threshold on it either keeps
  everything or discards everything.
- **Do not gate on `raw_confidence`** (carried in `evidence["raw_confidence"]`) either. On the
  only data where it can be checked, gating on it is strictly harmful.
- **Rank on `informativeness` instead.** It is a separate score, it was never pooled, and it
  still discriminates. Spec §8 keeps the two scores apart precisely so that "how sure am I
  this is real" and "how useful is it" do not collapse into one number — and here one of them
  has collapsed while the other has not.
- **Operate by triage, not by threshold.** Take the top N by informativeness and have an
  analyst confirm them. §10.
- The fix is a re-weighting of the confidence sub-scores against correctness, and it was
  deliberately **not** attempted: with 70 detections and 4 false positives there is not enough
  signal to re-weight responsibly, and tuning weights until the reliability curve flatters the
  detector is exactly the overfitting this project was built to avoid. Refit on Sellforte's
  own data, where the sample is larger — `scripts/fit_calibration.py` is self-protecting and
  always fits on the raw score, so it cannot double-calibrate.

---

## 5. Important parameters and their sensitivity

All live in `detection/params.py`, each with its documented failure mode. Values were set
from first principles (weekly multiples, a conventional robust cut) and adjusted **only**
against the development split.

### The ones that matter most

| parameter | value | governs | measured / stated cost of getting it wrong |
|---|---|---|---|
| `RHO` | **0.15** | off-day threshold, as a fraction of the series' own active level | **Changed from the spec's 0.05.** On dev: P 0.867→0.934, R 0.693→0.760, F1 0.770→0.838; `natural_holdout` recall 0.550→0.800 and `step_change` false positives 4→0 from this single change. Null FP unchanged at 0.000. Too low misses near-zero events (which then resurface as spurious level shifts); too high reads ordinary low-spend days as off. **First parameter to revisit on real data** — see §11. |
| `Z_THRESH` | 3.5 | step sensitivity | The dominant precision/recall lever for step changes. The benchmark's 50-day gradual ramp — a non-event — clears the z gate (max \|z\| 4.761 after the sigma fix); `SHARPNESS`, not this, is what rejects it. |
| `SHARPNESS` / `SHARPNESS_WINDOW` | 0.6 / 3 | ramp rejection | **The load-bearing gate against gradual ramps.** The benchmark's own blocked ramp lands 0.355 of its change inside the window, ≈41% below threshold — rejected with room to spare. Persistence does *not* reject a ramp: a ramp's new level genuinely holds. Lowering this turns ramps into false steps. Measured boundary on a synthetic flat rise: ≤4 days reads as a step, ≥7 days never does, crossover near 5. |
| `MIN_DAYS` | 7 | shortest reportable event | At 7 the benchmark's deliberate 5-day event is undetectable and surfaces as an honest false negative (test duration bucket `<14 days`: recall 0.000, n=1). Lowering it to catch that case floods the output with noise. |
| `RUN_RATIO` / `MIN_RUNS_FOR_RATIO` | 3.0 / 5 | intermittent-channel guard | The main false-positive lever on flighting channels. A channel with routine 3-day gaps needs ≈9 off-days to register; a never-off channel needs 7. **A single global threshold fails one of those two cases whichever value is chosen** — and this rule is also what suppresses a six-window pulse train entirely (§11). |
| `W` | 21 | level-shift comparison half-window | A multiple of 7 so day-of-week structure cancels. Shorter gives a noisier z; longer misses short steps. Also the width of the non-maximum suppression neighbourhood, which is where a step adjacent to a pause loses its closing edge (§6). |
| `EPISODE_MATCH_BAND` | 2.0 | step-episode pairing tolerance | Wider pairs a step's end with an unrelated later shift; narrower leaves real episodes open-ended, running to the series end — and open-ended episodes are currently **dropped**. |
| `ONSET_SPREAD` | 14 | staggered-launch trigger | Below it, differing onset dates read as coincidental start-up jitter. |

### The rest

| parameter | value | role |
|---|---|---|
| `EPS_ABS` | 1e-6 | absolute floor so float noise around zero is not spend |
| `SIGMA_FLOOR` | 0.05 | lower bound on MAD scale; a near-constant window cannot produce an unbounded z |
| `PERSIST` / `PERSIST_FRACTION` | 14 / 0.5 | the new level must still hold 14 days later at ≥ half the opening delta — rejects spikes |
| `ROLLING` | 7 | centred rolling median for level work |
| `MAD_TO_SIGMA` | 1.4826 | MAD → σ for a normal distribution (in params because a test bans this literal from logic modules) |
| `PULSE_MIN_RUNS` / `PULSE_LEN_IQR_RATIO` | 2 / 0.5 | pulse-train quorum and length-similarity |
| `Z_SATURATION` | 8.0 | \|z\| at which step magnitude evidence saturates |
| `DURATION_SATURATION_MULT` | 2.0 | duration evidence saturates at 2 × `MIN_DAYS` |
| `DISTINCTIVENESS_SATURATION` | 3.0 | run length vs series p90 at which distinctiveness saturates |
| `CORROBORATION_CONTRADICTED` / `CORROBORATION_UNKNOWN` | 0.2 / 0.6 | spend zero with impressions flowing is tracking loss, not a pause; no impressions data is unknown, not contradicted |
| `CONFIDENCE_WEIGHTS` | 6 rows | per-type sub-score weights, each summing to 1 |
| `TYPE_PRIOR` | 6 values | informativeness prior; dark 1.0 > single_channel / pulse 0.9 > holdout 0.75 > launch 0.6 > step 0.45 |
| `CONTROL_SCORE` | peers 1.0 / siblings 0.7 / none 0.3 | control availability |
| `INFORMATIVENESS_WEIGHTS` | 6 drivers | duration 0.25, control 0.2, contrast 0.15, cleanliness 0.15, sales_snr 0.15, type_prior 0.1 |
| `ADSTOCK_HALF_LIFE` | 7.0 | **assumed, not measured.** Feeds informativeness ranking only — never detection |
| `ADSTOCK_WINDOWS_FOR_FULL_CREDIT` | 4.0 | duration adequacy saturates at 4 half-lives |
| `CENSORING_PENALTY` / `CONFOUNDED_PENALTY` | 0.7 / 0.6 | an event censored at a series edge, or confounded by an overlapping event, is worth less |
| `SALES_BASELINE_WEEKS` | 8 | trailing baseline for sales SNR (matches spec §8's worked example) |
| `SALES_SIGMA_FLOOR_FRAC` / `SALES_SNR_SATURATION` / `SALES_SNR_UNKNOWN` | 0.02 / 3.0 / 0.5 | sales-readability driver; the floor is a *fraction* of the trailing median because the panel spans 15× in market size |
| `CALIBRATION_BINS` | 10 | deciles, per spec §8 |

**Known weak coverage, disclosed rather than papered over:** re-weighting *within* the
ordering of `TYPE_PRIOR`, `CONTROL_SCORE` and `CONFIDENCE_WEIGHTS` is not caught by any test.
Those tables are pinned by their **ordering** (which is what the spec actually asserts), not
their values. Calibration was supposed to be what made the numbers accountable — and
calibration collapsed, so it is not. Treat those weights as policy, not as measurements.

---

## 6. Failure modes, with measured cost

| # | failure | measured cost |
|---|---|---|
| 1 | **Open-ended step episodes are dropped.** A step episode with no matching reversal has an end given by the series end, not by measurement, so it is discarded. | Safe on this benchmark — every generated step reverts inside the series, verified in `benchmark/spec/events.py`. On dev the raw step pass produces **187 open-ended episodes against 11 bounded ones**. A real budget change that never reverts is currently **missed entirely**. §11. |
| 2 | **Market-wide step moves are not collapsed.** A budget change applied to a whole market emits one event per channel rather than one market event. | Currently invisible: 0 same-day 3+-channel groups survive to the output. But in the raw episodes, **8 same-day clusters of up to nine channels** are waiting (dev_035/FR 9, dev_033/FI 8, dev_035/AT 8, dev_030/DE 7, dev_032/PL 7, dev_033/CH 7, dev_034/FI 7, dev_029/US 5). They are suppressed *only* as a side effect of failure 1. Relax 1 without fixing 2 and step precision collapses. |
| 3 | **A step that reverts shortly before the channel pauses loses its closing edge.** `W`-wide non-maximum suppression: the drop into the off-window is a far larger delta and suppresses the step's closing shift inside the 21-day neighbourhood. | Reproduced deliberately: the bounded episode is lost whenever the gap is ≤ 8 days. Not caused by the off-window exclusion (identical with `exclude=[]`). Pre-existing, and **not exercised by the benchmark at all** — truth never puts a step and a holdout on the same channel in the same country. Real data produces this shape routinely. |
| 4 | **A six-or-more-window pulse train is suppressed entirely.** At six windows the `RUN_RATIO` rule switches on and every run's peers are its own length, so none is notable. | Cannot fire on this benchmark (the pulse family draws 2–4 windows), so no score reveals it. On real flighting data it is the normal case. Documented and pinned by a test that states it. **Highest-priority spec question before production use.** §11. |
| 5 | **Two-channel markets are label-ambiguous.** The same spend shape is `single_channel` in some truth families and `natural_holdout` in others. | Test recall 0.429 in the `n_channels = 2` bucket (7 scenarios), precision 1.000. Worth 6 events on dev. No spend-only rule separates them; the detector's rule is a deliberate prior and was not fitted to the scenarios that would reveal it. |
| 6 | **Holdout versus staggered launch on a dormant market.** A market dormant for the first 60 days with no ramp-up is both, depending on which family drew it. | 1 FP + 1 FN on the test split (the single type substitution). On dev, launch wins 10 events to 1. Benchmark ambiguity, not a detector defect. |
| 7 | **Edge-censored windows.** A holdout at day 0, at the series end, or spanning the whole series. | Accounted for 3 FN plus the single launch FP on dev. On test, the edge family scores P 0.900 / R 0.643 / F1 0.750 — the worst family by a wide margin, and where essentially all remaining loss lives. |
| 8 | **A channel that starts and immediately stops is still called a launch.** Nothing checks that a launch is *sustained*. | Not exercised anywhere on dev, so no gate was added (this branch's standard: no gate that no input can trip). Real data will need one. |
| 9 | **`channel_discontinued` and `multi_channel_holdout` are not emitted**, though spec §7 names them. | Deliberate: the benchmark cannot generate either, so emitting them would produce false positives by construction, and grouping N holdouts into one event would score 1 TP + (N−1) FN instead of N TP. On real data a discontinued channel is a real phenomenon and will be reported as a censored holdout instead. |
| 10 | **A slow phase-in reads as a ramp and is dropped.** The ramp/step sharpness boundary moved from ≈2 days to ≈5 when the sigma fix landed. | A real budget change phased in over more than ~5 days will be rejected by `SHARPNESS`; a phase-in of ≤4 days now reads as a step where it previously did not. Both are sharpness decisions; neither involves the z gate. |
| 11 | **`global_pause` is overloaded.** The harness uses it for panel-wide dark periods; the detector applies it to any all-peers-off event. | No score impact — no harness metric reads tags. The two meanings should be separated before anything relies on the tag. |

---

## 7. Edge-case behaviour

| case | behaviour | evidence |
|---|---|---|
| **5-day event, below `MIN_DAYS`** | Not detected. **This is correct behaviour**, not a bug: it is reported as an honest false negative rather than chased. | Test duration bucket `< 14 days`: 1 truth event, 0 matched, recall 0.000. |
| **Censored start / censored end** | Detected as events, discounted via the informativeness `CENSORING_PENALTY` (0.7) because the true extent is unknown. A start-anchored run is a launch *candidate* only until cross-market confirms it. | Edge family test recall 0.643; censoring is a main driver. On dev, 3 FN + 1 FP. |
| **Near-zero versus exact-zero** | Both handled. Near-zero recall is **1.000** on the test split (10 events), exact-zero 0.944 (72 events). The explanation wording distinguishes them: "was cut to a trickle … to an average of 405 per day" rather than "stopped … to exactly 0". | Test magnitude and `zero_kind` breakdowns. |
| **Back-to-back events** (a holdout ends, a step begins the next day) | Handled structurally: regime segmentation cuts the timeline wherever the active-channel set changes *or* a P2 change point lands, so adjacent events do not merge. The step pass excludes the drop *into* an off-window but keeps the rise *out* of it, since the rise carries the level change across the pause. | Dedicated tests in `tests/detection/test_level_shift.py`; the sealed split contains a back-to-back edge case. |
| **Overlapping events on one channel** | Both can be reported; the overlap is priced into informativeness through `CONFOUNDED_PENALTY` (0.6) rather than one event being suppressed. | `cleanliness` driver in `detection/score.py`. |
| **Single-channel market** | Dark and holdout are genuinely indistinguishable with one channel, and the benchmark says so. The `single_channel` label requires `len(off) > 1`, so a one-channel market can only produce a dark period. | Test `n_countries = 1` bucket scores 1.000 across the board (8 scenarios). |
| **All-zero channel** (booked but never run) | Excluded everywhere. `channels_in()` filters on spend > 0, so such a channel is neither reported as one giant holdout nor counted as a peer control. A market that never bought a channel **abstains** from the cross-market vote in both directions. | `find_off_runs` returns `[]` on an all-zero series; verified for NaN/inf safety across all five normalize functions. |
| **Naturally intermittent channel** (≈20 alternating 3-day gaps) | Correctly silent. `RUN_RATIO` makes the notability floor `max(7, 3 × p90(other runs)) = 9`, so routine gaps never register — and the injected 42-day holdout in that same series still does. | Boundary tests pin that an 8-day run is not notable and a 10-day run is. This is also failure mode 4. |
| **Gradual ramp** (10-day blocks at 1.2×…2.0×) | No step, no event. Rejected by `SHARPNESS`, which holds with ≈41% margin. | `test_the_benchmarks_own_blocked_ramp_emits_no_step`. |
| **Global pause** (every channel, every market) | Reported as `dark_period` per market with the `global_pause` tag, `control_available: none`, and validity suspicion raised. Indistinguishable from a pipeline outage by spend alone — the detector says so rather than deciding. | `detection/validity.py`; harness maps truth `global_pause` → `dark_period`. |
| **180-day event, longer than any rolling window** | Detected. | Test duration bucket `90+ days`: 35 events, recall 0.943. |
| **Empty media frame** | Returns `[]` with no exception. A filtered export or a market with no bookings yet is a real shape. | `run_detection` guards it explicitly. |

---

## 8. Production-readiness verdict, per detector

Spec §11's going-in expectation was: dark, holdout and single-channel production-ready; step
change production-ready with a ramp caveat; pulse and staggered launch promising; the adstock
half-life estimate research-only. **The measurements disagree in three places.**

| detector | verdict | grade against the expectation |
|---|---|---|
| **dark_period** | **Production-ready.** 1.000/1.000/1.000 on test, 1.000 on dev, zero errors of any kind on either split. | As expected. |
| **channel_pulse** | **Production-ready on this benchmark — but do not deploy without answering the six-window question.** 1.000 across the board on both splits. | **Better than expected on the score, worse in reality.** The expectation was "promising"; the score says perfect. Both readings are wrong for production, because the benchmark's pulse family draws only 2–4 windows and a real flighting channel with six or more regular windows is suppressed **entirely** — and no score in this report can reveal that. A perfect F1 on a family whose defining production case is excluded by construction is the most misleading number in the document. |
| **single_channel** | **Production-ready with a caveat.** 1.000 on test, but 0.824 on dev (R 0.700) — and the gap is the two-channel labelling ambiguity, where test recall is 0.429. | As expected on clean multi-channel markets. Add: **on two-channel markets the label is a coin-flip in the truth itself**, so treat `single_channel` versus `natural_holdout` there as an open question for a human, not an answer. |
| **staggered_launch** | **Production-ready for detection, not for the label.** R 1.000 on test with one FP; 0.952 on dev. Onset dates are exact. | **Better than expected** — "promising" understates it. Two real gaps: a launch is never checked for being *sustained*, and a dormant-then-active market is genuinely ambiguous with a censored holdout. |
| **natural_holdout** | **Production-ready with supervision.** 0.898 on test (P 0.957 / R 0.846), 0.821 on dev. Near-zero recall 1.000. All errors are label substitutions on correctly located windows. | **Slightly below the expectation.** It is the type the `RHO` change rescued (dev recall 0.550 → 0.800) and it remains the type most sensitive to that threshold. `RHO` is a business question on real data, so this detector's real-world accuracy is **the accuracy of that one decision**. |
| **step_change** | **NOT production-ready. Ship it behind a review queue, not into a pipeline.** 0.857 on test (P 1.000 / R 0.750), 0.917 on dev, and 1.000 on the clean `step` family. | **This is the significant disagreement with the expectation.** The expectation was "production-ready with a ramp caveat". The ramp caveat is in fact the *least* of it — sharpness rejects the benchmark's ramp with 41% margin. The real disqualifier is structural: **187 of 198 raw step episodes are open-ended and thrown away unexamined**, a real never-reverting budget change is silently dropped, a step that reverts shortly before a pause loses its closing edge, and behind the drop rule sit 8 uncollapsed market-wide clusters. Test precision of 1.000 is real, but it is achieved by a filter that is also discarding the most common real-world step shape. |
| **adstock half-life estimate** | **Not implemented.** `ADSTOCK_HALF_LIFE = 7.0` is an assumed constant feeding informativeness ranking only; no decay is fitted and no half-life is estimated. | As expected (research-only), but stated plainly: there is nothing here to grade. |
| **confidence calibration** | **Not usable.** Constant 0.943; cannot rank, cannot gate. | Not in the expectation. §4 and §11. |

**Overall:** the *finding* layer is strong — mean IoU 0.991, boundary error 0 days at the
median, zero false positives on event-free data, and a null-scenario rate that separates this
from both degenerate baselines. The *naming* layer is good but has two known ambiguities that
no spend-only rule can settle. The *scoring* layer is the weakest part of the system and the
confidence number should not be relied on at all.

---

## 9. Development trajectory

`benchmark/eval/dev_history.jsonl` holds **23 records**, every one against the 45-scenario
development split. `benchmark/eval/final_runs.jsonl` holds **one**. That asymmetry is the
evidence.

| # | label | source hash | P | R | F1 | null FP |
|---|---|---|---|---|---|---|
| 1 | baseline `perfect_oracle` | absent | 1.000 | 1.000 | 1.000 | 0.000 |
| 2 | baseline `never_detect` | absent | 0.000 | 0.000 | 0.000 | 0.000 |
| 3 | baseline `detect_everything` | absent | 0.000 | 0.000 | 0.000 | **0.654** |
| 4 | plan3-first-run | 3010329f | 0.867 | 0.693 | 0.770 | 0.000 |
| 5 | controller independent verification | 3010329f | 0.867 | 0.693 | 0.770 | 0.000 |
| 6 | after J1–J5 fixes | a1df8279 | 0.867 | 0.693 | 0.770 | 0.000 |
| 7 | RHO sweep 0.05 *(the spec's value)* | a1df8279 | 0.867 | 0.693 | 0.770 | 0.000 |
| 8 | RHO sweep 0.10 | 30db452a | 0.797 | 0.733 | 0.764 | 0.000 |
| 9 | RHO sweep 0.12 | 934ba654 | 0.797 | 0.733 | 0.764 | 0.000 |
| 10 | **RHO sweep 0.15** | a04f8819 | **0.934** | **0.760** | **0.838** | 0.000 |
| 11 | RHO 0.15 + J6 pairing fix | ccbf63b8 | 0.934 | 0.760 | 0.838 | 0.000 |
| 12 | J6 pairing: exclude drop-in, pair zero-origin on sign | 5eeb23b5 | 0.934 | 0.760 | 0.838 | 0.000 |
| 13 | scores wired, uncalibrated | 36c0cb54 | 0.934 | 0.760 | 0.838 | 0.000 |
| 14 | controller verify: curves populated | 36c0cb54 | 0.934 | 0.760 | 0.838 | 0.000 |
| 15 | review verification | 36c0cb54 | 0.934 | 0.760 | 0.838 | 0.000 |
| 16 | task8 baseline | 5472f13e | 0.934 | 0.760 | 0.838 | 0.000 |
| 17 | **closing-shift fix** | f4a0b9a2 | **0.943** | **0.880** | **0.910** | 0.000 |
| 18 | closing-shift fix | c20c1009 | 0.943 | 0.880 | 0.910 | 0.000 |
| 19 | task8 fix round 1 | dfb9ca07 | 0.943 | 0.880 | 0.910 | 0.000 |
| 20 | controller review of Task 8 | dfb9ca07 | 0.943 | 0.880 | 0.910 | 0.000 |
| 21 | before-calibration | 8b141200 | 0.943 | 0.880 | 0.910 | 0.000 |
| 22 | **calibrated** | **aef7282c** | 0.943 | 0.880 | 0.910 | 0.000 |
| 23 | controller: post-calibration curves | aef7282c | 0.943 | 0.880 | 0.910 | 0.000 |

What the trajectory shows:

- **The trivial baselines were run first, before any real algorithm existed**, to confirm the
  metrics behave. `detect_everything` scoring F1 0.000 at a null FP rate of 0.654 is what
  makes the 0.000 in §3 mean something.
- **Exactly two changes moved the score**, and each was driven by a measurement rather than a
  sweep. `RHO` 0.05 → 0.15 (record 7 → 10): F1 0.770 → 0.838. The closing-shift sigma fix
  (record 16 → 17): F1 0.838 → 0.910, driven by step recall 0.154 → 0.846. The intermediate
  `RHO` values of 0.10 and 0.12 were *worse* on F1 than the spec's own 0.05 — the parameter
  has a real optimum at the point where it clears the defined near-zero band, not a monotone
  gradient a tuner would have followed.
- **Twelve of the 23 records are re-verifications at an unchanged score** — independent
  controller runs, reviewer runs, and before/after checks confirming a change was
  score-neutral. That is what a project not chasing the number looks like.
- **The null false-positive rate is 0.000 in every record** from the first real run onward. It
  was never traded away for recall.
- **The frozen hash `aef7282c…` in record 22 is byte-identical to the hash in
  `final_runs.jsonl`.** The code that scored 0.910 on dev is exactly the code that scored
  0.947 on the sealed split. Nothing was changed between calibration and the final run.

Two further checks were recorded during development: the test split's seal verified clean and
`final_runs.jsonl` was confirmed absent immediately before the final run, and the newest file
mtime anywhere in the test split is the sealing moment itself — nothing in it was touched
during any development task.

---

## 10. Recommendations for the real Sellforte dataset

Spec §11's five points, with what this project measured about each.

**1. Missing-row versus zero-spend is the largest single risk.** An export that omits rows
instead of writing zeros looks like a perfect holdout, and after reindexing onto a date grid
the two are indistinguishable. The machinery for this exists and works: `build_panel` counts
presence **before** any fill, and the validity gate raises `suspect_data_gap` when a window's
rows are missing rather than zero. But which convention the real export uses is a question
about Sellforte's pipeline, not about this code. **Do this first**, before trusting any
output: take a known-paused channel-market and confirm whether its rows exist with
`media_investment = 0` or do not exist at all. Everything downstream depends on the answer.

**2. Aggregate campaigns to channel level per country-day before detection.** `media.csv` is
campaign-grained and **a campaign ending is not a channel holdout**. `build_panel` already
sums `media_investment` over campaigns by `(date, country_code, advertising_channel)`, so this
is handled — but it is handled by summing whatever the `advertising_channel` column says.
Confirm that column is the channel taxonomy the MMM actually uses, and that no channel is
split across two spellings. On the synthetic data this aggregation is a no-op, so it is
**untested against real campaign grain**.

**3. Holiday calendars produce all-channel pauses that look like dark periods.** A Christmas
shutdown is a real dark period and also a seasonal closure, and the difference matters for
MMM. Two checks separate them, and both are implemented: the **cross-market check** — a
genuine marketing decision rarely lands in every market on the same day, and a `global_pause`
tag with `control_available: none` means there is no control group and the event may be an
outage rather than a decision — and the **sales response**: a marketing pause with sales
continuing normally is a different thing from a closure. Supply a holiday calendar per market
and exclude or flag those windows before triage; the detector has no calendar and cannot do
this for you.

**4. There is no ground truth on real data.** Validate three ways: **(a)** cross-market
consistency — a detected holdout whose peers kept running is self-corroborating, and the
`control_available` field already reports it; **(b)** have Sellforte analysts confirm a
sampled top-N, using the `explanation` field, which is written so that every claim in it is
checkable against the export (typical daily level, window average, run ranking, peer count,
sibling-channel status); **(c)** check that detected dark periods do in fact show sales
settling to a readable baseline. Note that **spec §8's fourth validity trigger — a spend drop
with no sales response — is not implemented** (§11), so (c) is currently a manual check.

**5. Do not run at a confidence cut. Triage top-N by informativeness.** Spec §11 asks for an
operating cut chosen from the operating curve, and **this report cannot give you one**: the
calibrated confidence is a constant and the raw score's curve is anti-correlated with
correctness (§4). The usable procedure instead:

- Run the detector. At benchmark density expect roughly 1.7 events per scenario (92
  detections across 55 test scenarios).
- **Drop nothing on confidence.** Record it as "these are ~94% right in aggregate".
- **Filter on `validity` first** — anything not `ok` is a data question, not a marketing
  finding, and belongs with the pipeline owner rather than the analyst.
- **Sort by `informativeness`** and review the top N. That score still discriminates: it ranks
  a long, clean, exact-zero holdout with live peer markets above a short reduced-spend step in
  a noisy market, which is the right order for an analyst choosing what to look at.
- **Prefer events with `control_available: peers`.** Those are the cross-market holdouts — the
  most MMM-valuable finding the system produces, because the peers are a ready-made control
  group.
- **Refit the calibration on your own data** once you have a few hundred confirmed events.
  `scripts/fit_calibration.py` fits on `evidence["raw_confidence"]` and is idempotent by
  construction, so it cannot accidentally calibrate an already-calibrated score. With a real
  sample there may be enough signal to re-weight the sub-scores against correctness, which is
  the thing this project deliberately did not attempt on 70 detections.

---

## 11. What this benchmark cannot tell you

The most important section in this document. Everything above is a measurement; this is the
list of things the measurements do not cover.

### The edge family's event locations are partly disclosed by the dev split

The edge-case scenarios are built from **hardcoded offsets** — `holdout(c0, ch0, 0, 60)`,
`holdout(c0, ch0, n_days-60, 60)`, `holdout(…, 200, 42) + step(…, 242, 56)`, a ramp at day 200,
a global pause at day 300 — and they always target `countries[0]` and `channels[0..1]`. Only
country codes, channel names and the noise draw differ between splits. **Ten of the 55 test
scenarios therefore have event locations that the development split already discloses.**

This was not exploited — no detector code references a scenario id, a family, or a date, and
`tests/eval/test_gating.py` statically bans `benchmark.eval`, `benchmark.spec` (the event
builder, i.e. the answer key itself), `benchmark.harness`, `_truth` and `ground_truth` from
every file under `detection/` — but it caps what "unseen" means for that family. The edge family is also the weakest
family on test (F1 0.750), so the disclosure did not help; that is evidence, not a defence.

### `RHO` was changed from the spec's value, on development measurement

The spec's parameter table set `RHO = 0.05`, which is **exactly the top of the near-zero band
the spec itself defines** (a near-zero event is 0.02× to 0.08× of normal spend). A threshold
sitting inside the band it has to classify decides those days by the noise realisation rather
than by the event. It was raised to 0.15 — roughly double the band's upper bound — which is a
domain-derived bound, not a fit to individual scenarios. The measurement is in
`dev_history.jsonl` records 7–10 and in the §5 table.

That said: **this is the one parameter changed on the basis of the development split**, and
the near-zero recall of 1.000 on the test split is downstream of it. On real data "what counts
as spend paused versus spend low" is a business question, and the number should be re-derived
from Sellforte's own definition rather than inherited from here.

### The calibration is constant, so confidence cannot rank or gate

`KNOTS = [(1.0, 0.9428571428571428)]`. Every event scores 0.943. PAV collapsed the whole
range into one block because confidence discriminates by **event type** rather than by
correctness, and on dev it is mildly **anti-correlated** with it: every false positive sits in
the high-confidence group, while the two lowest-confidence types (`channel_pulse` median 0.640,
`step_change` median 0.710) are perfectly precise. The raw operating curve loses precision
*and* recall as the cut rises. Full numbers in §4.

The honest statement is: **"we are right about 94% of the time and cannot tell you which
ones."** Spec §9 item 9 exists to give the handover a precision/recall trade-off; this
handover does not have one to give.

### The step pass is held together by a rule that must be relaxed for production

On the development split, the raw step pass produces **187 open-ended episodes against 11
bounded ones**. The only thing keeping those 187 out of the output is the rule that drops any
episode without a matching reversal. Seventeen unpairable shifts for every paired one.

That rule **must be relaxed for production** — a real budget change that never reverts is
currently dropped, and real budgets change without reverting all the time. But relaxing it
requires **collapsing market-wide moves first, not afterwards**: behind the drop rule sit
**8 same-day clusters of up to nine channels** (dev_035/FR 9, dev_033/FI 8, dev_035/AT 8,
dev_030/DE 7, dev_032/PL 7, dev_033/CH 7, dev_034/FI 7, dev_029/US 5). Every one is currently
open-ended and therefore already discarded. Relax the drop rule without a market-wide collapse
rule and those return as *closed* episodes, and step precision — 1.000 on the test split —
collapses. The order of operations is not negotiable.

A market-wide collapse rule was deliberately **not** built: with zero such groups surviving to
the output it could not have been verified by any behavioural test, only by a value assertion,
and this branch twice removed unreachable gates for exactly that reason. The consequence is
quantified above rather than predicted.

### A regular pulse train of six or more windows is suppressed entirely

At six windows the `RUN_RATIO` notability rule switches on and every run's peers are its own
length, so no run clears `3 × p90(other runs)` and **nothing is reported at all**. The code is
faithful to spec §7's P1 definition; **the conflict is in the spec**, not in the
implementation.

The benchmark's pulse family draws 2–4 windows, so no score in this report can reveal it —
`channel_pulse` reads a perfect 1.000 on both splits. **On real flighting data, six or more
regular windows is the normal case.** This is the **highest-priority spec question before
production use**: the type with the best score in the benchmark may be the type that fails
most often in production, and nothing here would have caught it.

### Two dev scenarios are the same data with different truth labels

`dev_036` (truth `natural_holdout`, edge case `censored_start`) and `dev_022` (truth
`staggered_launch`) are **the same data** — 2 markets, 4 channels, 730 days, one market dormant
for exactly 60 days, no ramp-up in either. No spend-only rule separates them.

Separately, **two-channel markets are label-ambiguous in the truth itself**: the identical
spend shape is `single_channel` in `dev_008/009/010` and `natural_holdout` in `dev_011/013`,
depending only on which scenario family drew it. Six events on dev; test recall 0.429 in the
`n_channels = 2` bucket. The detector's rule here is a **prior, not a reading**, and it was
deliberately not fitted to those scenarios — fitting it would have been fitting to three dev
scenarios and would not have generalised.

### A step change that reverts shortly before the channel pauses is dropped

`W`-wide non-maximum suppression: the drop into the off-window is a much larger delta and
suppresses the step's closing shift inside the 21-day neighbourhood, so the episode never
closes and is dropped as open-ended. Reproduced deliberately: the bounded episode is lost
whenever the gap is **≤ 8 days**. It is not caused by the off-window exclusion — with
`exclude=[]` the loss is identical.

**The benchmark never produces that shape.** Truth assigns holdouts, steps and pulses to
different channels within a country, so a same-channel step-then-holdout never arises. Real
marketing data produces it routinely: cut the budget, restore it, then pause the channel.

### Spec §8's fourth validity trigger is not implemented

The validity gate implements three of spec §8's four triggers: rows missing rather than zero
(`suspect_data_gap`), spend at zero while impressions continue (`suspect_tracking_loss`), and
every channel in every market off at once. **The fourth — a spend drop with no sales response
in a window where sales SNR was adequate to show one — is not implemented.**

It cannot be validated here: in this benchmark **sales are generated from the same spend
series the detector reads**, so the check would be scoring the detector against its own answer
key. Building a sales-SNR estimator to serve a single unvalidatable flag was scope this
project declined rather than shipping on faith. It is recorded as not-implemented rather than
quietly omitted. On real data it is a genuine and worthwhile check, and it is the missing half
of recommendation 10(c).

(The related `sales_snr` **informativeness** driver *is* implemented, and it is not circular:
it asks only whether this market's turnover is quiet enough for *any* response to be readable
— a noise measurement — never whether the spend change caused a sales change.)

### Review provenance: two of twelve tasks were reviewed by the controller

Of the twelve tasks in the final plan, ten received an independent review by a separate agent.
**Two did not** — Task 4 (the validity gate) and Task 8 (the closing-shift sigma fix, the
largest single behavioural change in the project) — because their assigned reviewers died to
session rate limits. Both were reviewed inline by the controller that wrote the briefs, which
is weaker provenance: the same party wrote the specification, fixed the defect, and judged the
fix.

Both reviews were run against real data rather than by inspection — Task 8's included
reproducing the `|z|` notch on `dev_016` SE/Affiliate, where the pooled sigma gives 1.52–2.71
at the closing edge, under the 3.5 threshold, while the two-sided form gives 9.28 — and Task
8's result is corroborated by the sealed split, where step change scores P 1.000. But if you
are deciding how much of this to re-verify yourself, **start with `detection/validity.py` and
`detection/primitives/level_shift.py`.**

### Two other things worth knowing

- **`ADSTOCK_HALF_LIFE = 7.0` is assumed, not measured.** It feeds informativeness ranking
  only and never detection, so it cannot cause a false positive — but the "duration adequacy"
  driver is scored against a number nobody measured. Spec §7's post-pulse adstock decay fit is
  not implemented at all.
- **The test suite has two known failures**, both in `tests/eval/test_gating.py` (lines 182 and
  266), and both assert that `benchmark/eval/final_runs.jsonl` **does not exist** — a
  precondition that held throughout development and became false the moment the sanctioned
  final run was made. They are stale test scaffolding, not a detector defect: 579 of 581 tests
  pass and no detection test is affected.

---

## Appendix — where things live

| path | what |
|---|---|
| `detection/pipeline.py` | `run_detection(media_df, sales_df, sid) -> list[DetectedEvent]` — the entry point |
| `detection/params.py` | every threshold, each with its documented failure mode |
| `detection/primitives/` | P1 `zero_runs`, P2 `level_shift`, P3 `pulse`, P4 `onset` |
| `detection/compose/` | `label.py` (regime segmentation), `cross_market.py` (peers, launches) |
| `detection/score.py` | confidence sub-scores and informativeness drivers |
| `detection/calibrate.py`, `detection/calibration_fit.py` | the PAV fit; the frozen knots |
| `detection/validity.py`, `detection/explain.py` | the validity gate and the prose |
| `benchmark/BENCHMARK.md` | benchmark composition, confounds, loader traps |
| `benchmark/eval/dev_history.jsonl` | 23 development runs |
| `benchmark/eval/final_runs.jsonl` | **one** line — the single sealed-split run |
| `REPORT_final_metrics.md` | the raw harness output for that run; the authority for every test-split number here |

Module docstrings throughout `detection/` carry the measured findings behind each design
decision, including the ones that contradict the spec and why.
