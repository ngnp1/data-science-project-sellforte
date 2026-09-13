# Detecting Informative Periods in Marketing Data — Design

**Project:** Group 4 Sellforte / Aalto
**Date:** 2026-09-13
**Status:** Approved design, ready for implementation planning

---

## 1. Problem

Marketing mix modelling is hard because channels move together. Buried in the data
are short periods that behave like accidental controlled experiments: everything
goes dark, one channel stops, only one channel runs, a budget jumps. Those periods
carry disproportionate information. Analysts find them by eye, inconsistently.

Build a system that scans a marketing dataset, finds these periods automatically,
labels them, ranks them by how informative they are, and explains each finding in
terms a non-statistician can interrogate.

Required event types: dark period, single-channel period, natural holdout, step
change. Additional valuable patterns are in scope: on/off/on pulses, staggered
regional launches, cross-market holdouts.

**The governing constraint:** algorithms are developed without access to the
ground-truth event locations in the final benchmark, so the reported performance
reflects genuine generalization rather than fitting to synthetic test cases.

### Out of scope

The Streamlit UI from the PDF brief. The event output schema is designed to feed
one — ranked rows, explanation string, machine-readable evidence dict — but
building it is a separate piece of work.

---

## 2. Existing assets

`synthetic_data_generator/` holds a working two-stage generator:

1. `generate_with_simmmulator.R` — drives Meta's siMMMulator through baseline
   sales, ad spend, media, CVR, adstock decay, Hill saturation, conversions.
   Injects events into step-2 spend, exploiting the fact that with
   `frequency_of_campaigns = 1` siMMMulator's `campaign_id` equals the 1-indexed
   day number.
2. `reformat.py` — reshapes the wide per-country output into the Sellforte
   `media.csv` / `sales.csv` schema and writes `ground_truth.csv`, `true_roi.csv`.

Both read `config.yaml` and `events_config.yaml`. R with `siMMMulator`, `dplyr`,
`yaml` is installed; `synthetic_data_generator/.venv` has pandas 3.0.5 and numpy
2.5.2 only.

### Data schema

`media.csv`: `date, ad_platform, advertising_channel, campaign_name, campaign_id,
media_investment, clicks, impressions, conversions, conversion_value,
country_code` — campaign-grained, one row per campaign per day.

`sales.csv`: `country, turnover, customer_type, country_code, granted_discounts,
sales_channel, date` — daily turnover split across customer type × sales channel.

`ground_truth.csv`: `pattern_id, pattern_type, country_code, channel, start_date,
end_date, multiplier, description`.

### Measured constraints

- **Runtime:** a 5-country × 2-year run takes 252 s, roughly 50 s per
  country-year-pair. The machine has 8 cores (6 performance).
- **Fixed seed:** `set.seed(42)` is hardcoded, and both scripts read config from
  and write output to the working directory. Generating varied scenarios requires
  parametrizing the generator.
- **Noise knobs exist but are coarse:** `baseline.error_std`, `trend_p`,
  `temp_var`, `temp_coef_sd`, and per-channel `std_noisy_cpm_cpc` /
  `std_noisy_cvr`. Sufficient for the variation the benchmark needs. Near-zero
  spend is expressible through a small `multiplier`.

### Two generator quirks the evaluation loader must handle

1. **`ground_truth.csv` end_date is off by one.** `reformat.py` computes
   `end_date = dates[end_day]`, but `end_day` is slice-exclusive, so the recorded
   end is the day *after* the last affected day. `DE_DARK_01` records
   `2024-08-18` while the README's own table gives the window as ending
   `2024-08-17`. Left unhandled this biases every IoU and every boundary-error
   statistic by one day. The eval loader normalizes ground truth to an inclusive
   `[start, end]` interval.
2. **`channel` means opposite things by type.** For `single_channel` it names the
   channel that stays **on**; for `natural_holdout`, `channel_pulse` and
   `staggered_launch` it names the channel that goes **off**. Matching logic must
   respect the difference.

---

## 3. Architecture

```
detection/                      the deliverable; never imports from benchmark/
  io/panel.py                   load media/sales -> daily panel + presence mask
  io/normalize.py               market-scale estimation, robust scaling
  primitives/zero_runs.py       P1: off-runs per country x channel
  primitives/level_shift.py     P2: robust change-points + persistence
  primitives/pulse.py           P3: repeated off-runs
  primitives/onset.py           P4: series-start off-run -> launch candidate
  compose/label.py              regime segmentation -> event labels
  compose/cross_market.py       peer-market comparison, control availability
  score/confidence.py           detection certainty, named sub-scores
  score/informativeness.py      MMM usefulness, sales-driven
  score/validity.py             data-gap discrimination
  score/explain.py              templated natural-language rationale
  pipeline.py                   run_detection(media_df, sales_df) -> events
  params.py                     every threshold, one file, documented

benchmark/
  spec/scenarios.py             100 scenario definitions, seeded
  harness/generate.py           config synthesis + parallel Rscript runner
  datasets/dev/<sid>/           media.csv, sales.csv
  datasets/dev_truth/<sid>/     ground_truth.csv, meta.json
  datasets/test/<sid>/          media.csv, sales.csv
  datasets/test_truth/<sid>/    ground_truth.csv, meta.json   SEALED
  eval/match.py                 interval matching
  eval/metrics.py               P/R/F1, IoU, boundary error, accuracies
  eval/run_dev.py               free to run
  eval/run_final.py             gated, append-only audit record
  eval/dev_history.jsonl        appended on every dev evaluation
  eval/final_runs.jsonl         append-only, one line per final run
```

### Generator changes

Add `--seed`, `--config`, `--events`, `--outdir` arguments to
`generate_with_simmmulator.R` and `reformat.py`, each defaulting to current
behaviour so the documented `Rscript generate_with_simmmulator.R` invocation
continues to work unchanged. No other modification to the generator.

### Dependencies

Pure pandas and numpy. No scipy, sklearn or ruptures. The `.venv` has none of
them, and everything required — robust z, MAD, binary segmentation, PAV
calibration, greedy interval matching — is a few dozen lines each. A reviewer
being able to read the change-point cost function is worth more here than a
library import, given the interpretability requirement.

---

## 4. Black-box seal

The black-box claim needs to be auditable, not asserted. Four layers:

1. **Physical separation.** Ground truth never lives inside a dataset directory.
   `detection/` has no import path to any truth loader; only `benchmark/eval/`
   can read `*_truth/`.
2. **Sealing.** After generation, `test/` and `test_truth/` receive a
   `manifest.sha256` covering every file, plus a `SEALED` marker recording the
   timestamp and the scenario-spec hash. Regeneration or editing changes the
   manifest and is detectable.
3. **Gated final run.** `run_final.py` requires an explicit `--finalize` flag,
   verifies the seal, and appends to `final_runs.jsonl`: timestamp, SHA-256 of
   the whole `detection/` tree, dataset manifest hash, resulting metrics. If
   `detection/` changes after a final run, the next final report carries a banner
   saying so. A second final run is possible but never silent.
4. **Disjointness.** Dev draws seeds 1000–1999, test 5000–5999; the spec asserts
   no scenario id appears in both.

Generation parameters are hidden as thoroughly as event locations: dataset
directories contain **only `media.csv` and `sales.csv`**. The `meta.json`
recording noise level, trend, channel count and market spread — needed for
evaluation breakdowns — sits on the truth side.

---

## 5. Data model

Panel keyed `(country_code, advertising_channel, date)`, reindexed onto a
complete date grid per series:

| field | purpose |
|---|---|
| `spend` | primary signal, aggregated from campaign level to channel level |
| `present` | did a row exist, or was it reindexed in? Always true in synthetic data; **not** in the real dataset, where missing-row versus zero-spend is the largest single failure mode |
| `clicks`, `impressions`, `conversions` | corroboration — spend zero *and* impressions zero is a real pause; spend zero with impressions still flowing is a billing or tracking artifact |

Sales aggregated to daily turnover per country, summed over customer type and
sales channel.

### Normalization

Three scales, used in different places. Raw EUR is never compared across markets.

- **Within-series:** spend ÷ trailing median of that series' active days, giving a
  scale-free index. A Finnish holdout and a US holdout then look identical.
- **Within-country:** each channel's share of country daily total, so composition
  changes (single-channel periods) are detected independently of swings in the
  total budget.
- **Cross-market:** spend ÷ market-scale proxy, where the proxy is the long-run
  median of that country's total spend. The current config spans 6.5× between US
  (1.30) and FI (0.20); the benchmark pushes this to 15×.

---

## 6. Benchmark

100 scenarios: 45 development, 55 hold-out test. A scenario is a
`(config.yaml, events_config.yaml, seed)` triple, built programmatically by
`benchmark/spec/scenarios.py` from a seeded RNG. Runtime is held near 45 minutes
wall-clock at 6 workers by keeping the mean at 3 countries × 2 years
(≈150 core-seconds); 8-country scenarios drop to 1 year to stay in budget.

### Families

| family | dev | test | purpose |
|---|---|---|---|
| null — no events at all | 4 | 5 | **false-positive rate**; precision is unmeasurable without these |
| dark only | 3 | 4 | also exercised inside the mixed scenarios |
| single-channel only | 3 | 4 | also exercised inside the mixed scenarios |
| natural holdout only | 3 | 4 | also exercised inside the mixed scenarios |
| step change only | 5 | 6 | up 3× and 5×, down 0.33×, mild 1.5× |
| pulse only | 3 | 4 | 2–4 off-windows, varying spacing |
| staggered launch | 3 | 4 | onset offsets differ per market |
| cross-market holdout | 3 | 4 | channel off in one market, live in peers |
| mixed, 3–5 simultaneous | 8 | 10 | the realistic case |
| edge cases | 10 | 10 | one scenario per listed case, below |
| **total** | **45** | **55** | |

Within every family, noise, trend, seasonality, duration and magnitude are
stratified rather than sampled independently, so each family spans all three
noise levels instead of clustering by chance.

### Variation axes

- **noise:** low / medium / high — scales `baseline.error_std`, `temp_coef_sd`,
  and per-channel `std_noisy_cpm_cpc` / `std_noisy_cvr`
- **trend:** `trend_p` ∈ {0, 0.5, 1.0}
- **seasonality:** `temp_var` ∈ {0.5, 2, 5}
- **channels:** 1, 2, 4, 6, 9, 12 — synthesised by sampling and rescaling
  `spend_share_min` / `spend_share_max` so shares stay valid and the last channel
  still takes the remainder, per the generator's contract
- **countries:** 1, 2, 3, 5, 8
- **market spread:** tight (all ≈1.0), moderate (0.20–1.30, current), extreme
  (0.10–1.50, i.e. 15×)
- **duration:** 5, 14, 42, 90, 180 days
- **magnitude:** exact 0, near-zero (0.02–0.08×), 0.33×, 1.5×, 3×, 5×

### Edge cases

One scenario each on both splits:

- event starting at day 0 (censored start), and one running to the final day
  (censored end)
- back-to-back events — a holdout ends and a step begins the next day
- two overlapping events on the same channel
- a 5-day event, below any sane detectability floor — tests whether the system
  reports an honest false negative rather than chasing it
- a 180-day event, longer than any rolling window
- a 1-channel market, where dark and holdout are genuinely indistinguishable
- **naturally intermittent channel** — roughly 20 alternating 3-day off-windows,
  so short zero-runs are normal for that series. This scenario punishes any fixed
  "7 days off means holdout" rule and forces the distribution-relative logic in
  section 7
- **gradual ramp, not a step** — consecutive 10-day blocks at 1.2×, 1.4×, 1.6×,
  1.8×, 2.0×. Ground truth records *no* step change; a detector that fires here
  is wrong
- **global pause** — every channel in every country, same window. Real, but with
  no control group, and indistinguishable by spend alone from a data outage

---

## 7. Detectors

All detection is rolling medians, MAD-scaled robust z, and run-length logic on a
log-ratio series. No parametric model and no learned thresholds, for two reasons:
every intermediate number must appear verbatim in an event's explanation, and a
Gaussian z on raw EUR spend breaks on both the 15× market spread and the heavy
right tail of daily spend. A robust z on `log1p(spend / series_level)` is
scale-free, tolerant of multiplicative noise, and prints legibly.

### Preprocessing, per series (country × channel)

- `L_s` = median of spend over days with spend > 0 — the series **active level**.
  Robust while off-days stay under roughly half the series.
- `y_t = log1p(spend_t / L_s)` — the scale-free working series.
- Level work runs on a 7-day centred rolling median to remove day-of-week
  structure. Run detection stays on raw daily values so boundaries land on exact
  dates.

### P1 — off-runs

```
off_t = spend_t <= max(EPS_ABS, RHO * L_s)          RHO = 0.05
```

Each off-day is classified `exact_zero`, `near_zero`, or `missing` (from the
presence mask). Maximal runs of `off_t` are candidates.

Notability is judged **relative to the series' own history**:

```
notable(r)  <=>  len(r) >= MIN_DAYS (7)
            AND  (len(r) >= 3 * p90(other off-runs in this series)
                  if at least 5 other runs exist)
```

A flighting channel whose normal gaps are 3 days needs at least 9 off-days to
register; a channel that is otherwise never off needs only 7. Any single global
threshold fails one of those two cases whichever value is chosen.

Evidence recorded per run: length, **depth** (`1 - mean(spend in run) / L_s`), and
**edge sharpness** (median spend in the 7 days either side, ÷ `L_s`).

### P2 — level shifts

At each candidate boundary `t`, with window `W = 21` (a multiple of 7):

```
delta_t = median(y[t : t+W]) - median(y[t-W : t])
sigma_t = 1.4826 * MAD(y over the union of both windows)
z_t     = delta_t / max(sigma_t, SIGMA_FLOOR)
```

Candidates are local maxima of `|z_t|` with `|z_t| >= 3.5`, separated by at least
`W`. Two filters then apply:

- **Persistence** — the new level must still hold 14 days later, at no less than
  half the original `delta`. This rejects spikes.
- **Sharpness** — `|median(y[t:t+3]) - median(y[t-3:t])| / |delta_t| >= 0.6`.
  This is the gradual-ramp defence: a genuine step concentrates its change into
  2–3 days (sharpness near 1), while a 50-day ramp spreads it out (sharpness near
  0.1) and is rejected. Persistence alone does **not** reject a ramp, because a
  ramp's new level genuinely does hold.

Boundaries are then refined by binary segmentation under the same robust cost.
Reported magnitude is `median(spend after) / median(spend before)` in original
units, so a tripling prints as "3.02×", directly comparable to the ground-truth
multiplier.

Opposite-sign change-point pairs with comparable `|delta|` form a bounded step
episode. An unpaired change-point is an open-ended step running to series end.

### P3 — pulses

Two or more notable P1 runs on one series, with similar lengths
(IQR / median ≤ 0.5), each bounded by active spend. Emitted as one grouped event
spanning first start to last end, with the individual windows attached as
`components`. Carries the highest informativeness weight: per the brief, on/off/on
is the only place where adstock decay is observable. Where sales SNR allows, a
post-pulse decay is additionally fitted and an **estimated adstock half-life**
reported, explicitly flagged as a secondary, lower-confidence output.

### P4 — onsets

A P1 run anchored at series start with sustained activity afterwards. On its own
this is only a `censored_holdout`; it becomes `staggered_launch` only with
cross-market confirmation. Symmetrically, a run anchored at series end is
`channel_discontinued`.

### Composition — regime segmentation

Not interval clustering, which would merge unrelated concurrent events. Instead,
per country, compute the **active-channel set `A_t`** for each day, then cut the
timeline wherever `A_t` changes or a P2 change-point lands. Each maximal run of
constant `A_t` is a regime, and regimes are labelled directly, where `K` is the
set of channels ever active in that country:

| condition | label |
|---|---|
| `A_t = {}` | **dark_period** |
| `\|A_t\| = 1`, at least 2 channels exist, and the off channels were active in adjacent regimes | **single_channel** (channel = the live one) |
| `0 < \|K \ A_t\| < \|K\|` | **natural_holdout** per off-channel; grouped as **multi_channel_holdout** when 2 or more share boundaries |
| `A_t = K` and a P2 episode spans the regime | **step_change** |

This resolves the nesting of event types structurally rather than by precedence
rules: a dark period simply *is* the regime where `A_t` is empty, and its
constituent per-channel holdouts are attached as `components` instead of being
emitted as competing events. Regimes shorter than `MIN_DAYS`, and regimes whose
off-channels were not genuinely active both before and after, are rejected.

### Cross-market layer

For each event `(country, channel, [start, end])`, the same channel is checked in
every peer country over the same window, on the market-scale-normalized series:

- **peers normal** — `control_available: peers`. A holdout in this situation is
  additionally tagged **cross_market_holdout**, the most MMM-valuable finding the
  system can produce, since the peers form a ready-made control group.
- **all peers also off** — `global_pause`, `control_available: none`, and validity
  suspicion raised. This is what a pipeline outage looks like.
- **some peers off** — partial control.

Separately, onset dates for each channel are collected across countries; a spread
greater than `ONSET_SPREAD` days emits a panel-level **staggered_launch** event
listing each market's onset date. Within a country, sibling channels that stayed
normal during a holdout are recorded as `control_available: sibling_channels`.

### Parameters

All in `detection/params.py`.

| param | default | governs | risk if wrong |
|---|---|---|---|
| `RHO` | 0.05 | near-zero threshold | too low misses near-zero events; too high reads low-spend days as off |
| `EPS_ABS` | 1e-6 | float-noise floor | |
| `MIN_DAYS` | 7 | shortest reportable event | at 7 the 5-day edge case is deliberately undetectable and should surface as an honest false negative; lowering it to catch that case floods the output with noise |
| `RUN_RATIO` | 3 × p90 | intermittent-channel guard | the main false-positive lever on flighting channels |
| `SIGMA_FLOOR` | 0.05 | lower bound on the MAD scale, so a near-constant window cannot yield an unbounded z | too low makes flat series fire spuriously |
| `W` | 21 | level-shift window | shorter gives a noisier z; longer misses short steps |
| `Z_THRESH` | 3.5 | step sensitivity | dominant precision/recall trade-off for steps |
| `PERSIST` | 14 | spike rejection | |
| `SHARPNESS` | 0.6 | ramp rejection | lower turns ramps into false steps |
| `ONSET_SPREAD` | 14 | staggered-launch trigger | |

Every default is set from first principles — weekly multiples, a conventional
3.5σ robust cut — and then adjusted **only against the development split**, with
each change logged to `dev_history.jsonl`.

---

## 8. Scoring and explanation

Three scores, deliberately kept separate. Collapsing them is the standard
mistake: "how sure am I this is real" and "how useful is it" are different
questions, and a genuine but uninformative global pause should score high on the
first and low on the second.

### `detection_confidence` in [0, 1]

A weighted mean of named sub-scores, all printed alongside the result:

| sub-score | source |
|---|---|
| `magnitude_evidence` | run depth, or `min(1, \|z\| / 8)` for steps |
| `duration_evidence` | `min(1, len / (2 * MIN_DAYS))` |
| `distinctiveness` | run length against that series' own p90 off-run |
| `edge_sharpness` | transition concentration |
| `corroboration` | spend 0 **and** impressions 0 scores 1.0; spend 0 with impressions still flowing scores 0.2 |
| `consistency` | for country-level events, the fraction of channels agreeing |

Weights differ by event type and are stated in `params.py`.

The score is then **calibrated on the development split only**: confidence is
binned into deciles, empirical precision measured per bin, a monotone
pool-adjacent-violators mapping fitted, and that mapping frozen before the final
run. This turns the number into a testable claim — *events at confidence 0.9 are
correct about 90% of the time on data the algorithm has never seen* — which is
the most useful single line in the handover.

### `informativeness` in [0, 1]

Drives ranking, which the brief explicitly asks for. Drivers: duration adequacy
against assumed adstock half-life, contrast, cleanliness (penalty when another
event overlaps and confounds the window), control availability (peers 1.0,
sibling channels 0.7, none 0.3), sales signal-to-noise within the window, a
penalty for censoring at a series edge, and a type prior — dark periods (baseline
readable directly), single-channel periods (unambiguous attribution) and pulses
(the only place adstock is visible) rank above a plain step change.

### `validity`

`ok`, `suspect_data_gap`, or `suspect_tracking_loss`, with the triggering reasons
listed. Raised by: rows missing rather than zero; spend at zero while impressions
continue; every channel in every country off at once; or a spend drop with no
sales response in a window where sales SNR was adequate to show one.

### Explanation

Every event carries a templated natural-language `explanation` plus a
machine-readable `evidence` dict holding the numbers behind each sub-score. The
target register:

> Facebook spend in AT fell from a typical €1,240/day to exactly €0 for 42
> consecutive days (2025-02-04 to 2025-03-17) — the longest off-run in this
> series by a factor of 21, the next longest being 2 days. All five other AT
> channels ran at normal levels throughout, and Facebook stayed active in DE, CH,
> US and FI, so those four markets are available as controls. AT turnover fell
> 6.2% against its trailing 8-week median (2.1σ). Label: natural_holdout
> (cross-market). Confidence 0.94, informativeness 0.81, validity ok.

### Output schema

One row per event: `event_id, type, country_code, channel, start_date, end_date,
n_days, detection_confidence, informativeness, validity, magnitude_ratio,
control_available, components, evidence, explanation, censored_start,
censored_end`.

---

## 9. Evaluation

Detected events are matched to ground truth greedily by descending temporal IoU,
one-to-one, requiring the same country, type and channel, with IoU at least 0.5.

Metrics reported:

1. Event-level **precision, recall, F1** — overall and per event type
2. Mean and median **temporal IoU** over matched pairs
3. **Boundary error** — median and p90 of `|Δstart|` and `|Δend|` in days
4. **Channel accuracy** and **market accuracy**, measured under a *relaxed* match
   (IoU ≥ 0.5, ignoring type and channel). Under the strict match these are
   degenerate: a channel mix-up disappears into a false negative plus a false
   positive and the accuracy reads 100%
5. **Day-level segmentation P/R/F1** as a secondary view, robust to the
   split/merge disagreements that make event counting brittle
6. **Type confusion matrix**
7. **False-positive rate on the null scenarios**, in events per country-year —
   the cleanest headline number, and unobtainable without those 9 scenarios
8. **Reliability curve** — confidence decile against empirical precision
9. **Operating curve** — confidence cut swept across its range, precision and
   recall plotted, so the handover can state a trade-off rather than one point
10. **Breakdowns** from the truth-side `meta.json`: noise level, duration,
    magnitude, channel count, market spread, mixed versus single-event, near-zero
    versus exact-zero

Both single-event and mixed-event scenarios are reported separately throughout.

---

## 10. Iteration protocol

`develop → run_dev.py → refine`. Every dev run appends the detector source hash,
the params hash and the resulting metrics to `dev_history.jsonl`. That file goes
into the report as the **development trajectory** — evidence that tuning happened
where it was permitted.

`run_final.py` refuses to run without `--finalize`, verifies that the `test/` and
`test_truth/` manifests still match their seal, and appends an audit record to
`final_runs.jsonl`. It is run once, after the algorithms are considered final. A
second run remains possible but is permanently banner-flagged in the report.

---

## 11. Handover

`REPORT.md` covers:

- final algorithm design per event type
- benchmark results on the unseen test set, with the breakdowns from section 9
- important parameters and their sensitivity
- failure modes and edge-case behaviour
- reliability curve, operating curve, development trajectory
- per-detector production-readiness verdict, graded honestly. The expectation
  going in: dark, holdout and single-channel production-ready; step change
  production-ready with a ramp caveat; pulse and staggered launch promising; the
  adstock half-life estimate research-only
- recommended end-to-end pipeline

### Recommendations for the real Sellforte dataset

Driven by what the synthetic benchmark cannot exercise:

- **Missing-row versus zero-spend** is the largest risk. The presence mask and
  the validity gate exist for it, but the real data will need a pass to confirm
  which convention the export uses.
- **Aggregate campaigns to channel level** per country-day before detection.
  `media.csv` is campaign-grained, and a campaign ending is not a channel
  holdout.
- **Holiday calendars** produce all-channel pauses that look like dark periods.
  The cross-market check and the sales-response check separate a genuine
  marketing pause from a seasonal closure.
- **No ground truth exists** on real data. Validate by cross-market consistency,
  by having Sellforte analysts confirm a sampled top-N, and by checking that
  detected dark periods do in fact show sales settling to a readable baseline.
- Run at the confidence cut chosen from the operating curve, then triage the top
  N by informativeness rather than attempting exhaustive review.

---

## 12. Build order

1. Patch the generator for `--seed` / `--config` / `--events` / `--outdir`,
   preserving current defaults.
2. Build `benchmark/spec/scenarios.py` and `benchmark/harness/generate.py`.
3. Generate all 100 scenarios, seal the test split, verify manifests.
4. Build `benchmark/eval/` — matching, metrics, the dev and final runners —
   against the dev split with a deliberately trivial detector, to confirm the
   metrics behave before any real algorithm exists.
5. Build `detection/io/` and the primitives, then composition, then cross-market.
6. Build scoring, calibration and explanations.
7. Iterate on the dev split, logging each change.
8. Freeze, run the final benchmark once, write `REPORT.md`.
