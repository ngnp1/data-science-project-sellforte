# The Frozen Benchmark

100 scenarios: 45 development, 55 hold-out test. Test split sealed 2026-09-13T16:23:57+00:00, spec_hash `d0eeba7999820bfbc2563f177b2a407972358a2886560dc652a3859567e5c8c7`.

Dataset directories hold only `media.csv` and `sales.csv`. Ground truth, `meta.json`, `scenario.json` and `true_roi.csv` live under `<split>_truth/`.

Scenario ids are `<split>_<counter>` — `dev_001`, `test_042` — and deliberately carry **no family suffix**. They used to read `test_001_null` / `test_018_step` / `test_042_mixed`, which announced the event family in the one path a detector is allowed to see: five of 55 test paths said "this scenario contains no events" outright. Nothing may reintroduce a suffix, and nothing may parse meaning out of a sid. The family lives in the truth-side `meta.json`, which is where evaluation breakdowns read it.

## The black-box boundary is procedural, not cryptographic

Two things are off limits to detector development, and they are equally off limits:

1. `benchmark/datasets/<split>_truth/` — the answers as files.
2. `benchmark/spec/` — the answers as code. `scenarios.py` is pure, deterministic and committed, so `build_split("test")[5].events[0]` returns the complete ground truth of the sealed split in a single import, with no file to open. A seal on the CSVs does nothing about this.

`detection/` must never import `benchmark.spec` or `benchmark.eval`, and whoever (or whatever) is developing detectors must not read `benchmark/spec/`, execute it, or ask another agent what it contains.

**What the seal proves, precisely:** the sealed data has not *changed* since it was frozen, and the scenario definitions in the repo still hash to what the marker recorded. `verify_seal("test")` checks both — every file against `manifest.sha256` on the data and truth sides, and the marker's `spec_hash` against what `benchmark/spec/` produces today.

**What the seal does not prove:** that nobody *looked*. It is a tamper record, not an access control. There is no mechanism here that prevents a reader; the discipline is the control. Overselling that would be worse than useless, because it would invite someone to rely on it.

## Regenerating

```bash
python -m benchmark.harness.generate --split dev
python -m benchmark.harness.generate --split test --seal
```

`--seal` and `--seal-only` require an explicit `--split`; they refuse `--split all`, which would otherwise seal dev. Dev must stay writable — it is where iteration is allowed to happen.

Generation is deterministic: the same code produces the same 100 scenarios, so `spec_hash` is a full description of the benchmark.

## Families

| family | dev | test |
|---|---|---|
| null | 4 | 5 |
| dark | 3 | 4 |
| single_channel | 3 | 4 |
| holdout | 3 | 4 |
| step | 5 | 6 |
| pulse | 3 | 4 |
| launch | 3 | 4 |
| cross_market | 3 | 4 |
| mixed | 8 | 10 |
| edge | 10 | 10 |
| **total** | **45** | **55** |

## Coverage

- **noise_level**: `high`x28, `low`x41, `med`x31
- **market_spread**: `extreme`x33, `moderate`x33, `tight`x34
- **n_countries**: `1`x10, `2`x29, `3`x24, `5`x22, `8`x15
- **n_channels**: `12`x19, `2`x19, `4`x34, `6`x15, `9`x13
- **trend_p**: `0.0`x37, `0.5`x36, `1.0`x27
- **years**: `1`x15, `2`x85

## Which breakdowns this benchmark actually supports

With 45 dev scenarios and four 3-level variation axes, full orthogonality is arithmetically impossible: 3⁴ = 81 > 45. Something has to be confounded, and the honest thing is to say which.

**`noise_level` is the only axis stratified within family, and therefore the only one whose breakdown is causally interpretable.** It is keyed to the family-local counter, so every one of the 10 families spans all three noise levels on both splits — 0 of 10 families is missing a level on dev or on test.

**`trend_p` and `market_spread` are confounded with family on both splits and must not be reported as axis effects.** They are keyed to base-3 digits of the split-global counter, which walks families in order, so consecutive-counter buckets land inside family boundaries.

`trend_p` × family, counts per level:

| family | dev 0.0 / 0.5 / 1.0 | test 0.0 / 0.5 / 1.0 |
|---|---|---|
| null | 4 / 0 / 0 | 5 / 0 / 0 |
| dark | 3 / 0 / 0 | 4 / 0 / 0 |
| single_channel | 2 / 1 / 0 | 0 / 4 / 0 |
| holdout | 0 / 3 / 0 | 0 / 4 / 0 |
| step | 0 / 5 / 0 | 0 / 1 / 5 |
| pulse | 0 / 0 / 3 | 0 / 0 / 4 |
| launch | 0 / 0 / 3 | 4 / 0 / 0 |
| cross_market | 0 / 0 / 3 | 4 / 0 / 0 |
| mixed | 8 / 0 / 0 | 1 / 9 / 0 |
| edge | 1 / 9 / 0 | 1 / 0 / 9 |

- **8 of 10 dev families and 7 of 10 test families sit entirely at a single `trend_p`.** All 10 families on both splits are missing at least one level.
- **dev's `trend_p = 1.0` bucket contains zero `null` scenarios** (all 4 dev nulls are at 0.0, all 5 test nulls likewise). The null scenarios are the only source of a false-positive denominator, so a precision or false-positive-rate figure quoted for that bucket has nothing under the line.
- **test's `trend_p = 1.0` bucket holds 9 of the 10 edge cases**, including `too_short`, `gradual_ramp` and `single_channel_market`. A "detection degrades at high trend" reading there is measuring edge-case difficulty.

`market_spread` is confounded the same way: **7 of 10 test families and 8 of 10 dev families are missing at least one spread level** (on dev, `cross_market`, `launch` and `pulse` are pinned to one level outright).

So:

- **Report as axis effects:** `noise_level`; per-family and per-event-type breakdowns; duration; magnitude; mixed vs single-event; near-zero vs exact-zero; channel count and market size within a family.
- **Do not report as axis effects:** `trend_p`, `market_spread`. If they appear at all, they must be labelled as confounded with family, and never sliced family × axis — those cells hold 0–9 scenarios and most hold 0.

## Reading ground truth: three things the eval loader must handle

Ground-truth granularity does not match the detector output spec section 7 prescribes. All three mismatches are frozen into the sealed data, so the loader — not the detector — is where they get reconciled. A detector that is working perfectly will score zero on a third of the test events if these are skipped.

### 1. `channel_pulse` truth is one row per off-window; the detector emits one grouped event

`ground_truth.csv` records **one row per individual off-window**. `dev_019` has 4 rows of 14 days each, spanning 2024-05-16 to 2024-09-12:

```
DE_PULSE_AFFILIATE_136,channel_pulse,DE,Affiliate,2024-05-16,2024-05-30,,...(pulse 1 of 4).
DE_PULSE_AFFILIATE_171,channel_pulse,DE,Affiliate,2024-06-20,2024-07-04,,...(pulse 2 of 4).
DE_PULSE_AFFILIATE_206,channel_pulse,DE,Affiliate,2024-07-25,2024-08-08,,...(pulse 3 of 4).
DE_PULSE_AFFILIATE_241,channel_pulse,DE,Affiliate,2024-08-29,2024-09-12,,...(pulse 4 of 4).
```

Spec section 7's P3 emits **one** grouped event spanning first start to last end, with the individual windows attached as `components`. Spec section 9 matches one-to-one at IoU ≥ 0.5. The grouped event's IoU against any single truth row here is about 14/119 ≈ 0.118 — far under the threshold — so **a perfect pulse detector matches nothing and scores zero.**

This is not a rounding error in the headline: `channel_pulse` is **42 of the 127 matchable test events**, a third of the test-split F1 (27 of 92 on dev).

**The loader must group pulse truth rows by `(country_code, channel)`** into one event spanning min(start) to max(end), keeping the individual rows as components for a component-level secondary view. The `description` field carries `pulse i of n`, which gives both the group size and the ordering.

### 2. `global_pause` is a truth `pattern_type` but a detector *tag*

Truth records `global_pause` as the `pattern_type`, one row per country, with `channel = "ALL"` (3 rows on test, 2 on dev). Spec section 7 labels the underlying regime **`dark_period`** and applies `global_pause` as a **tag** on the cross-market layer when every peer is also off. Strict type matching therefore fails on every one of these.

The loader needs a type-mapping table, not a string comparison. At minimum:

| truth `pattern_type` | detector label | note |
|---|---|---|
| `global_pause` | `dark_period` + tag `global_pause` | match on the regime label; the tag is a secondary correctness check |
| `dark_period` | `dark_period` | |
| `single_channel` | `single_channel` | `channel` names the channel that stays **on** |
| `natural_holdout` | `natural_holdout` (or `cross_market_holdout`) | `channel` names the channel that goes **off** |
| `channel_pulse` | `channel_pulse` | after grouping, see (1) |
| `step_change` | `step_change` | |
| `staggered_launch` | `staggered_launch` | see (3) |

### 3. `staggered_launch` truth is per-market; the detector emits one panel-level event

Truth records one row per late market, each with its own `country_code` and onset window. Spec section 7's P4 plus the cross-market layer emits a **panel-level** `staggered_launch` listing each market's onset date — an event with no single `country_code` to match on. 16 truth rows on test, 10 on dev.

The loader needs one of two conventions, chosen and stated once:

- **fan out** the panel event into one per listed market and match per-country (keeps the section 9 matcher unchanged); or
- **group** truth rows by `(channel, scenario)` into one panel event and match on the set of onset dates.

Either works; silently comparing a country-less detection against per-country truth does not.

## Two more loader traps

### `end_date` is exclusive — except at the series end

Spec section 2 documents `ground_truth.csv`'s `end_date` as slice-exclusive and tells the loader to subtract a day. That is right most of the time but **not uniform**. `reformat.py` clamps the index:

```python
end_idx = min(ev["end_day"], len(dates) - 1)
```

so for any window running to the end of the series, `end_day == n_days` is clamped to the last date and the recorded `end_date` is **already inclusive**. The two conventions are indistinguishable from the CSV alone.

**The rule:**

> `inclusive_end = end_date − 1 day`, **except** where `end_date` equals the series' last date, in which case `end_date` is already inclusive.

That rule is correct for all 100 scenarios today. Verified on disk: `censored_start` needs the subtraction; `censored_end` and `single_channel_market` do not.

Affected: 4 of 100 scenarios (`dev_037`, `dev_042`, `test_047`, `test_052` — the `censored_end` and `single_channel_market` edge cases), 2 of them in the sealed split.

**The alternative, and arguably better, route:** read `end_day` from the truth-side `scenario.json` and use `end_day − 1` as the inclusive end index. `end_day` is the unclamped slice-exclusive day number, so it needs no special case — the ambiguity exists only in the date rendering.

### Magnitude is not in `ground_truth.csv` or `meta.json` — read `scenario.json`

`ground_truth.csv` leaves `multiplier` **blank for every pattern type except `step_change`** (see `reformat.py`'s `build_ground_truth`), and `meta.json` carries no magnitude field at all. Spec section 9 item 10 requires breakdowns by magnitude and by near-zero versus exact-zero — and holdout magnitudes are drawn from `[0, 0, 0.02, 0.05, 0.08]`, none of which reaches either file.

The frozen CSVs are not changing. **Magnitude breakdowns must read the per-event `multiplier` from `<split>_truth/<sid>/scenario.json`**, where every event carries its full definition (`start_day`, `end_day`, `multiplier`, `pattern_type`, `pattern_id`, `description`). Join to `ground_truth.csv` on `pattern_id`.

For orientation, from the spec (not from the frozen files): test's 26 `natural_holdout` events split 16 exact-zero / 10 near-zero (`0.02`×4, `0.05`×1, `0.08`×5); dev's 20 split 15 / 5.

`scenario.json` is required by `runner._is_complete`, so a scenario missing it counts as ungenerated and neither the resume logic nor `--seal-only` will pass over it.

## Negative controls

Two pattern types change the data but must never be reported as detections; anything found inside their windows counts as a false positive:

- `ramp_block` — gradual drift, not a step change
- `intermittent_baseline` — a flighting channel whose short gaps are normal

Plus 9 `null` scenarios carrying no events at all, which are the only way to measure a false-positive rate.

## Runtime

The cost model in `benchmark/harness/generate.py` was measured **single-process** and underestimates by roughly 2× under contention. What the full run actually cost:

| | |
|---|---|
| model estimate | 52 min at 6 workers |
| **measured** | **99.3 min at `--workers 6`** |
| contention factor | ≥ 2.68× |
| outcome | 97 generated, 2 skipped, **1 failed** |

`dev_018` (then `dev_018_step`) — 8 countries × 12 channels × 1 year, the most expensive point in the whole variation space — **timed out at 1801.0 s** against `runner.R_TIMEOUT_S = 1800`. It is modelled at ~840 s, only ~2.14× of headroom against the timeout, which is less than the contention factor already observed.

`R_TIMEOUT_S` was deliberately left alone. The retry recipe, which succeeded in 10.1 min:

```bash
python -m benchmark.harness.generate --split dev --workers 1
```

Generation is resumable, so this skips every complete scenario and retries only the missing one; alone on the machine it gets a whole core instead of a sixth of one. Raise `R_TIMEOUT_S` only if a scenario also times out at `--workers 1` — that would mean the constant is genuinely too low, rather than that six R processes were fighting over six performance cores.

## The evaluation harness

`benchmark/eval/` implements every reconciliation this document describes, and
`benchmark/eval/README.md` explains how to run it. The loader traps above are
not advisory — `tests/eval/` asserts on every suite run that a perfect oracle
scores exactly 1.0 over the whole dev split, which is only true if all three
reconciliations are applied.
