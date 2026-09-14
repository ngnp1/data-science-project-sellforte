# The Evaluation Harness

Scores a detector against the frozen benchmark. Built and validated **before**
any real detection algorithm existed, so a broken metric could not silently
flatter the results it was later used to judge.

## Running it

```bash
# Free. Run as often as you like; every run appends to dev_history.jsonl.
python -m benchmark.eval.run_dev --detector detection.pipeline:run_detection

# The sealed hold-out split. Deliberate, gated, recorded. Run ONCE.
python -m benchmark.eval.run_final --detector detection.pipeline:run_detection --finalize
```

A detector is any callable `(media_df, sales_df, sid) -> list[Event]`.

## What the harness guarantees

`tests/eval/test_baseline.py` asserts, on every suite run, that a perfect
oracle scores exactly 1.0 across precision, recall, F1, IoU, channel accuracy,
market accuracy and day-level F1 over the whole dev split. If that ever breaks,
the harness can no longer recognise a correct detector and nothing it reports
can be trusted.

Three synthetic detectors define the range:

| detector | what it proves |
|---|---|
| `perfect_oracle` | the harness recognises a correct detector |
| `never_detect` | the floor, and that a do-nothing detector still has a flawless false-positive rate — which is why FP rate alone is not a sufficient headline |
| `detect_everything` | brute-force recall is destroyed by precision and by the null-scenario FP rate |
| `ungrouped_pulse_oracle` | the pulse-grouping reconciliation is actually exercised |
| `panel_launch_oracle` | a `staggered_launch` left at spec §7's panel shape matches nothing — the fan-out is the detector's job |
| `shifted_oracle(n)` | IoU and boundary error respond to localisation error |
| `wrong_channel_oracle` | channel errors surface in channel accuracy rather than vanishing into FN+FP |

Recorded on the dev split (`baseline-*` entries in `dev_history.jsonl`):

| detector | precision | recall | F1 | null FP rate (per country-year) |
|---|---|---|---|---|
| `perfect_oracle` | 1.000 | 1.000 | 1.000 | 0.000 |
| `never_detect` | 0.000 | 0.000 | 0.000 | 0.000 |
| `detect_everything` | 0.000 | 0.000 | 0.000 | 0.654 |

> **`never_detect` and `detect_everything` are identical on precision, recall,
> and F1** — 0.000 across all three, for both — despite being opposite
> pathologies: one reports nothing, the other reports everything. The
> null-scenario false-positive rate is the *only* number that tells them
> apart. Read F1 alone on this benchmark and a detector that hallucinates
> events everywhere looks exactly as bad as one that does nothing at all;
> F1 must never be treated as a standalone headline here.

## Three shape mismatches — two the loader fixes, one the DETECTOR must

Ground-truth shape and detector output shape were fixed independently — by the
generator and by spec §7 — and they do not agree. `benchmark/eval/truth.py` is
where the convention is written down, and `BENCHMARK.md` documents why:

1. **Pulse trains** are one truth row per off-window but one detector event.
   Ungrouped, a perfect pulse detector scores zero on a third of the test split.
   *The loader groups them.*
2. **`global_pause`** is a truth `pattern_type` but a detector *tag* on a
   `dark_period` regime. *The loader relabels and tags it.*
3. **`staggered_launch`** truth is per-market; spec §7 has the detector emit
   **one panel-level event with no `country_code`**. The per-market shape is
   the convention, because it leaves the §9 matcher unchanged — but **the
   loader does nothing here.** Truth rows are already per-market, so **the
   DETECTOR must fan its panel-level `staggered_launch` out to one event per
   market**, each carrying that market's `country_code`. A detector that emits
   the panel event as spec §7 describes it scores **0.000 precision, recall and
   F1** on every `staggered_launch` — 16 of the 127 matchable test events.
   `panel_launch_oracle` is the negative control that keeps this stated.

Interval ends come from `scenario.json`'s `end_day`, never the CSV's
`end_date`, which is slice-exclusive except where it is clamped at the series
end.

## Breakdowns

Spec §9 item 10 is covered in two places, because no single source has all of
it:

| where | axes | bucketed by |
|---|---|---|
| `breakdowns` | `noise_level`, `family`, `n_countries`, `n_channels`, `years`, `trend_p`, `market_spread` | whole scenarios, on `meta.json` keys |
| `event_breakdowns` | `duration`, `magnitude`, `zero_kind` (exact-zero vs near-zero) | individual truth events, on `Event.n_days` and `Event.multiplier` |

`meta.json` carries no duration, magnitude or zero-kind field, so those three
cannot be scenario-level at all — they are properties of an event, and
`BENCHMARK.md` is explicit that magnitude has to be read per event from
`scenario.json` (which `truth.py` already does). The event-level tables are
**recall-only**: a truth event is matched or it is not, but a false positive
belongs to no truth bucket, so precision there would have no denominator.

### Which breakdowns to trust

Only `noise_level` is stratified within family. `trend_p` and `market_spread`
are confounded with family on both splits, and `breakdown_by` attaches a
warning to them that travels with the numbers. Reporting them as axis effects
would be reporting family difficulty under another name.

## The black-box gate

`run_final.py` refuses without `--finalize`, verifies the test split's seal
before reading anything, and appends to `final_runs.jsonl`: a SHA-256 over
**every `*.py` file under `detection/`** — their relative paths and their
contents, nothing else, so config, notebooks and data files under that tree are
*not* covered — plus the sealed `spec_hash`. A second final run is possible but
permanently visible there.

The audit record is appended **before** the report is rendered or `--out` is
written, and in a `finally`, so a mistyped `--out` cannot spend the seal
without leaving a record — which would let the next run call itself the first.
