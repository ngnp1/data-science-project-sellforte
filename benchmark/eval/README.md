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
| `shifted_oracle(n)` | IoU and boundary error respond to localisation error |
| `wrong_channel_oracle` | channel errors surface in channel accuracy rather than vanishing into FN+FP |

## Three things the loader reconciles

Ground-truth shape and detector output shape were fixed independently — by the
generator and by spec §7 — and they do not agree. `benchmark/eval/truth.py` is
where they are brought together, and `BENCHMARK.md` documents why:

1. **Pulse trains** are one truth row per off-window but one detector event.
   Ungrouped, a perfect pulse detector scores zero on a third of the test split.
2. **`global_pause`** is a truth `pattern_type` but a detector *tag* on a
   `dark_period` regime.
3. **`staggered_launch`** truth is per-market; the detector emits a panel event.
   Convention chosen here: fan out to one event per market.

Interval ends come from `scenario.json`'s `end_day`, never the CSV's
`end_date`, which is slice-exclusive except where it is clamped at the series
end.

## Which breakdowns to trust

Only `noise_level` is stratified within family. `trend_p` and `market_spread`
are confounded with family on both splits, and `breakdown_by` attaches a
warning to them that travels with the numbers. Reporting them as axis effects
would be reporting family difficulty under another name.

## The black-box gate

`run_final.py` refuses without `--finalize`, verifies the test split's seal
before reading anything, and appends the SHA-256 of the entire `detection/`
tree plus the sealed `spec_hash` to `final_runs.jsonl`. A second final run is
possible but permanently visible there.
