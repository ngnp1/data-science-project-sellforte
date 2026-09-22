# The Evaluation Harness

This scores a detector against the frozen benchmark. The team built and tested it before any real detector existed. It could not be tuned to flatter a result it had not yet seen.

## How to run it

```bash
# Free to run as often as you like. Every run is added to dev_history.jsonl.
python -m benchmark.eval.run_dev --detector benchmark.eval.adapter:detect --load-data

# The sealed test split. Run this ONCE, at the very end.
python -m benchmark.eval.run_final --detector benchmark.eval.adapter:detect --load-data --finalize
```

A detector is any function with the shape `(media_df, sales_df, sid) -> list[Event]`.

## Why you can trust the score

`tests/eval/test_baseline.py` checks this every time the test suite runs. A perfect detector must score exactly 1.0 on precision, recall, F1, IoU, channel accuracy, market accuracy, and day-level F1 across the whole dev split. If that check ever fails, the harness can no longer tell a correct detector from a broken one. Nothing it reports can be trusted after that.

A few built-in detectors prove the harness measures what it claims to measure:

| Detector | What it proves |
|---|---|
| `perfect_oracle` | The harness recognizes a correct detector. |
| `never_detect` | A detector that finds nothing still gets a perfect false-positive rate. FP rate alone is not enough to judge a detector. |
| `detect_everything` | Reporting everything destroys precision and the null-scenario FP rate. |
| `ungrouped_pulse_oracle` | Checks that the detector groups pulse events correctly (see below). |
| `panel_launch_oracle` | Checks that staggered launches are split per market (see below). |
| `shifted_oracle(n)` | Checks that IoU and boundary error respond to a shifted event window. |
| `wrong_channel_oracle` | Checks that a wrong channel shows up as a channel error, not just a missed event. |

**`never_detect` and `detect_everything` score identically on precision, recall, and F1: 0.000 for both.** One detector finds nothing, and the other finds everything. Only the null-scenario false-positive rate tells them apart. Never read F1 alone on this benchmark. A detector that sees events everywhere looks exactly as bad as one that finds nothing.

## Three things that can silently score a good detector as zero

The benchmark's ground truth and a detector's expected output do not always match shape. If a detector or its loader misses one of these, a correct detector can score zero without any error message.

1. **Pulse trains.** The ground truth has one row per individual off-window. A detector should emit one grouped event per pulse train. An ungrouped detector, emitting one event per window, matches no single truth row. A perfect pulse detector would then score zero on a third of the test split. The loader already groups the truth rows to match.
2. **`global_pause`.** The ground truth calls this pattern type `global_pause`. The detector should label it `dark_period` and add a `global_pause` tag. The loader already relabels it.
3. **`staggered_launch`.** The ground truth has one row per market that turned the channel on late. A detector must emit one event **per market**, each carrying that market's country code, rather than a single event for the whole panel. The loader does not do this one for you. A detector that emits one panel-wide event scores 0.000 on every staggered launch, 16 of the 127 matchable test events. `panel_launch_oracle` exists to catch this mistake.

Event end dates come from `scenario.json`'s `end_day` field, never from the CSV's `end_date` column, which uses a different convention.

## Breakdowns

The benchmark can slice results in two ways:

| Function | Slices by | Based on |
|---|---|---|
| `breakdowns` | Noise level, event family, country count, channel count, years, trend, market spread | Whole scenarios |
| `event_breakdowns` | Duration, magnitude, exact-zero vs. near-zero | Individual events |

Event-level breakdowns show recall only. A false positive does not belong to any truth event, so there is nothing to compute precision against.

**Only trust the `noise_level` breakdown as a real effect.** The generator ties `trend_p` and `market_spread` to event family on both splits. A number reported against either one really measures family difficulty, not that axis.

## The one-shot final run

`run_final.py` refuses to run without `--finalize`. It checks the test split's seal before reading anything. Then it records a SHA-256 hash of every Python file under `detection/`, plus the sealed spec hash, to `final_runs.jsonl`. You can run it more than once, but every run after the first stays permanently visible in that file.

`run_final.py` writes that record before it renders the report. A typo in `--out` cannot spend your one run without leaving a trace.

## Revised detector versus historical results

The committed final-run log describes the earlier detector. It has not been
rerun or overwritten after the correctness changes. The revised detector uses
uncalibrated heuristic scores, so its adapter leaves probability confidence
unset and reliability curves are unavailable. Use `python -m scripts.evaluate_sample`
for a reproducible check on the included CSVs; this is not a new held-out score.
