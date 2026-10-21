# Evaluating a detector

Evaluation compares a detector's findings with the events inserted into synthetic data.

## Run it

```bash
python -m scripts.evaluate_sample                                                   # included sample
python -m benchmark.eval.run_dev --detector benchmark.eval.adapter:detect --load-data   # dev split
```

The dev split must be generated first ([setup](../BENCHMARK.md#get-data-ready)). Every dev run is appended to `dev_history.jsonl`. To score another detector, see [adding a detector](../../detectors/README.md#score-it).

## How findings are matched

A finding matches an expected event when it has the same type, market and channel, and its dates overlap enough. Overlap is **IoU**: the shared days divided by all days covered by either window. A match needs an IoU of at least 0.5. Each finding and each expected event can be matched only once, and the matcher maximises the number of valid matches.

| Metric | Meaning |
|---|---|
| Precision | Of the reported findings, the share that matched. |
| Recall | Of the expected events, the share that were found. |
| F1 | Combines precision and recall. |
| Boundary error | Days between the reported and expected start and end dates. |
| Null-scenario false positives | Findings reported in data with no inserted events. |

Read these together. A detector that reports nothing has no false alarms, but it also finds nothing.

## Conventions

- Both dates of a window are inclusive.
- Several pauses of one pulse form a single event. Separate per-pause scores are reported as well.
- A global pause counts as a `dark_period`.
- A staggered launch is reported once per affected market.
- Breakdowns by family, duration or noise are descriptive. Settings vary together, so they do not isolate causes.
- The detector's confidence is a heuristic, not a probability, so reliability curves are not shown.

## Final evaluation

At the end of development, with the original sealed test files in place:

```bash
python -m benchmark.eval.run_final --detector benchmark.eval.adapter:detect --load-data --finalize
```

It checks the seal before reading anything. It then records hashes of `detection/`, `detectors/` and the scenario specification in `final_runs.jsonl`, before printing the report. Every run stays visible in that file. The committed final run describes an older detector version.
