# Evaluating the detector

Evaluation compares detected events with known events in synthetic data. It checks both whether the right pattern was found and whether its dates overlap the expected window.

## Run a check

For the included sample, run from the repository folder with the Python environment active:

```bash
python -m scripts.evaluate_sample
```

For the larger development split, first follow the [benchmark setup](../BENCHMARK.md#get-data-ready), then run:

```bash
python -m benchmark.eval.run_dev --detector benchmark.eval.adapter:detect --load-data
```

You can repeat development runs. Each run is added to `dev_history.jsonl`.

## How events match

A finding must have the correct event type, market, and channel. Its dates must also overlap the expected dates enough to count as a match.

The overlap measure is **intersection over union (IoU)**: the number of days shared by both windows divided by the number of days covered by either window. For example, 10 shared days out of 15 total days gives an IoU of 0.67. The matching threshold is 0.5.

The matcher takes the strongest overlaps first. Each finding can match only one expected event, and each expected event can match only one finding.

| Metric | Plain-language meaning |
|---|---|
| Precision | Of the events reported, how many matched an expected event? |
| Recall | Of the expected events, how many were found? |
| F1 | A combined score that rewards both precision and recall. |
| Boundary error | How far the reported start and end dates are from the expected dates. |
| Null-scenario false positives | How often the detector reports events in data with no inserted events. |

Read the metrics together: finding nothing avoids false alarms but also misses every event. Built-in test detectors with perfect, empty, shifted, and incorrect answers check the evaluator's behavior.

## Date and grouping rules

- **Dates:** both ends of an evaluation window are inclusive. Truth comes from `scenario.json`, converting the exclusive `end_day` to the last affected day. The legacy truth CSV uses a different end-date convention.
- **Pulses:** several off-windows are grouped into one pulse event. Standard overlap scoring uses the group's outer window, so a good score alone does not prove each pause is correct. The sample check also verifies individual pulse windows.
- **Global pauses:** these are evaluated as `dark_period` events; the detector can add a `global_pause` tag.
- **Staggered launches:** return one event per affected market, rather than one event for all markets.

## Interpret the results

Scenario breakdowns compare groups such as event family or noise level. Event breakdowns report recall by duration, magnitude, and zero versus near-zero spend. They do not report precision because an unmatched finding has no truth event to assign it to.

These are descriptive comparisons. Some settings vary together, and simulated noise mainly affects sales and media response. A breakdown does not isolate the effect of one setting or establish robustness to noisy spend.

Current confidence scores are heuristics, not probabilities. The adapter leaves probability confidence unset, so probability reliability curves are unavailable for this detector.

## Final evaluation and historical results

The final command is intended for the end of development, once the original sealed test files are available:

```bash
python -m benchmark.eval.run_final --detector benchmark.eval.adapter:detect --load-data --finalize
```

It verifies the seal and appends a record to `final_runs.jsonl`, including hashes of the detector code and scenario specification. Repeating it creates another record; the flag does not enforce a single lifetime run. Preserve the original records and seal.

The committed final-run log describes an older detector. It has not been rerun for the correctness fixes. The current sample result is a regression check, not a replacement for an unseen test result.
