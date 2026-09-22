# Benchmark data

The benchmark defines 100 synthetic datasets, called scenarios. Each has daily media spend and sales, plus a separate answer file listing the inserted events.

- **Development (`dev`):** 45 scenarios for testing and improving the detector.
- **Test (`test`):** 55 scenarios intended for evaluation after development is finished.

Using test answers to adjust the detector makes a later test score less useful as evidence of performance on unseen data.

## What it covers

Scenarios include the [six event types](../README.md), events across markets, several events at once, short or unusual cases, and data with no inserted event. Scenarios without events help measure false alarms.

| Scenario family | Dev | Test |
|---|---:|---:|
| No event | 4 | 5 |
| Dark period | 3 | 4 |
| Single-channel period | 3 | 4 |
| Natural holdout | 3 | 4 |
| Step change | 5 | 6 |
| Channel pulse | 3 | 4 |
| Staggered launch | 3 | 4 |
| Cross-market event | 3 | 4 |
| Several events | 8 | 10 |
| Edge cases | 10 | 10 |
| **Total** | **45** | **55** |

## Get data ready

The full generated CSVs are **not included** in this repository. For a quick check, use the sample and commands in the [main README](../README.md#check-the-results).

To generate development data, first install the project's Python dependencies and the [R dependencies](../synthetic_data_generator/README.md#setup). With the Python environment active, run from the repository folder:

```bash
python -m benchmark.harness.generate --split dev
```

This writes data under `benchmark/datasets/dev/` and answers under `benchmark/datasets/dev_truth/`. Each scenario has its own folder. The test split uses `test/` and `test_truth/` in the same way.

The repository contains a seal and file checksums for the original test data. To reproduce that recorded evaluation, you need the matching original files. Seeds support repeatable generation, but library or environment changes can change the output. Do not replace the original seal or treat regenerated data as the original run.

For a separate experiment, generate into another directory:

```bash
python -m benchmark.harness.generate --split all --root /tmp/sellforte-benchmark
```

The standard evaluation commands use `benchmark/datasets/`; they do not automatically use this separate directory.

## What the benchmark can tell us

A score measures how well the detector finds inserted patterns under these simulation settings. It does not establish accuracy on real company data or prove that a detected period supports a causal conclusion.

Scenario settings also overlap: for example, some event families use particular trends and market sizes. Their score differences cannot be attributed to one setting alone. The noise settings mainly affect sales and media response, so they are not a direct test of noisy spend detection.

Continue with [evaluation instructions](eval/README.md) or the [folder guide](STRUCTURE.md).
