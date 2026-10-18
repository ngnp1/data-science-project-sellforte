# Benchmark data

The benchmark has 100 synthetic datasets ("scenarios"), each with daily spend, sales and a separate answer file listing the inserted events.

- **dev**: 45 scenarios for developing the detector.
- **test**: 55 sealed scenarios for the final evaluation. Looking at test answers during development makes the test score meaningless.

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

Scenarios with no event measure false alarms.

## Get data ready

The generated CSVs are not in the repository. To create the dev split, install the [R dependencies](../synthetic_data_generator/README.md#setup), then run:

```bash
python -m benchmark.harness.generate --split dev
```

This writes inputs to `benchmark/datasets/dev/` and answers to `benchmark/datasets/dev_truth/`. To generate somewhere else, add `--root <folder>`; the evaluation commands only read `benchmark/datasets/`.

The test split is sealed with checksums. Regenerated data may differ across library versions, so do not replace the seal.

## Folder layout

```text
spec/       Scenario and event definitions.
harness/    Runs the generator and records checksums.
datasets/   Inputs (dev/, test/) and answers (dev_truth/, test_truth/), one folder per scenario.
eval/       Scores detectors against the answers (see eval/README.md).
```

| File | Purpose |
|---|---|
| [spec/scenarios.py](spec/scenarios.py) | Builds the scenario definitions. |
| [harness/generate.py](harness/generate.py) | Generates datasets. |
| [harness/seal.py](harness/seal.py) | Records and checks test-file checksums. |
| [eval/adapter.py](eval/adapter.py) | Runs a registered detector for evaluation. |
| [eval/run_dev.py](eval/run_dev.py), [eval/run_final.py](eval/run_final.py) | Evaluation commands. |

## What a score means

A score shows how well the detector finds the inserted patterns in this simulation. It says nothing certain about real company data. Some scenario settings vary together, so a difference between families cannot be blamed on one setting. The noise settings mainly affect sales, not spend.

Next: [how evaluation works](eval/README.md).
