# The Benchmark

This benchmark contains synthetic marketing-spend datasets. Each dataset has a known answer stored in a separate file. You run a detector on the data, then compare its results with the hidden answers.

There are 100 datasets, called **scenarios**. Each scenario contains a company’s daily advertising spend and sales data for one or more countries and channels, covering one or two years.

## The two splits

The scenarios are divided into two groups:

* **`dev`**: 45 scenarios for developing and testing your detector. You can check the correct answers as often as needed.
* **`test`**: 55 scenarios for the final evaluation. Do not check the answers in advance.

This separation makes the final score meaningful. If you tune a detector using the same data on which it is evaluated, the score may no longer reflect how well it performs on unseen data.

## File locations

```text
benchmark/datasets/dev/<scenario_id>/media.csv
benchmark/datasets/dev/<scenario_id>/sales.csv
benchmark/datasets/dev_truth/<scenario_id>/ground_truth.csv
```

The test split follows the same structure:

```text
benchmark/datasets/test/<scenario_id>/...
benchmark/datasets/test_truth/<scenario_id>/...
```

## What each scenario contains

Each scenario contains a small number of artificial **events** inserted into otherwise normal spend data. An event is a period during which something unusual happens: for example, a channel stops spending or its budget suddenly changes.

The benchmark includes the following event types:

| Event                 | Description                                                    |
| --------------------- | -------------------------------------------------------------- |
| Dark period           | All channels in a market stop spending at the same time.       |
| Single-channel period | All channels except one stop spending.                         |
| Natural holdout       | One channel stops while the others continue running.           |
| Step change           | A channel’s spend jumps to a new level and remains there.      |
| Channel pulse         | A channel repeatedly switches on and off.                      |
| Staggered launch      | A channel starts running later in some markets than in others. |

Some scenarios contain no event. These scenarios measure how often a detector raises a false alarm on normal data.

## Scenario distribution

| Scenario family         |    Dev |   Test |
| ----------------------- | -----: | -----: |
| No event                |      4 |      5 |
| Dark period             |      3 |      4 |
| Single-channel period   |      3 |      4 |
| Natural holdout         |      3 |      4 |
| Step change             |      5 |      6 |
| Channel pulse           |      3 |      4 |
| Staggered launch        |      3 |      4 |
| Cross-market event      |      3 |      4 |
| Several events combined |      8 |     10 |
| Edge cases              |     10 |     10 |
| **Total**               | **45** | **55** |

The scenarios also vary in noise level, trends, number of countries, number of channels, and market size. This ensures that the detector is tested under a range of conditions.

## Generating the data

The data has already been generated and frozen. You only need these commands if you want to rebuild it from scratch:

```bash
python -m benchmark.harness.generate --split dev
python -m benchmark.harness.generate --split test --seal
```

The generator is deterministic. The `--seal` option locks the test split to prevent accidental changes.

## Scoring a detector

See [`benchmark/eval/README.md`](eval/README.md) for the exact command and an explanation of the score.

## More information

[`benchmark/STRUCTURE.md`](STRUCTURE.md) explains how the benchmark directory is organized and is intended for anyone extending the benchmark.
