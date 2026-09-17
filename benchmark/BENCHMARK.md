# The Benchmark

This is a set of fake marketing-spend datasets. Each one has a known answer
hidden in a separate file. You run a detector on the data and check its
answers against the hidden one.

There are 100 datasets, called **scenarios**. Each scenario is one company's
daily ad spend and sales, across one or more countries and channels, for one
or two years.

## The two splits

The scenarios are split into two groups:

- **`dev`**, 45 scenarios. Use this one while you build and test a detector.
  You can look at its answers as often as you like.
- **`test`**, 55 scenarios. Use this one only once, at the very end, to get
  an honest score. Do not look at its answers before that.

This split exists so the final score means something. If a detector is tuned
against the same data it is scored on, the score stops telling you anything
useful.

## Where the files live

```
benchmark/datasets/dev/<scenario_id>/media.csv         the data
benchmark/datasets/dev/<scenario_id>/sales.csv
benchmark/datasets/dev_truth/<scenario_id>/ground_truth.csv   the answer

benchmark/datasets/test/<scenario_id>/...               same shape, for test
benchmark/datasets/test_truth/<scenario_id>/...
```

A detector reads only `media.csv` and `sales.csv`. It never reads anything
under `dev_truth/` or `test_truth/`. Those folders exist only to check the
detector's answers afterward.

## What is inside each scenario

Every scenario plants a small number of "events" in otherwise normal spend
data. An event is a stretch of days where something unusual happens, such as
a channel going dark or a budget jumping to a new level. The full list:

| event | what happens |
|---|---|
| dark period | every channel in a market stops at once |
| single-channel period | every channel but one stops |
| natural holdout | one channel stops while the rest keep running |
| step change | a channel's spend jumps to a new level and holds |
| channel pulse | a channel switches on and off several times |
| staggered launch | a channel turns on later in some markets than others |

Some scenarios have no event at all. These are here so you can measure how
often a detector raises a false alarm on ordinary data.

## How many of each

| family | dev | test |
|---|---|---|
| no event | 4 | 5 |
| dark period | 3 | 4 |
| single-channel | 3 | 4 |
| natural holdout | 3 | 4 |
| step change | 5 | 6 |
| channel pulse | 3 | 4 |
| staggered launch | 3 | 4 |
| cross-market event | 3 | 4 |
| several events mixed together | 8 | 10 |
| edge cases | 10 | 10 |
| **total** | **45** | **55** |

Scenarios also vary in noise level, trend, number of countries, number of
channels, and market size, so a detector gets tested under more than one
condition.

## How to generate the data

The data is already generated and frozen. You only need this if you want to
rebuild it from scratch:

```bash
python -m benchmark.harness.generate --split dev
python -m benchmark.harness.generate --split test --seal
```

Rebuilding uses the same code every time, so it produces the same 100
scenarios. `--seal` locks the test split so nobody can change it by
accident afterward.

## How to score a detector

See [`benchmark/eval/README.md`](eval/README.md). It has the exact command
and explains what the score means.

## Want more detail?

[`benchmark/STRUCTURE.md`](STRUCTURE.md) explains how this folder is put
together, for anyone extending the benchmark itself.
