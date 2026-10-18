# Benchmark folder guide

The benchmark has four parts:

```text
spec/       Defines the scenarios and inserted events.
harness/    Runs the generator and records checksums.
datasets/   Holds generated inputs and separate answers.
eval/       Runs detectors and compares findings with answers.
```

The full generated datasets are not included in the repository. See [data setup](BENCHMARK.md#get-data-ready).

## Data and answers

For each scenario, the folder layout is:

```text
datasets/dev/<scenario_id>/
    media.csv
    sales.csv
datasets/dev_truth/<scenario_id>/
    ground_truth.csv
    scenario.json
    meta.json
    true_roi.csv
```

The test split follows the same layout using `test/` and `test_truth/`.

The detector reads only media and sales data. Evaluation code reads the answers after detection. Keep `detection/` independent of `benchmark.spec`, `benchmark.eval`, and truth files so the detector cannot use the answers to produce findings.

## Useful entry points

| File | Purpose |
|---|---|
| [spec/scenarios.py](spec/scenarios.py) | Assembles scenario definitions. |
| [harness/generate.py](harness/generate.py) | Command to generate datasets. |
| [harness/runner.py](harness/runner.py) | Runs R and Python for each scenario. |
| [harness/seal.py](harness/seal.py) | Records and verifies test-file checksums. |
| [eval/adapter.py](eval/adapter.py) | Converts detector output into evaluation events. |
| [eval/truth.py](eval/truth.py) | Loads answers and applies date and grouping conventions. |
| [eval/matching.py](eval/matching.py) | Pairs findings with expected events. |
| [eval/metrics.py](eval/metrics.py) | Calculates scores. |
| [eval/run_dev.py](eval/run_dev.py), [eval/run_final.py](eval/run_final.py) | Commands to evaluate a detector. |

## Saved runs and the test seal

`eval/dev_history.jsonl` records development runs. `eval/final_runs.jsonl` records final runs, including code and specification hashes. Preserve old records so results remain traceable to the version evaluated.

The test seal records file checksums and the scenario specification hash. Verification detects changes or missing files; it does not hide test answers or prove they were never consulted. Logs are ordinary files, with version history providing an additional record.

The saved final results belong to the earlier detector. Read [how evaluation works](eval/README.md) before comparing them with current results.
