# How the Benchmark Directory Is Organized

This document explains how the benchmark directory is structured.

The main [`BENCHMARK.md`](BENCHMARK.md) file explains the purpose of the benchmark and summarizes the scenarios. This file focuses only on the directory layout and how data moves through it.

At a high level, the data goes through these components in order:

1. `spec/` defines the scenarios.
2. `harness/` generates and seals the data.
3. `datasets/` stores the frozen output.
4. `eval/` scores a detector against that output.

## `spec/`: Defining the scenarios

The `spec/` package contains the complete scenario definitions. It is deterministic and committed to the repository, so it effectively serves as the benchmark's answer key.

A detector must not read or import this package. It must also not ask another agent to reveal what it contains.

The main files are:

* **`axes.py`** defines the values that scenarios can use, such as countries, channels, noise levels, and market sizes. It does not define events.
* **`events.py`** creates the event definitions. It contains one function for each event family and defines the two negative-control patterns in `NON_EVENT_TYPES`.
* **`scenarios.py`** assembles all 100 scenarios. `FAMILY_COUNTS` defines how many scenarios belong to each family, `build_split()` creates a split, and `spec_hash()` creates a hash of the definitions for sealing.

## `harness/`: Generating and sealing the data

The `harness/` package turns the scenario definitions into datasets and protects the test split from accidental changes.

* **`config_writer.py`** converts a `Scenario` into the two YAML files required by the R generator.
* **`runner.py`** runs each scenario through the R and Python generators and stores the results. It writes `media.csv` and `sales.csv` to the dataset side. All other outputs go to the truth side. A single R run is limited to 1,800 seconds by `R_TIMEOUT_S`.
* **`generate.py`** is the command-line entry point. It also contains the measured cost model and prevents the dev split from being sealed.
* **`seal.py`** creates `manifest.sha256` files for both sides and adds the `SEALED` marker. `verify_seal()` checks the manifests and confirms that the recorded `spec_hash` is correct.

## `datasets/`: Storing the frozen output

The generated datasets are stored in the following directories:

| Directory     | Scenarios | Contents                                                         |
| ------------- | --------: | ---------------------------------------------------------------- |
| `dev/`        |        45 | `media.csv`, `sales.csv`                                         |
| `dev_truth/`  |        45 | `ground_truth.csv`, `meta.json`, `scenario.json`, `true_roi.csv` |
| `test/`       |        55 | `media.csv`, `sales.csv`, plus `SEALED` and `manifest.sha256`    |
| `test_truth/` |        55 | The same four truth files, plus `manifest.sha256`                |

## The detector boundary

The important boundary is between the dataset files and the truth files.

A detector may read:

* `datasets/dev/`
* `datasets/test/`

A detector must never read:

* `datasets/dev_truth/`
* `datasets/test_truth/`

A detector must also never import `benchmark.spec` or `benchmark.eval`. Those packages hold or produce the answers, and importing either one reaches the answer key directly, without opening a truth file at all.

The truth directories are used only after the detector produces its results. They allow you to evaluate how accurate the detector was.

## `eval/`: Scoring a detector

The `eval/` package exists now. The core path, in call order:

* **`model.py`** holds the shared vocabulary: the `Event` dataclass, `TYPE_MAP`, and `NON_EVENT_TYPES`. It holds no logic.
* **`truth.py`** loads ground truth into matchable intervals. It settles two of the loader traps `BENCHMARK.md` lists and decides the convention for a third.
* **`matching.py`** matches detections to truth greedily by temporal IoU, one to one, at a threshold of 0.5.
* **`metrics.py`** computes the ten metrics of spec section 9.
* **`breakdowns.py`** slices the results per axis. It carries the confounding warning along with the numbers.
* **`runner.py`** runs a detector over a whole split and aggregates everything above.
* **`report.py`** renders one result as Markdown.

Entry points and support:

* **`adapter.py`** converts a `detection.model.DetectedEvent` into the harness `Event`. The dependency points this way so that `detection/` stays blind to the harness.
* **`run_dev.py`** scores a detector against the dev split. Run it as often as you like.
* **`run_final.py`** scores a detector against the sealed test split. It needs an explicit `--finalize` flag and checks the seal before it reads anything.
* **`detectors_for_testing.py`** holds the oracles that prove the harness recognizes a correct detector. They are for the dev split only.
* **`README.md`** explains how to run all of this.

## The audit trail

Two files record what happened. Neither one is a log that anybody may rewrite.

* **`eval/dev_history.jsonl`** records every development run. It holds 23 records today.
* **`eval/final_runs.jsonl`** records the single final run against the sealed test split. It holds 1 record.

`BENCHMARK.md` explains why that matters.
