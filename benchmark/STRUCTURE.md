# How the Benchmark Directory Is Organized

This document explains how the benchmark directory is structured.

The main [`BENCHMARK.md`](BENCHMARK.md) file explains the purpose of the benchmark and summarizes the scenarios. This file focuses only on the directory layout and how data moves through it.

At a high level, the data goes through these components in order:

1. `spec/` defines the scenarios.
2. `harness/` generates and seals the data.
3. `datasets/` stores the frozen output.

The `eval/` directory will be used to score detectors against the generated data. It does not exist on this branch yet.

## `spec/`: Defining the scenarios

The `spec/` package contains the complete scenario definitions. It is deterministic and committed to the repository, so it effectively serves as the benchmark’s answer key.

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

The truth directories are used only after the detector produces its results. They allow you to evaluate how accurate the detector was.
