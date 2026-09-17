# How the benchmark directory fits together

This is an orientation document. `BENCHMARK.md` holds the use case and the
numbers. This file holds only the layout.

At this stage, data flows through three parts, in this order:

1. `spec/` defines the scenarios.
2. `harness/` generates and seals them.
3. `datasets/` holds the frozen output.

A fourth part, `eval/`, scores a detector against that output. It does not
exist yet on this branch.

## `spec/` defines the scenarios

This package is pure, deterministic and committed, so it is the answer key.
No detector may read it, import it, or ask another agent what it contains.

- `axes.py` holds the pools and presets that a scenario draws from: countries,
  channels, noise levels, market spread. It knows nothing about events.
- `events.py` builds the event entries, one function per event family. It also
  names the two negative-control pattern types in `NON_EVENT_TYPES`.
- `scenarios.py` assembles the 100 scenarios. `FAMILY_COUNTS` sets the family
  sizes, `build_split()` returns one split, and `spec_hash()` hashes the
  definitions for the seal.

## `harness/` generates and seals

- `config_writer.py` writes one `Scenario` out as the two YAML files that the R
  generator reads.
- `runner.py` runs one scenario through the R and Python generator, then files
  the output. `media.csv` and `sales.csv` go to the data side. Every other
  output goes to the truth side. `R_TIMEOUT_S` caps a single R run at 1800 s.
- `generate.py` is the command-line entry point. It also holds the measured
  cost model, and it refuses to seal the dev split.
- `seal.py` writes `manifest.sha256` for both sides and stamps the `SEALED`
  marker. `verify_seal()` checks the manifests and the recorded `spec_hash`.

## `datasets/` holds the frozen output

| directory | scenarios | contents |
|---|---|---|
| `dev/` | 45 | `media.csv`, `sales.csv` |
| `dev_truth/` | 45 | `ground_truth.csv`, `meta.json`, `scenario.json`, `true_roi.csv` |
| `test/` | 55 | `media.csv`, `sales.csv`, plus `SEALED` and `manifest.sha256` |
| `test_truth/` | 55 | the same four truth files, plus `manifest.sha256` |

**The boundary sits here.** A detector under test may read `datasets/dev/`
and `datasets/test/`. It must never read `dev_truth/` or `test_truth/`.
