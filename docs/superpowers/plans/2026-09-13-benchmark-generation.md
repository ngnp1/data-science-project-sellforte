# Benchmark Generation Implementation Plan (Plan 1 of 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a frozen, sealed synthetic benchmark of 100 scenarios (45 development, 55 hold-out test) whose ground truth is physically separated from the data a detector is allowed to read.

**Architecture:** The existing R + Python siMMMulator generator is patched to accept `--seed`, `--config`, `--events` and `--outdir`, with defaults that reproduce its current behaviour exactly. A Python layer then synthesises 100 `(config.yaml, events_config.yaml, seed)` triples from a seeded RNG, runs the generator over them in parallel across 6 workers, and files the output so that `datasets/<split>/<sid>/` contains only `media.csv` and `sales.csv` while everything else lands in `datasets/<split>_truth/<sid>/`. The test split is then sealed with a SHA-256 manifest.

**Tech Stack:** R 4.x (`siMMMulator`, `dplyr`, `yaml`, base `commandArgs` — no `optparse`), Python 3.13 with pandas 3.0.5, numpy 2.5.2, PyYAML 6.0.3, pytest. Virtualenv at `synthetic_data_generator/.venv`.

**Spec:** `docs/superpowers/specs/2026-09-13-informative-periods-detection-design.md`

## Global Constraints

- **Pure pandas and numpy.** No scipy, sklearn, or ruptures anywhere in the project. pytest and PyYAML are permitted; pytest is test-only.
- **The R generator gains no new package dependencies.** Argument parsing uses base `commandArgs(trailingOnly = TRUE)`.
- **Backward compatibility is mandatory.** After patching, `cd synthetic_data_generator && Rscript generate_with_simmmulator.R && .venv/bin/python reformat.py` must behave exactly as the existing README documents.
- **Dataset directories contain only `media.csv` and `sales.csv`.** Ground truth, `meta.json`, `true_roi.csv` and `raw_daily_wide.csv` all live on the truth side. This is asserted by a test, not by convention.
- **Seed ranges:** development scenarios draw seeds 1000–1999, test scenarios 5000–5999. Asserted disjoint.
- **Split sizes:** development 45, test 55. Family counts are fixed by the table in Task 6 and asserted exactly.
- **Python interpreter** is always `synthetic_data_generator/.venv/bin/python`. Never bare `python3`.
- All paths in this plan are relative to the project root, `/Users/user123/Desktop/projects/datascience-project`.

## File Structure

| file | responsibility |
|---|---|
| `synthetic_data_generator/cli_args.R` | **new** — `parse_cli_args()`, isolated so it is testable without running a simulation |
| `synthetic_data_generator/generate_with_simmmulator.R` | **modify** — source the parser, honour `--seed/--config/--events/--outdir` |
| `synthetic_data_generator/reformat.py` | **modify** — same four arguments via `argparse` |
| `benchmark/spec/axes.py` | **new** — country pool, channel pool, noise/trend/seasonality presets, market-spread presets |
| `benchmark/spec/events.py` | **new** — one builder function per event family, returning `events_config.yaml` entries |
| `benchmark/spec/scenarios.py` | **new** — assembles the 100 `Scenario` objects deterministically |
| `benchmark/harness/config_writer.py` | **new** — `Scenario` → the two YAML files on disk |
| `benchmark/harness/runner.py` | **new** — runs one scenario end to end; parallel driver over many |
| `benchmark/harness/seal.py` | **new** — manifest writing, sealing, verification |
| `benchmark/harness/generate.py` | **new** — CLI entry point |
| `tests/benchmark/…` | **new** — one test module per source module above |

---

### Task 1: Project scaffolding

Sets up git, pytest, and the package skeleton. Everything later commits into this.

**Files:**
- Create: `.gitignore`
- Create: `pytest.ini`
- Create: `benchmark/__init__.py`, `benchmark/spec/__init__.py`, `benchmark/harness/__init__.py`
- Create: `tests/__init__.py`, `tests/benchmark/__init__.py`, `tests/conftest.py`
- Create: `tests/benchmark/test_scaffolding.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `PROJECT_ROOT: pathlib.Path` and `GENERATOR_DIR: pathlib.Path` fixtures from `tests/conftest.py`, used by every later test module.

- [ ] **Step 1: Initialise git and install pytest**

```bash
cd /Users/user123/Desktop/projects/datascience-project
git init
synthetic_data_generator/.venv/bin/pip install pytest
```

- [ ] **Step 2: Write `.gitignore`**

Create `.gitignore`:

```gitignore
__pycache__/
*.pyc
.venv/
.idea/
.pytest_cache/

# Generator intermediates
synthetic_data_generator/raw_daily_wide.csv

# The benchmark is large and regenerable from benchmark/spec/. Seals and
# manifests ARE committed, so the frozen test set stays provable.
benchmark/datasets/*/*/media.csv
benchmark/datasets/*/*/sales.csv
benchmark/datasets/*/*/ground_truth.csv
benchmark/datasets/*/*/meta.json
benchmark/datasets/*/*/true_roi.csv
!benchmark/datasets/*/manifest.sha256
!benchmark/datasets/*/SEALED
```

- [ ] **Step 3: Write `pytest.ini`**

Create `pytest.ini`:

```ini
[pytest]
testpaths = tests
markers =
    slow: invokes the R simulator; takes tens of seconds
```

- [ ] **Step 4: Create the package skeleton**

```bash
cd /Users/user123/Desktop/projects/datascience-project
mkdir -p benchmark/spec benchmark/harness tests/benchmark
touch benchmark/__init__.py benchmark/spec/__init__.py benchmark/harness/__init__.py
touch tests/__init__.py tests/benchmark/__init__.py
```

- [ ] **Step 5: Write the failing test**

Create `tests/benchmark/test_scaffolding.py`:

```python
"""The scaffolding test exists so Task 1 has a real gate: it proves the
fixtures resolve and that the R toolchain this whole plan depends on is
actually present."""
import shutil
import subprocess


def test_fixtures_point_at_real_directories(project_root, generator_dir):
    assert (project_root / "pytest.ini").is_file()
    assert (generator_dir / "generate_with_simmmulator.R").is_file()
    assert (generator_dir / ".venv" / "bin" / "python").is_file()


def test_rscript_is_available_with_simmmulator():
    assert shutil.which("Rscript") is not None
    proc = subprocess.run(
        ["Rscript", "-e", 'cat("siMMMulator" %in% rownames(installed.packages()))'],
        capture_output=True, text=True, check=True,
    )
    assert proc.stdout.strip() == "TRUE"
```

- [ ] **Step 6: Run it to make sure it fails**

Run: `synthetic_data_generator/.venv/bin/python -m pytest tests/benchmark/test_scaffolding.py -v`
Expected: FAIL — `fixture 'project_root' not found`.

- [ ] **Step 7: Write `tests/conftest.py`**

```python
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GENERATOR_DIR = PROJECT_ROOT / "synthetic_data_generator"
VENV_PYTHON = GENERATOR_DIR / ".venv" / "bin" / "python"


@pytest.fixture
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture
def generator_dir() -> Path:
    return GENERATOR_DIR


@pytest.fixture
def venv_python() -> Path:
    return VENV_PYTHON
```

- [ ] **Step 8: Run the tests and make sure they pass**

Run: `synthetic_data_generator/.venv/bin/python -m pytest tests/benchmark/test_scaffolding.py -v`
Expected: 2 passed.

- [ ] **Step 9: Commit**

```bash
cd /Users/user123/Desktop/projects/datascience-project
git add .gitignore pytest.ini benchmark tests docs
git commit -m "chore: scaffold benchmark package, pytest, and design docs"
```

---

### Task 2: Parametrize the R generator

The generator currently hardcodes `set.seed(42)` and reads and writes the working directory. Without this task every scenario would carry identical noise.

**Files:**
- Create: `synthetic_data_generator/cli_args.R`
- Modify: `synthetic_data_generator/generate_with_simmmulator.R`
- Test: `tests/benchmark/test_r_cli.py`

**Interfaces:**
- Consumes: `project_root`, `generator_dir` fixtures from Task 1.
- Produces: the CLI contract `Rscript generate_with_simmmulator.R [--seed INT] [--config PATH] [--events PATH] [--outdir DIR]`, writing `<outdir>/raw_daily_wide.csv`. Defaults: `42`, `config.yaml`, `events_config.yaml`, `.`. Task 8 depends on this exact contract.

- [ ] **Step 1: Write the failing test**

Create `tests/benchmark/test_r_cli.py`:

```python
"""Argument parsing is tested by sourcing cli_args.R directly, so these tests
run in milliseconds instead of invoking siMMMulator."""
import json
import subprocess

import pytest


def _parse(generator_dir, argv):
    """Call parse_cli_args() in R and bring the result back as a dict."""
    quoted = ", ".join(f'"{a}"' for a in argv)
    script = (
        f'source("{generator_dir / "cli_args.R"}"); '
        f"o <- parse_cli_args(c({quoted})); "
        'cat(sprintf(\'{"seed":%d,"config":"%s","events":"%s","outdir":"%s"}\', '
        "o$seed, o$config, o$events, o$outdir))"
    )
    proc = subprocess.run(["Rscript", "-e", script], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr)
    return json.loads(proc.stdout)


def test_defaults_reproduce_original_behaviour(generator_dir):
    assert _parse(generator_dir, []) == {
        "seed": 42,
        "config": "config.yaml",
        "events": "events_config.yaml",
        "outdir": ".",
    }


def test_all_four_options_are_honoured(generator_dir):
    got = _parse(generator_dir, ["--seed", "1007", "--config", "/tmp/c.yaml",
                                 "--events", "/tmp/e.yaml", "--outdir", "/tmp/out"])
    assert got == {"seed": 1007, "config": "/tmp/c.yaml",
                   "events": "/tmp/e.yaml", "outdir": "/tmp/out"}


def test_unknown_option_is_rejected(generator_dir):
    with pytest.raises(RuntimeError, match="unknown option"):
        _parse(generator_dir, ["--nope", "1"])


def test_option_without_value_is_rejected(generator_dir):
    with pytest.raises(RuntimeError, match="needs a value"):
        _parse(generator_dir, ["--seed"])


def test_non_integer_seed_is_rejected(generator_dir):
    with pytest.raises(RuntimeError, match="must be an integer"):
        _parse(generator_dir, ["--seed", "abc"])
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `synthetic_data_generator/.venv/bin/python -m pytest tests/benchmark/test_r_cli.py -v`
Expected: all 5 FAIL — `cannot open file '.../cli_args.R'`.

- [ ] **Step 3: Write `synthetic_data_generator/cli_args.R`**

```r
# Command-line argument parsing for generate_with_simmmulator.R.
#
# Kept in its own file so it can be sourced and tested without running a
# simulation. Uses only base R -- the generator gains no new package
# dependency.
#
# All options are optional, and the defaults reproduce the script's original
# behaviour exactly, so `Rscript generate_with_simmmulator.R` is unchanged.
#
#   --seed    integer RNG seed                        (default 42)
#   --config  path to the simulation config           (default config.yaml)
#   --events  path to the injected-pattern config     (default events_config.yaml)
#   --outdir  directory for raw_daily_wide.csv        (default .)

parse_cli_args <- function(argv) {
  opts <- list(
    seed = "42",
    config = "config.yaml",
    events = "events_config.yaml",
    outdir = "."
  )

  i <- 1
  while (i <= length(argv)) {
    if (!startsWith(argv[i], "--")) {
      stop(sprintf("expected an option starting with '--', got '%s'", argv[i]))
    }
    key <- substring(argv[i], 3)
    if (!key %in% names(opts)) {
      stop(sprintf("unknown option '--%s'; known options are: %s",
                   key, paste(names(opts), collapse = ", ")))
    }
    if (i + 1 > length(argv)) {
      stop(sprintf("option '--%s' needs a value", key))
    }
    opts[[key]] <- argv[i + 1]
    i <- i + 2
  }

  seed <- suppressWarnings(as.integer(opts$seed))
  if (is.na(seed)) stop("--seed must be an integer")
  opts$seed <- seed

  opts
}
```

- [ ] **Step 4: Run the tests and make sure they pass**

Run: `synthetic_data_generator/.venv/bin/python -m pytest tests/benchmark/test_r_cli.py -v`
Expected: 5 passed.

- [ ] **Step 5: Wire the parser into the generator**

In `synthetic_data_generator/generate_with_simmmulator.R`, replace this block:

```r
set.seed(42)

# ---------------------------------------------------------------------------
# Config -- everything simulation-wide lives in config.yaml, informative
# periods live in events_config.yaml. See DETAILS.md for the full reference.
# ---------------------------------------------------------------------------
config <- yaml::read_yaml("config.yaml")
events <- yaml::read_yaml("events_config.yaml")
```

with:

```r
# Resolve this script's own directory so cli_args.R can be sourced and the
# script can be invoked from any working directory.
.this_file <- sub("^--file=", "",
                  grep("^--file=", commandArgs(FALSE), value = TRUE)[1])
.script_dir <- if (is.na(.this_file)) "." else dirname(normalizePath(.this_file))
source(file.path(.script_dir, "cli_args.R"))

opts <- parse_cli_args(commandArgs(trailingOnly = TRUE))

set.seed(opts$seed)

# ---------------------------------------------------------------------------
# Config -- everything simulation-wide lives in config.yaml, informative
# periods live in events_config.yaml. See DETAILS.md for the full reference.
# ---------------------------------------------------------------------------
config <- yaml::read_yaml(opts$config)
events <- yaml::read_yaml(opts$events)
```

- [ ] **Step 6: Honour `--outdir` on write**

In the same file, replace:

```r
write.csv(all_countries_df, "raw_daily_wide.csv", row.names = FALSE)
```

with:

```r
dir.create(opts$outdir, showWarnings = FALSE, recursive = TRUE)
raw_path <- file.path(opts$outdir, "raw_daily_wide.csv")
write.csv(all_countries_df, raw_path, row.names = FALSE)
```

and update the closing message to report `raw_path` rather than the bare filename:

```r
cat("\nDone. Wrote", raw_path, "(", nrow(all_countries_df), "rows ).\n")
```

- [ ] **Step 7: Write the end-to-end seed test**

Append to `tests/benchmark/test_r_cli.py`:

```python
@pytest.mark.slow
def test_seed_changes_the_data_and_is_reproducible(generator_dir, tmp_path):
    """Two runs at the same seed must be byte-identical; a different seed must
    produce different data. Without this, every benchmark scenario of the same
    shape would carry identical noise."""
    tiny_config = tmp_path / "config.yaml"
    tiny_config.write_text((generator_dir / "config.yaml").read_text()
                           .replace("years: 2", "years: 1"))
    no_events = tmp_path / "events.yaml"
    no_events.write_text("[]\n")

    outs = []
    for name, seed in [("a", "7"), ("b", "7"), ("c", "8")]:
        outdir = tmp_path / name
        subprocess.run(
            ["Rscript", str(generator_dir / "generate_with_simmmulator.R"),
             "--seed", seed, "--config", str(tiny_config),
             "--events", str(no_events), "--outdir", str(outdir)],
            check=True, capture_output=True, text=True,
        )
        outs.append((outdir / "raw_daily_wide.csv").read_bytes())

    assert outs[0] == outs[1], "same seed must reproduce byte-identically"
    assert outs[0] != outs[2], "different seed must produce different data"
```

- [ ] **Step 8: Run the slow test**

Run: `synthetic_data_generator/.venv/bin/python -m pytest tests/benchmark/test_r_cli.py -v -m slow`
Expected: PASS. Takes roughly 6 minutes — the tiny config still carries 5 countries for 1 year, three times over.

- [ ] **Step 9: Verify backward compatibility by hand**

```bash
cd /Users/user123/Desktop/projects/datascience-project/synthetic_data_generator
cp raw_daily_wide.csv /tmp/raw_before.csv
Rscript generate_with_simmmulator.R
diff -q /tmp/raw_before.csv raw_daily_wide.csv && echo "IDENTICAL — backward compatible"
```
Expected: `IDENTICAL — backward compatible`. The committed `raw_daily_wide.csv` was produced at seed 42 from the committed configs, so the patched script must reproduce it exactly.

- [ ] **Step 10: Commit**

```bash
cd /Users/user123/Desktop/projects/datascience-project
git add synthetic_data_generator/cli_args.R \
        synthetic_data_generator/generate_with_simmmulator.R \
        tests/benchmark/test_r_cli.py
git commit -m "feat(generator): add --seed/--config/--events/--outdir to the R generator"
```

---

### Task 3: Parametrize `reformat.py`

**Files:**
- Modify: `synthetic_data_generator/reformat.py`
- Test: `tests/benchmark/test_reformat_cli.py`

**Interfaces:**
- Consumes: the `--outdir` contract from Task 2 — `reformat.py` reads `<outdir>/raw_daily_wide.csv`.
- Produces: `python reformat.py [--config PATH] [--events PATH] [--outdir DIR]`, writing `media.csv`, `sales.csv`, `ground_truth.csv`, `true_roi.csv` into `<outdir>`. Task 8 depends on this.

- [ ] **Step 1: Write the failing test**

Create `tests/benchmark/test_reformat_cli.py`:

```python
import subprocess

import pandas as pd


def _run(venv_python, generator_dir, tmp_path, events_yaml):
    """Reformat a two-row slice of the committed raw output into tmp_path."""
    raw = pd.read_csv(generator_dir / "raw_daily_wide.csv")
    raw.groupby("country_code").head(3).to_csv(tmp_path / "raw_daily_wide.csv", index=False)
    events = tmp_path / "events.yaml"
    events.write_text(events_yaml)
    subprocess.run(
        [str(venv_python), str(generator_dir / "reformat.py"),
         "--config", str(generator_dir / "config.yaml"),
         "--events", str(events), "--outdir", str(tmp_path)],
        check=True, capture_output=True, text=True,
    )
    return tmp_path


def test_writes_all_four_outputs_into_outdir(venv_python, generator_dir, tmp_path):
    out = _run(venv_python, generator_dir, tmp_path, "[]\n")
    for name in ["media.csv", "sales.csv", "ground_truth.csv", "true_roi.csv"]:
        assert (out / name).is_file(), f"{name} missing from --outdir"


def test_empty_events_yields_empty_ground_truth(venv_python, generator_dir, tmp_path):
    """Null scenarios carry no events; ground_truth.csv must still be a valid
    CSV with headers so the eval loader does not special-case it."""
    out = _run(venv_python, generator_dir, tmp_path, "[]\n")
    gt = pd.read_csv(out / "ground_truth.csv")
    assert len(gt) == 0
    assert list(gt.columns) == ["pattern_id", "pattern_type", "country_code",
                                "channel", "start_date", "end_date",
                                "multiplier", "description"]


def test_events_reach_ground_truth(venv_python, generator_dir, tmp_path):
    out = _run(venv_python, generator_dir, tmp_path, """
- pattern_id: T_HOLD_01
  pattern_type: natural_holdout
  country: DE
  channel: Radio
  start_day: 0
  end_day: 2
  multiplier: 0
  description: test holdout
""")
    gt = pd.read_csv(out / "ground_truth.csv")
    assert gt.loc[0, "pattern_id"] == "T_HOLD_01"
    assert gt.loc[0, "country_code"] == "DE"
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `synthetic_data_generator/.venv/bin/python -m pytest tests/benchmark/test_reformat_cli.py -v`
Expected: FAIL — `unrecognized arguments: --config`.

- [ ] **Step 3: Add argparse to `reformat.py`**

Add `import argparse` and `from pathlib import Path` to the imports, then replace the whole `main()` function's opening and its four `to_csv` calls. Replace:

```python
def main():
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    with open("events_config.yaml") as f:
        events = pd.DataFrame(yaml.safe_load(f))

    raw = pd.read_csv("raw_daily_wide.csv", parse_dates=["DATE"])
```

with:

```python
def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config.yaml",
                   help="path to the simulation config (default: config.yaml)")
    p.add_argument("--events", default="events_config.yaml",
                   help="path to the injected-pattern config "
                        "(default: events_config.yaml)")
    p.add_argument("--outdir", default=".",
                   help="directory holding raw_daily_wide.csv, and where the "
                        "reshaped CSVs are written (default: .)")
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        config = yaml.safe_load(f)
    with open(args.events) as f:
        raw_events = yaml.safe_load(f) or []
    events = pd.DataFrame(raw_events, columns=[
        "pattern_id", "pattern_type", "country", "channel",
        "start_day", "end_day", "multiplier", "description",
    ])

    raw = pd.read_csv(outdir / "raw_daily_wide.csv", parse_dates=["DATE"])
```

The explicit `columns=` argument matters: a null scenario has an empty event list, and `pd.DataFrame([])` has no columns at all, which would crash `build_ground_truth`.

- [ ] **Step 4: Make `build_ground_truth` tolerate an empty frame**

Replace the `return` line of `build_ground_truth` with:

```python
    return pd.DataFrame(rows, columns=[
        "pattern_id", "pattern_type", "country_code", "channel",
        "start_date", "end_date", "multiplier", "description",
    ])
```

- [ ] **Step 5: Route the four writes through `outdir`**

Replace:

```python
    media_df.to_csv("media.csv", index=False)
    sales_df.to_csv("sales.csv", index=False)
    ground_truth_df.to_csv("ground_truth.csv", index=False)
```

with:

```python
    media_df.to_csv(outdir / "media.csv", index=False)
    sales_df.to_csv(outdir / "sales.csv", index=False)
    ground_truth_df.to_csv(outdir / "ground_truth.csv", index=False)
```

and replace:

```python
    roi.reset_index().to_csv("true_roi.csv", index=False)
```

with:

```python
    roi.reset_index().to_csv(outdir / "true_roi.csv", index=False)
```

- [ ] **Step 6: Run the tests and make sure they pass**

Run: `synthetic_data_generator/.venv/bin/python -m pytest tests/benchmark/test_reformat_cli.py -v`
Expected: 3 passed.

- [ ] **Step 7: Verify backward compatibility by hand**

```bash
cd /Users/user123/Desktop/projects/datascience-project/synthetic_data_generator
cp media.csv /tmp/media_before.csv
.venv/bin/python reformat.py
diff -q /tmp/media_before.csv media.csv && echo "IDENTICAL — backward compatible"
```
Expected: `IDENTICAL — backward compatible`.

- [ ] **Step 8: Commit**

```bash
cd /Users/user123/Desktop/projects/datascience-project
git add synthetic_data_generator/reformat.py tests/benchmark/test_reformat_cli.py
git commit -m "feat(generator): add --config/--events/--outdir to reformat.py"
```

---

### Task 4: Variation axes

The pools and presets every scenario is drawn from. Isolated from scenario assembly so the two can be tested separately.

**Files:**
- Create: `benchmark/spec/axes.py`
- Test: `tests/benchmark/test_axes.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `COUNTRY_POOL: list[tuple[str, str]]` — 10 `(code, name)` pairs
  - `CHANNEL_POOL: list[dict]` — 12 channel templates, each a complete `config.yaml` channel entry minus `spend_share_min`/`spend_share_max`
  - `NOISE_PRESETS: dict[str, dict]` keyed `"low"`, `"med"`, `"high"`
  - `MARKET_SPREAD_PRESETS: dict[str, tuple[float, float]]` keyed `"tight"`, `"moderate"`, `"extreme"`
  - `pick_countries(rng, n, spread) -> list[dict]` — each dict has `code`, `name`, `market_size`
  - `pick_channels(rng, n) -> list[dict]` — complete channel entries with valid spend shares
  - `baseline_for(noise_level, trend_p, temp_var) -> dict`

- [ ] **Step 1: Write the failing test**

Create `tests/benchmark/test_axes.py`:

```python
import numpy as np
import pytest

from benchmark.spec import axes


def rng(seed=0):
    return np.random.default_rng(seed)


def test_pools_are_large_enough_for_the_widest_scenarios():
    assert len(axes.COUNTRY_POOL) >= 8
    assert len(axes.CHANNEL_POOL) >= 12


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
def test_pick_countries_returns_n_distinct_markets(n):
    got = axes.pick_countries(rng(), n, "moderate")
    assert len(got) == n
    assert len({c["code"] for c in got}) == n
    assert all(c["market_size"] > 0 for c in got)


def test_extreme_spread_really_is_fifteen_fold():
    got = axes.pick_countries(rng(), 5, "extreme")
    sizes = [c["market_size"] for c in got]
    assert max(sizes) / min(sizes) == pytest.approx(15.0, rel=1e-6)


def test_tight_spread_keeps_markets_comparable():
    got = axes.pick_countries(rng(), 5, "tight")
    sizes = [c["market_size"] for c in got]
    assert max(sizes) / min(sizes) < 1.3


@pytest.mark.parametrize("n", [1, 2, 4, 6, 9, 12])
def test_pick_channels_produces_a_config_the_generator_accepts(n):
    got = axes.pick_channels(rng(), n)
    assert len(got) == n
    assert len({c["name"] for c in got}) == n

    # Every channel except the last needs min/max; the last takes the remainder.
    for ch in got[:-1]:
        assert 0 < ch["spend_share_min"] < ch["spend_share_max"] < 1
    assert "spend_share_min" not in got[-1]

    # The generator's contract: the maxima must leave room for the last channel.
    assert sum(c["spend_share_max"] for c in got[:-1]) < 0.92

    required = {"name", "type", "platform", "true_cvr", "decay",
                "alpha_saturation", "gamma_saturation", "assumed_ctr"}
    for ch in got:
        assert required <= set(ch)
        assert ("true_cpm" in ch) == (ch["type"] == "impression")
        assert ("true_cpc" in ch) == (ch["type"] == "click")


def test_pick_channels_is_deterministic_for_a_given_seed():
    assert axes.pick_channels(rng(5), 6) == axes.pick_channels(rng(5), 6)


def test_noise_presets_increase_monotonically():
    lo, med, hi = (axes.NOISE_PRESETS[k] for k in ["low", "med", "high"])
    assert lo["error_std"] < med["error_std"] < hi["error_std"]
    assert lo["cvr_mult"] < med["cvr_mult"] < hi["cvr_mult"]


def test_baseline_for_threads_trend_and_seasonality_through():
    got = axes.baseline_for("high", trend_p=1.0, temp_var=5)
    assert got["trend_p"] == 1.0
    assert got["temp_var"] == 5
    assert got["error_std"] == axes.NOISE_PRESETS["high"]["error_std"]
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `synthetic_data_generator/.venv/bin/python -m pytest tests/benchmark/test_axes.py -v`
Expected: FAIL — `No module named 'benchmark.spec.axes'`.

- [ ] **Step 3: Write `benchmark/spec/axes.py`**

```python
"""Pools and presets that benchmark scenarios are drawn from.

Nothing here knows what an event is or how many scenarios exist. This module
answers only: what markets, what channels, how noisy, how spread out.
"""
from __future__ import annotations

import numpy as np

COUNTRY_POOL: list[tuple[str, str]] = [
    ("DE", "Germany"), ("AT", "Austria"), ("CH", "Switzerland"),
    ("US", "United States"), ("FI", "Finland"), ("SE", "Sweden"),
    ("NO", "Norway"), ("NL", "Netherlands"), ("FR", "France"),
    ("PL", "Poland"),
]

# Twelve channel templates spanning both siMMMulator types. Decay rates are
# chosen to span the realistic range: TV lingers, Search does not. Templates
# carry no spend shares -- pick_channels assigns those, because valid shares
# depend on how many channels a scenario ends up with.
CHANNEL_POOL: list[dict] = [
    dict(name="TV", type="impression", platform="TV", true_cvr=0.00003,
         true_cpm=5.0, mean_noisy_cpm_cpc=0.0, std_noisy_cpm_cpc=0.3,
         mean_noisy_cvr=0.0, std_noisy_cvr=0.00001, decay=0.55,
         alpha_saturation=2.0, gamma_saturation=0.4, assumed_ctr=0.001),
    dict(name="Radio", type="impression", platform="Radio", true_cvr=0.00002,
         true_cpm=3.0, mean_noisy_cpm_cpc=0.0, std_noisy_cpm_cpc=0.2,
         mean_noisy_cvr=0.0, std_noisy_cvr=0.000008, decay=0.35,
         alpha_saturation=2.0, gamma_saturation=0.3, assumed_ctr=0.0008),
    dict(name="OOH", type="impression", platform="OOH", true_cvr=0.000015,
         true_cpm=4.0, mean_noisy_cpm_cpc=0.0, std_noisy_cpm_cpc=0.25,
         mean_noisy_cvr=0.0, std_noisy_cvr=0.000006, decay=0.45,
         alpha_saturation=2.0, gamma_saturation=0.35, assumed_ctr=0.0005),
    dict(name="Print", type="impression", platform="Print", true_cvr=0.000012,
         true_cpm=6.0, mean_noisy_cpm_cpc=0.0, std_noisy_cpm_cpc=0.3,
         mean_noisy_cvr=0.0, std_noisy_cvr=0.000005, decay=0.40,
         alpha_saturation=2.0, gamma_saturation=0.35, assumed_ctr=0.0004),
    dict(name="Cinema", type="impression", platform="Cinema", true_cvr=0.00002,
         true_cpm=9.0, mean_noisy_cpm_cpc=0.0, std_noisy_cpm_cpc=0.4,
         mean_noisy_cvr=0.0, std_noisy_cvr=0.000008, decay=0.50,
         alpha_saturation=2.0, gamma_saturation=0.4, assumed_ctr=0.0003),
    dict(name="Google Discovery", type="impression", platform="Google Ads",
         true_cvr=0.00006, true_cpm=8.0, mean_noisy_cpm_cpc=0.0,
         std_noisy_cpm_cpc=0.5, mean_noisy_cvr=0.0, std_noisy_cvr=0.00002,
         decay=0.20, alpha_saturation=2.0, gamma_saturation=0.3,
         assumed_ctr=0.015),
    dict(name="Facebook", type="impression", platform="Meta", true_cvr=0.00005,
         true_cpm=10.0, mean_noisy_cpm_cpc=0.0, std_noisy_cpm_cpc=0.5,
         mean_noisy_cvr=0.0, std_noisy_cvr=0.000015, decay=0.30,
         alpha_saturation=2.0, gamma_saturation=0.3, assumed_ctr=0.012),
    dict(name="Instagram", type="impression", platform="Meta", true_cvr=0.00004,
         true_cpm=12.0, mean_noisy_cpm_cpc=0.0, std_noisy_cpm_cpc=0.6,
         mean_noisy_cvr=0.0, std_noisy_cvr=0.000012, decay=0.25,
         alpha_saturation=2.0, gamma_saturation=0.3, assumed_ctr=0.010),
    dict(name="TikTok", type="impression", platform="TikTok", true_cvr=0.000035,
         true_cpm=7.0, mean_noisy_cpm_cpc=0.0, std_noisy_cpm_cpc=0.7,
         mean_noisy_cvr=0.0, std_noisy_cvr=0.000014, decay=0.18,
         alpha_saturation=2.0, gamma_saturation=0.3, assumed_ctr=0.009),
    dict(name="YouTube", type="impression", platform="Google Ads",
         true_cvr=0.000025, true_cpm=11.0, mean_noisy_cpm_cpc=0.0,
         std_noisy_cpm_cpc=0.5, mean_noisy_cvr=0.0, std_noisy_cvr=0.00001,
         decay=0.35, alpha_saturation=2.0, gamma_saturation=0.35,
         assumed_ctr=0.004),
    dict(name="Google Search", type="click", platform="Google Ads",
         true_cvr=0.02, true_cpc=0.8, mean_noisy_cpm_cpc=0.0,
         std_noisy_cpm_cpc=0.05, mean_noisy_cvr=0.0, std_noisy_cvr=0.005,
         decay=0.10, alpha_saturation=2.0, gamma_saturation=0.2,
         assumed_ctr=0.05),
    dict(name="Affiliate", type="click", platform="Affiliate", true_cvr=0.03,
         true_cpc=0.5, mean_noisy_cpm_cpc=0.0, std_noisy_cpm_cpc=0.04,
         mean_noisy_cvr=0.0, std_noisy_cvr=0.006, decay=0.12,
         alpha_saturation=2.0, gamma_saturation=0.2, assumed_ctr=0.08),
]

# error_std and temp_coef_sd feed step_1_create_baseline; the multipliers scale
# every channel's std_noisy_cpm_cpc and std_noisy_cvr.
NOISE_PRESETS: dict[str, dict] = {
    "low":  dict(error_std=50.0,  temp_coef_sd=250.0,  cpm_cpc_mult=0.5, cvr_mult=0.5),
    "med":  dict(error_std=100.0, temp_coef_sd=500.0,  cpm_cpc_mult=1.0, cvr_mult=1.0),
    "high": dict(error_std=300.0, temp_coef_sd=1500.0, cpm_cpc_mult=2.0, cvr_mult=2.5),
}

# (smallest market_size, largest market_size)
MARKET_SPREAD_PRESETS: dict[str, tuple[float, float]] = {
    "tight":    (0.90, 1.10),
    "moderate": (0.20, 1.30),
    "extreme":  (0.10, 1.50),
}

TREND_LEVELS: tuple[float, ...] = (0.0, 0.5, 1.0)
SEASONALITY_LEVELS: tuple[float, ...] = (0.5, 2.0, 5.0)
NOISE_LEVELS: tuple[str, ...] = ("low", "med", "high")


def pick_countries(rng: np.random.Generator, n: int, spread: str) -> list[dict]:
    """Choose n distinct markets and assign market sizes spanning `spread`.

    The extreme and tight endpoints are pinned to the preset bounds rather than
    sampled, so a scenario labelled "extreme" is guaranteed to exercise the full
    ratio instead of happening to draw two similar sizes.
    """
    lo, hi = MARKET_SPREAD_PRESETS[spread]
    idx = rng.choice(len(COUNTRY_POOL), size=n, replace=False)
    chosen = [COUNTRY_POOL[i] for i in idx]

    if n == 1:
        sizes = [1.0]
    else:
        # Log-spaced between the bounds, then shuffled so market size is not
        # correlated with position in the list.
        sizes = list(np.exp(np.linspace(np.log(lo), np.log(hi), n)))
        rng.shuffle(sizes)

    return [{"code": c, "name": name, "market_size": round(float(s), 4)}
            for (c, name), s in zip(chosen, sizes)]


def pick_channels(rng: np.random.Generator, n: int) -> list[dict]:
    """Choose n distinct channels and give them valid spend shares.

    The generator requires spend_share_min/max on every channel except the last,
    which receives whatever budget remains. The maxima must therefore leave
    headroom, or the last channel's share can go negative.
    """
    idx = rng.choice(len(CHANNEL_POOL), size=n, replace=False)
    chosen = [dict(CHANNEL_POOL[i]) for i in idx]

    if n == 1:
        return chosen

    # Dirichlet gives a random but sane budget split. Scale it so the first
    # n-1 channels claim at most 80% of the budget, leaving the last a real
    # share and keeping the sum of maxima under the 0.92 headroom limit.
    weights = rng.dirichlet(np.ones(n))
    weights = weights / weights.sum() * 0.80

    for ch, w in zip(chosen[:-1], weights[:-1]):
        mid = float(w)
        ch["spend_share_min"] = round(max(0.01, mid * 0.85), 4)
        ch["spend_share_max"] = round(mid * 1.15, 4)

    return chosen


def apply_noise(channels: list[dict], noise_level: str) -> list[dict]:
    """Scale per-channel noise to the scenario's noise level."""
    preset = NOISE_PRESETS[noise_level]
    out = []
    for ch in channels:
        ch = dict(ch)
        ch["std_noisy_cpm_cpc"] = round(ch["std_noisy_cpm_cpc"] * preset["cpm_cpc_mult"], 8)
        ch["std_noisy_cvr"] = round(ch["std_noisy_cvr"] * preset["cvr_mult"], 10)
        out.append(ch)
    return out


def baseline_for(noise_level: str, trend_p: float, temp_var: float) -> dict:
    """Build the config.yaml `baseline` block for a scenario."""
    preset = NOISE_PRESETS[noise_level]
    return dict(
        daily_mean=15000.0,
        trend_p=float(trend_p),
        temp_var=float(temp_var),
        temp_coef_mean=100.0,
        temp_coef_sd=preset["temp_coef_sd"],
        error_std=preset["error_std"],
    )
```

- [ ] **Step 4: Run the tests and make sure they pass**

Run: `synthetic_data_generator/.venv/bin/python -m pytest tests/benchmark/test_axes.py -v`
Expected: 17 passed.

- [ ] **Step 5: Verify a 1-channel config is actually accepted by siMMMulator**

The generator builds `MAX_MIN_PROPORTION` from all channels except the last. With one channel that vector is empty, which may or may not pass siMMMulator's input checks. Find out now rather than during the 45-minute generation run:

```bash
cd /tmp && rm -rf onech && mkdir onech && cd onech
/Users/user123/Desktop/projects/datascience-project/synthetic_data_generator/.venv/bin/python - <<'PY'
import yaml, sys
sys.path.insert(0, "/Users/user123/Desktop/projects/datascience-project")
import numpy as np
from benchmark.spec import axes
cfg = dict(years=1, start_date="2024/01/01", revenue_per_conv=40.0,
           customer_types=["New"], sales_channels=["Ecom"],
           baseline=axes.baseline_for("med", 0.5, 2.0),
           campaign_spend=dict(daily_total_mean=18500.0, daily_total_std=4000.0),
           countries=axes.pick_countries(np.random.default_rng(0), 1, "tight"),
           channels=axes.pick_channels(np.random.default_rng(0), 1))
yaml.safe_dump(cfg, open("config.yaml", "w"), sort_keys=False)
open("events.yaml", "w").write("[]\n")
PY
Rscript /Users/user123/Desktop/projects/datascience-project/synthetic_data_generator/generate_with_simmmulator.R \
  --seed 1 --config config.yaml --events events.yaml --outdir . 2>&1 | tail -5
```

If this succeeds, record in `axes.pick_channels`' docstring that a single-channel config is accepted, and move on.

If siMMMulator rejects the empty proportion vector, nothing needs changing — Task 6 already emulates the case rather than relying on `n_channels == 1`. Its `single_channel_market` edge case forces `force_channels=2` and has `_events_edge` hold every channel but the first at zero for the whole series, so the detector sees a market with exactly one active channel. Add a note to `pick_channels`' docstring recording that `n == 1` is unsupported by the simulator, and drop `1` from the `n_channels` choices in `scenarios._shape`.

- [ ] **Step 6: Commit**

```bash
cd /Users/user123/Desktop/projects/datascience-project
git add benchmark/spec/axes.py tests/benchmark/test_axes.py
git commit -m "feat(benchmark): add scenario variation axes -- country, channel, noise pools"
```

---

### Task 5: Event recipes

One builder per event family, each returning a list of `events_config.yaml` entries. Kept separate from scenario assembly so the injection semantics can be tested without generating anything.

**Files:**
- Create: `benchmark/spec/events.py`
- Test: `tests/benchmark/test_events.py`

**Interfaces:**
- Consumes: nothing.
- Produces: builders `dark`, `single_channel`, `holdout`, `step`, `pulse`, `launch`, `ramp`, `intermittent`, `global_pause`, each returning `list[dict]` with keys `pattern_id, pattern_type, country, channel, start_day, end_day, multiplier, description`. Also `NON_EVENT_TYPES: frozenset[str]` — pattern types that are deliberate negative controls and must never be matched as detections. Tasks 6 and Plan 2's truth loader both consume `NON_EVENT_TYPES`.

- [ ] **Step 1: Write the failing test**

Create `tests/benchmark/test_events.py`:

```python
import pytest

from benchmark.spec import events

REQUIRED_KEYS = {"pattern_id", "pattern_type", "country", "channel",
                 "start_day", "end_day", "multiplier", "description"}


def _check_shape(entries, n_days=730):
    assert entries, "builder produced no entries"
    for e in entries:
        assert set(e) == REQUIRED_KEYS, f"bad keys: {set(e) ^ REQUIRED_KEYS}"
        assert 0 <= e["start_day"] < e["end_day"] <= n_days
        assert isinstance(e["pattern_id"], str) and e["pattern_id"]


def test_dark_targets_every_channel():
    got = events.dark("DE", start=100, length=42)
    _check_shape(got)
    assert len(got) == 1
    assert got[0]["pattern_type"] == "dark_period"
    assert got[0]["channel"] == "ALL"
    assert got[0]["multiplier"] == 0
    assert got[0]["end_day"] - got[0]["start_day"] == 42


def test_single_channel_names_the_channel_that_stays_on():
    got = events.single_channel("FI", keep="Google Search", start=250, length=30)
    _check_shape(got)
    assert got[0]["pattern_type"] == "single_channel"
    assert got[0]["channel"] == "Google Search"


def test_holdout_names_the_channel_that_goes_off():
    got = events.holdout("AT", "Facebook", start=400, length=42)
    _check_shape(got)
    assert got[0]["pattern_type"] == "natural_holdout"
    assert got[0]["channel"] == "Facebook"
    assert got[0]["multiplier"] == 0


def test_holdout_supports_near_zero_spend():
    got = events.holdout("AT", "Facebook", start=400, length=42, multiplier=0.04)
    assert got[0]["multiplier"] == 0.04


def test_step_records_its_multiplier():
    for mult in [0.33, 1.5, 3, 5]:
        got = events.step("CH", "Google Search", start=500, length=56, multiplier=mult)
        _check_shape(got)
        assert got[0]["pattern_type"] == "step_change"
        assert got[0]["multiplier"] == mult


def test_pulse_emits_one_entry_per_off_window_with_unique_ids():
    got = events.pulse("DE", "Radio", starts=[40, 68, 96], length=14)
    _check_shape(got)
    assert len(got) == 3
    assert len({e["pattern_id"] for e in got}) == 3
    assert all(e["pattern_type"] == "channel_pulse" for e in got)
    assert [e["start_day"] for e in got] == [40, 68, 96]


def test_launch_is_anchored_at_day_zero():
    got = events.launch("US", "Instagram", length=90)
    _check_shape(got)
    assert got[0]["start_day"] == 0
    assert got[0]["pattern_type"] == "staggered_launch"


def test_global_pause_covers_every_country():
    got = events.global_pause(["DE", "AT", "CH"], start=300, length=21)
    _check_shape(got)
    assert len(got) == 3
    assert {e["country"] for e in got} == {"DE", "AT", "CH"}
    assert all(e["pattern_type"] == "global_pause" for e in got)


def test_ramp_is_a_negative_control_made_of_increasing_blocks():
    got = events.ramp("SE", "TV", start=200, block=10,
                      multipliers=[1.2, 1.4, 1.6, 1.8, 2.0])
    _check_shape(got)
    assert len(got) == 5
    assert [e["multiplier"] for e in got] == [1.2, 1.4, 1.6, 1.8, 2.0]
    # Contiguous blocks, no gaps and no overlaps.
    assert [(e["start_day"], e["end_day"]) for e in got] == [
        (200, 210), (210, 220), (220, 230), (230, 240), (240, 250)]
    assert got[0]["pattern_type"] in events.NON_EVENT_TYPES


def test_intermittent_is_a_negative_control_of_many_short_gaps():
    got = events.intermittent("NL", "TikTok", n=20, off_len=3, period=14, start=30)
    _check_shape(got)
    assert len(got) == 20
    assert all(e["end_day"] - e["start_day"] == 3 for e in got)
    assert got[0]["pattern_type"] in events.NON_EVENT_TYPES


def test_negative_controls_are_exactly_the_two_intended_types():
    assert events.NON_EVENT_TYPES == frozenset({"ramp_block", "intermittent_baseline"})


def test_ids_are_unique_across_a_mixed_scenario():
    entries = (events.dark("DE", 100, 42)
               + events.holdout("AT", "Facebook", 400, 42)
               + events.pulse("DE", "Radio", [40, 68, 96], 14)
               + events.step("CH", "Google Search", 500, 56, 3))
    ids = [e["pattern_id"] for e in entries]
    assert len(ids) == len(set(ids))


def test_length_beyond_the_series_is_rejected():
    with pytest.raises(ValueError, match="exceeds"):
        events.dark("DE", start=700, length=60, n_days=730)
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `synthetic_data_generator/.venv/bin/python -m pytest tests/benchmark/test_events.py -v`
Expected: FAIL — `No module named 'benchmark.spec.events'`.

- [ ] **Step 3: Write `benchmark/spec/events.py`**

```python
"""Builders for events_config.yaml entries, one function per event family.

Every builder returns a list of dicts in the generator's own schema, so the
output can be dumped straight to YAML. Nothing here reads config.yaml -- the
caller is responsible for passing country codes and channel names that exist.

Two pattern types are *negative controls*: they change the data but are not
events a detector should report. They are listed in NON_EVENT_TYPES, and the
evaluation truth loader drops them from the matchable set, so anything detected
inside their windows counts as a false positive -- which is exactly what should
happen for a gradual ramp or a naturally intermittent channel.
"""
from __future__ import annotations

NON_EVENT_TYPES: frozenset[str] = frozenset({"ramp_block", "intermittent_baseline"})

DEFAULT_N_DAYS = 730


def _entry(pattern_id, pattern_type, country, channel, start_day, end_day,
           multiplier, description, n_days):
    if end_day > n_days:
        raise ValueError(
            f"{pattern_id}: window ends on day {end_day}, which exceeds the "
            f"{n_days}-day series")
    if start_day < 0 or start_day >= end_day:
        raise ValueError(f"{pattern_id}: invalid window [{start_day}, {end_day})")
    return dict(pattern_id=pattern_id, pattern_type=pattern_type,
                country=country, channel=channel, start_day=int(start_day),
                end_day=int(end_day), multiplier=multiplier,
                description=description)


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text).upper().strip("_")


def dark(country, start, length, n_days=DEFAULT_N_DAYS):
    """Every channel in one country goes to zero."""
    return [_entry(f"{country}_DARK_{start}", "dark_period", country, "ALL",
                   start, start + length, 0,
                   f"All {country} advertising is off for {length} days.",
                   n_days)]


def single_channel(country, keep, start, length, n_days=DEFAULT_N_DAYS):
    """Only `keep` stays on; every other channel in the country goes to zero."""
    return [_entry(f"{country}_SINGLE_{_slug(keep)}_{start}", "single_channel",
                   country, keep, start, start + length, 0,
                   f"Only {keep} remains active in {country} for {length} days.",
                   n_days)]


def holdout(country, channel, start, length, multiplier=0, n_days=DEFAULT_N_DAYS):
    """One channel drops out; every other channel continues untouched."""
    depth = "to zero" if multiplier == 0 else f"to {multiplier:g}x"
    return [_entry(f"{country}_HOLD_{_slug(channel)}_{start}", "natural_holdout",
                   country, channel, start, start + length, multiplier,
                   f"{channel} drops {depth} in {country} for {length} days "
                   f"while other channels continue.", n_days)]


def step(country, channel, start, length, multiplier, n_days=DEFAULT_N_DAYS):
    """One channel's budget jumps by `multiplier` and holds."""
    direction = "rises" if multiplier > 1 else "falls"
    return [_entry(f"{country}_STEP_{_slug(channel)}_{start}", "step_change",
                   country, channel, start, start + length, multiplier,
                   f"{channel} spend {direction} to {multiplier:g}x in "
                   f"{country} for {length} days.", n_days)]


def pulse(country, channel, starts, length, n_days=DEFAULT_N_DAYS):
    """On/off/on. The only place adstock decay is observable."""
    return [
        _entry(f"{country}_PULSE_{_slug(channel)}_{i + 1}", "channel_pulse",
               country, channel, s, s + length, 0,
               f"{channel} off for {length} days in {country} "
               f"(pulse {i + 1} of {len(starts)}).", n_days)
        for i, s in enumerate(starts)
    ]


def launch(country, channel, length, n_days=DEFAULT_N_DAYS):
    """A channel is dormant at the start of the series, then launches."""
    return [_entry(f"{country}_LAUNCH_{_slug(channel)}", "staggered_launch",
                   country, channel, 0, length, 0,
                   f"{channel} launches in {country} {length} days into the "
                   f"series.", n_days)]


def global_pause(countries, start, length, n_days=DEFAULT_N_DAYS):
    """Every channel in every country stops at once -- real, but with no
    control group, and indistinguishable from a data outage by spend alone."""
    return [
        _entry(f"GLOBAL_PAUSE_{c}_{start}", "global_pause", c, "ALL",
               start, start + length, 0,
               f"All advertising stops in {c} during a {length}-day global "
               f"pause.", n_days)
        for c in countries
    ]


def ramp(country, channel, start, block, multipliers, n_days=DEFAULT_N_DAYS):
    """NEGATIVE CONTROL. Spend drifts upward across contiguous blocks rather
    than stepping. A step detector that fires here is wrong."""
    out = []
    for i, m in enumerate(multipliers):
        s = start + i * block
        out.append(_entry(f"{country}_RAMP_{_slug(channel)}_{i + 1}",
                          "ramp_block", country, channel, s, s + block, m,
                          f"Ramp block {i + 1} of {len(multipliers)} at "
                          f"{m:g}x -- gradual drift, not a step.", n_days))
    return out


def intermittent(country, channel, n, off_len, period, start=0,
                 n_days=DEFAULT_N_DAYS):
    """NEGATIVE CONTROL. A flighting channel whose short gaps are normal. Any
    single gap detected as a holdout is a false positive."""
    return [
        _entry(f"{country}_INTERMITTENT_{_slug(channel)}_{i + 1}",
               "intermittent_baseline", country, channel,
               start + i * period, start + i * period + off_len, 0,
               f"Routine {off_len}-day flighting gap {i + 1} of {n}.", n_days)
        for i in range(n)
    ]
```

- [ ] **Step 4: Run the tests and make sure they pass**

Run: `synthetic_data_generator/.venv/bin/python -m pytest tests/benchmark/test_events.py -v`
Expected: 13 passed.

- [ ] **Step 5: Commit**

```bash
cd /Users/user123/Desktop/projects/datascience-project
git add benchmark/spec/events.py tests/benchmark/test_events.py
git commit -m "feat(benchmark): add event recipe builders and negative controls"
```

---

### Task 6: Scenario assembly

Turns the axes and recipes into exactly 100 deterministic scenarios.

**Files:**
- Create: `benchmark/spec/scenarios.py`
- Test: `tests/benchmark/test_scenarios.py`

**Interfaces:**
- Consumes: `benchmark.spec.axes` (Task 4), `benchmark.spec.events` (Task 5).
- Produces:
  - `@dataclass(frozen=True) Scenario` with fields `sid: str`, `split: str`, `seed: int`, `family: str`, `years: int`, `countries: tuple[dict, ...]`, `channels: tuple[dict, ...]`, `baseline: dict`, `campaign_spend: dict`, `events: tuple[dict, ...]`, `meta: dict`
  - `FAMILY_COUNTS: dict[str, tuple[int, int]]` — family → (dev count, test count)
  - `build_all() -> list[Scenario]` — all 100, deterministic
  - `build_split(split: str) -> list[Scenario]`
  - `spec_hash(scenarios: list[Scenario]) -> str` — SHA-256 over canonical JSON
  - Tasks 7, 8, 9, 10 all consume `Scenario`.

- [ ] **Step 1: Write the failing test**

Create `tests/benchmark/test_scenarios.py`:

```python
import pytest

from benchmark.spec import scenarios
from benchmark.spec.events import NON_EVENT_TYPES


@pytest.fixture(scope="module")
def all_scenarios():
    return scenarios.build_all()


def test_exactly_one_hundred_scenarios(all_scenarios):
    assert len(all_scenarios) == 100
    assert sum(s.split == "dev" for s in all_scenarios) == 45
    assert sum(s.split == "test" for s in all_scenarios) == 55


def test_family_counts_match_the_spec_table(all_scenarios):
    for family, (n_dev, n_test) in scenarios.FAMILY_COUNTS.items():
        got_dev = sum(s.family == family and s.split == "dev" for s in all_scenarios)
        got_test = sum(s.family == family and s.split == "test" for s in all_scenarios)
        assert (got_dev, got_test) == (n_dev, n_test), f"family {family}"


def test_seed_ranges_are_disjoint_between_splits(all_scenarios):
    dev = {s.seed for s in all_scenarios if s.split == "dev"}
    test = {s.seed for s in all_scenarios if s.split == "test"}
    assert dev.isdisjoint(test)
    assert all(1000 <= s < 2000 for s in dev)
    assert all(5000 <= s < 6000 for s in test)


def test_scenario_ids_are_unique(all_scenarios):
    ids = [s.sid for s in all_scenarios]
    assert len(ids) == len(set(ids))


def test_build_is_deterministic():
    a, b = scenarios.build_all(), scenarios.build_all()
    assert scenarios.spec_hash(a) == scenarios.spec_hash(b)
    assert [s.sid for s in a] == [s.sid for s in b]


def test_null_scenarios_carry_no_events(all_scenarios):
    nulls = [s for s in all_scenarios if s.family == "null"]
    assert len(nulls) == 9
    assert all(s.events == () for s in nulls)


def test_every_event_references_a_country_and_channel_in_its_scenario(all_scenarios):
    for s in all_scenarios:
        codes = {c["code"] for c in s.countries}
        names = {c["name"] for c in s.channels}
        for e in s.events:
            assert e["country"] in codes, f"{s.sid}: unknown country {e['country']}"
            assert e["channel"] == "ALL" or e["channel"] in names, \
                f"{s.sid}: unknown channel {e['channel']}"


def test_every_event_window_fits_inside_the_series(all_scenarios):
    for s in all_scenarios:
        n_days = 365 * s.years
        for e in s.events:
            assert 0 <= e["start_day"] < e["end_day"] <= n_days, f"{s.sid}: {e}"


def test_channel_spend_shares_are_valid_for_the_generator(all_scenarios):
    for s in all_scenarios:
        head = s.channels[:-1]
        assert "spend_share_min" not in s.channels[-1]
        assert sum(c["spend_share_max"] for c in head) < 0.92, s.sid


def test_mixed_scenarios_really_do_mix(all_scenarios):
    mixed = [s for s in all_scenarios if s.family == "mixed"]
    assert len(mixed) == 18
    for s in mixed:
        kinds = {e["pattern_type"] for e in s.events} - NON_EVENT_TYPES
        assert len(kinds) >= 3, f"{s.sid} has only {kinds}"


def test_every_variation_axis_is_exercised_on_both_splits(all_scenarios):
    for split in ["dev", "test"]:
        subset = [s for s in all_scenarios if s.split == split]
        assert {s.meta["noise_level"] for s in subset} == {"low", "med", "high"}
        assert {s.meta["market_spread"] for s in subset} == \
            {"tight", "moderate", "extreme"}
        assert {s.meta["n_countries"] for s in subset} >= {1, 2, 3, 5, 8}
        assert {s.meta["n_channels"] for s in subset} >= {2, 4, 6, 9, 12}
        assert {s.meta["trend_p"] for s in subset} == {0.0, 0.5, 1.0}


def test_eight_country_scenarios_are_capped_at_one_year(all_scenarios):
    """Runtime control: 8 countries x 2 years would blow the generation budget."""
    for s in all_scenarios:
        if s.meta["n_countries"] >= 8:
            assert s.years == 1, s.sid


def test_edge_case_family_covers_every_named_case(all_scenarios):
    for split in ["dev", "test"]:
        cases = {s.meta["edge_case"] for s in all_scenarios
                 if s.family == "edge" and s.split == split}
        assert cases == {
            "censored_start", "censored_end", "back_to_back", "overlapping",
            "too_short", "very_long", "single_channel_market",
            "intermittent_channel", "gradual_ramp", "global_pause",
        }, f"{split}: {cases}"


def test_near_zero_and_exact_zero_are_both_represented(all_scenarios):
    mults = {e["multiplier"] for s in all_scenarios for e in s.events
             if e["pattern_type"] == "natural_holdout"}
    assert 0 in mults
    assert any(0 < m < 0.1 for m in mults)


def test_meta_carries_no_event_locations(all_scenarios):
    """meta.json lands on the truth side, but keeping it free of day offsets
    means a leak would be harmless rather than fatal."""
    for s in all_scenarios:
        blob = repr(s.meta)
        assert "start_day" not in blob and "end_day" not in blob
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `synthetic_data_generator/.venv/bin/python -m pytest tests/benchmark/test_scenarios.py -v`
Expected: FAIL — `No module named 'benchmark.spec.scenarios'`.

- [ ] **Step 3: Write `benchmark/spec/scenarios.py`**

```python
"""Deterministic assembly of the 100-scenario benchmark.

Every scenario is a (config.yaml, events_config.yaml, seed) triple. Given the
same code, build_all() always returns the same 100 scenarios -- that is what
makes the sealed test split provable rather than merely asserted.

Development scenarios draw seeds 1000-1999, test scenarios 5000-5999.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

import numpy as np

from benchmark.spec import axes, events as ev

# family -> (n_dev, n_test)
FAMILY_COUNTS: dict[str, tuple[int, int]] = {
    "null": (4, 5),
    "dark": (3, 4),
    "single_channel": (3, 4),
    "holdout": (3, 4),
    "step": (5, 6),
    "pulse": (3, 4),
    "launch": (3, 4),
    "cross_market": (3, 4),
    "mixed": (8, 10),
    "edge": (10, 10),
}

EDGE_CASES = (
    "censored_start", "censored_end", "back_to_back", "overlapping",
    "too_short", "very_long", "single_channel_market",
    "intermittent_channel", "gradual_ramp", "global_pause",
)

SEED_BASE = {"dev": 1000, "test": 5000}

CAMPAIGN_SPEND = dict(daily_total_mean=18500.0, daily_total_std=4000.0)


@dataclass(frozen=True)
class Scenario:
    sid: str
    split: str
    seed: int
    family: str
    years: int
    countries: tuple[dict, ...]
    channels: tuple[dict, ...]
    baseline: dict
    campaign_spend: dict
    events: tuple[dict, ...]
    meta: dict


def _shape(rng, family, index, gindex):
    """Pick the structural parameters for one scenario.

    Stratified rather than independently sampled, so each family reliably spans
    all three noise levels instead of clustering by chance.

    `index` counts within the family, `gindex` across the whole split. Noise and
    trend are driven by the family-local counter while seasonality and market
    spread are driven by the global one, so the axes stay decorrelated -- drive
    them all off `index` and every high-noise scenario would also be an
    extreme-spread scenario, which would make the breakdowns in spec section 9
    uninterpretable.
    """
    noise_level = axes.NOISE_LEVELS[index % 3]
    trend_p = axes.TREND_LEVELS[(index // 3) % 3]
    temp_var = axes.SEASONALITY_LEVELS[gindex % 3]
    spread = ("tight", "moderate", "extreme")[(gindex // 3) % 3]

    if family == "cross_market":
        n_countries = int(rng.choice([3, 5, 8]))
    elif family == "edge":
        n_countries = int(rng.choice([2, 3]))
    else:
        n_countries = int(rng.choice([1, 2, 3, 5, 8]))

    n_channels = int(rng.choice([2, 4, 6, 9, 12]))
    years = 1 if n_countries >= 8 else 2
    return noise_level, trend_p, temp_var, spread, n_countries, n_channels, years


def _mk(sid, split, seed, family, rng, *, force_countries=None,
        force_channels=None, force_years=None, index=0, gindex=0,
        extra_meta=None, event_fn=None):
    (noise_level, trend_p, temp_var, spread,
     n_countries, n_channels, years) = _shape(rng, family, index, gindex)

    n_countries = force_countries or n_countries
    n_channels = force_channels or n_channels
    years = force_years or (1 if n_countries >= 8 else years)

    countries = axes.pick_countries(rng, n_countries, spread)
    channels = axes.apply_noise(axes.pick_channels(rng, n_channels), noise_level)
    n_days = 365 * years

    entries = tuple(event_fn(rng, countries, channels, n_days)) if event_fn else ()

    sizes = [c["market_size"] for c in countries]
    meta = dict(
        family=family, noise_level=noise_level, trend_p=trend_p,
        temp_var=temp_var, market_spread=spread, n_countries=n_countries,
        n_channels=n_channels, years=years, n_days=n_days,
        market_size_ratio=round(max(sizes) / min(sizes), 3),
        n_events=len(entries),
        event_types=sorted({e["pattern_type"] for e in entries}),
    )
    meta.update(extra_meta or {})

    return Scenario(
        sid=sid, split=split, seed=seed, family=family, years=years,
        countries=tuple(countries), channels=tuple(channels),
        baseline=axes.baseline_for(noise_level, trend_p, temp_var),
        campaign_spend=dict(CAMPAIGN_SPEND), events=entries, meta=meta,
    )


# --- per-family event builders -------------------------------------------
# Each takes (rng, countries, channels, n_days) and returns a list of entries.

def _pick_window(rng, n_days, length):
    """A start offset that leaves the window inside the series with margin."""
    latest = n_days - length - 30
    return int(rng.integers(30, max(31, latest)))


def _events_dark(rng, countries, channels, n_days):
    c = countries[int(rng.integers(len(countries)))]["code"]
    length = int(rng.choice([14, 42, 90]))
    return ev.dark(c, _pick_window(rng, n_days, length), length, n_days)


def _events_single(rng, countries, channels, n_days):
    c = countries[int(rng.integers(len(countries)))]["code"]
    keep = channels[int(rng.integers(len(channels)))]["name"]
    length = int(rng.choice([14, 30, 42]))
    return ev.single_channel(c, keep, _pick_window(rng, n_days, length), length, n_days)


def _events_holdout(rng, countries, channels, n_days):
    c = countries[int(rng.integers(len(countries)))]["code"]
    ch = channels[int(rng.integers(len(channels)))]["name"]
    length = int(rng.choice([14, 42, 90]))
    mult = float(rng.choice([0, 0, 0.02, 0.05, 0.08]))
    return ev.holdout(c, ch, _pick_window(rng, n_days, length), length, mult, n_days)


def _events_step(rng, countries, channels, n_days):
    c = countries[int(rng.integers(len(countries)))]["code"]
    ch = channels[int(rng.integers(len(channels)))]["name"]
    length = int(rng.choice([42, 56, 90]))
    mult = float(rng.choice([0.33, 1.5, 3.0, 5.0]))
    return ev.step(c, ch, _pick_window(rng, n_days, length), length, mult, n_days)


def _events_pulse(rng, countries, channels, n_days):
    c = countries[int(rng.integers(len(countries)))]["code"]
    ch = channels[int(rng.integers(len(channels)))]["name"]
    length = int(rng.choice([10, 14, 21]))
    gap = length + int(rng.choice([14, 21, 28]))
    n_pulses = int(rng.choice([2, 3, 4]))
    first = _pick_window(rng, n_days, gap * n_pulses + length)
    return ev.pulse(c, ch, [first + i * gap for i in range(n_pulses)], length, n_days)


def _events_launch(rng, countries, channels, n_days):
    ch = channels[int(rng.integers(len(channels)))]["name"]
    out = []
    # Staggered: each market lights up at a different offset; at least one
    # market carries the channel from day one, to act as the comparison group.
    offsets = [0] + [int(rng.choice([45, 90, 150])) for _ in countries[1:]]
    for country, off in zip(countries, offsets):
        if off > 0:
            out += ev.launch(country["code"], ch, off, n_days)
    if not out:  # single-market scenario, force one launch
        out = ev.launch(countries[0]["code"], ch, 90, n_days)
    return out


def _events_cross_market(rng, countries, channels, n_days):
    """One market holds a channel out while every peer keeps running it."""
    ch = channels[int(rng.integers(len(channels)))]["name"]
    c = countries[int(rng.integers(len(countries)))]["code"]
    length = int(rng.choice([42, 56, 90]))
    return ev.holdout(c, ch, _pick_window(rng, n_days, length), length, 0, n_days)


def _events_mixed(rng, countries, channels, n_days):
    """Three to five simultaneous events of different kinds across markets."""
    builders = [_events_dark, _events_single, _events_holdout,
                _events_step, _events_pulse]
    k = int(rng.integers(3, 6))
    idx = rng.choice(len(builders), size=k, replace=False)
    out = []
    for i in idx:
        out += builders[i](rng, countries, channels, n_days)
    return out


def _events_edge(rng, countries, channels, n_days, case):
    c0 = countries[0]["code"]
    ch0 = channels[0]["name"]
    ch1 = channels[min(1, len(channels) - 1)]["name"]

    if case == "censored_start":
        return ev.holdout(c0, ch0, 0, 60, 0, n_days)
    if case == "censored_end":
        return ev.holdout(c0, ch0, n_days - 60, 60, 0, n_days)
    if case == "back_to_back":
        return (ev.holdout(c0, ch0, 200, 42, 0, n_days)
                + ev.step(c0, ch0, 242, 56, 3.0, n_days))
    if case == "overlapping":
        return (ev.holdout(c0, ch0, 200, 60, 0, n_days)
                + ev.step(c0, ch1, 230, 60, 2.5, n_days))
    if case == "too_short":
        return ev.holdout(c0, ch0, 300, 5, 0, n_days)
    if case == "very_long":
        return ev.holdout(c0, ch0, 100, 180, 0, n_days)
    if case == "single_channel_market":
        # Every channel but one is held out for the whole series, so the
        # detector sees a market with exactly one active channel.
        out = []
        for ch in channels[1:]:
            out += ev.holdout(c0, ch["name"], 0, n_days, 0, n_days)
        out += ev.dark(c0, 300, 42, n_days)
        return out
    if case == "intermittent_channel":
        return (ev.intermittent(c0, ch0, n=20, off_len=3, period=14, start=30,
                                n_days=n_days)
                + ev.holdout(c0, ch1, 400, 42, 0, n_days))
    if case == "gradual_ramp":
        return ev.ramp(c0, ch0, 200, 10, [1.2, 1.4, 1.6, 1.8, 2.0], n_days)
    if case == "global_pause":
        return ev.global_pause([c["code"] for c in countries], 300, 21, n_days)
    raise ValueError(f"unknown edge case: {case}")


EVENT_FNS = {
    "null": None,
    "dark": _events_dark,
    "single_channel": _events_single,
    "holdout": _events_holdout,
    "step": _events_step,
    "pulse": _events_pulse,
    "launch": _events_launch,
    "cross_market": _events_cross_market,
    "mixed": _events_mixed,
}


def build_split(split: str) -> list[Scenario]:
    base = SEED_BASE[split]
    out: list[Scenario] = []
    counter = 0

    for family, (n_dev, n_test) in FAMILY_COUNTS.items():
        n = n_dev if split == "dev" else n_test
        for i in range(n):
            seed = base + counter
            counter += 1
            sid = f"{split}_{counter:03d}_{family}"
            rng = np.random.default_rng(seed)

            if family == "edge":
                case = EDGE_CASES[i % len(EDGE_CASES)]
                forced_channels = 2 if case == "single_channel_market" else None
                out.append(_mk(
                    sid, split, seed, family, rng, index=i, gindex=counter - 1,
                    force_channels=forced_channels,
                    extra_meta={"edge_case": case},
                    event_fn=lambda r, co, ch, nd, _c=case:
                        _events_edge(r, co, ch, nd, _c),
                ))
            else:
                out.append(_mk(sid, split, seed, family, rng, index=i,
                               gindex=counter - 1,
                               event_fn=EVENT_FNS[family]))
    return out


def build_all() -> list[Scenario]:
    return build_split("dev") + build_split("test")


def spec_hash(scenarios: list[Scenario]) -> str:
    """SHA-256 over the canonical JSON of every scenario definition.

    Recorded in the SEALED marker, so a regenerated test split with even one
    changed parameter is detectable.
    """
    blob = json.dumps([asdict(s) for s in scenarios], sort_keys=True,
                      separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()
```

- [ ] **Step 4: Run the tests**

Run: `synthetic_data_generator/.venv/bin/python -m pytest tests/benchmark/test_scenarios.py -v`
Expected: 15 passed. If `test_every_variation_axis_is_exercised_on_both_splits` fails because a level is missing on one split, widen the stratification in `_shape` — cycle the axis on `index` rather than sampling it — rather than loosening the assertion. The assertion is the point.

- [ ] **Step 5: Eyeball the manifest of scenarios**

```bash
cd /Users/user123/Desktop/projects/datascience-project
synthetic_data_generator/.venv/bin/python - <<'PY'
from benchmark.spec import scenarios
allsc = scenarios.build_all()
print(f"{len(allsc)} scenarios, spec_hash={scenarios.spec_hash(allsc)[:16]}")
cy = sum(s.meta["n_countries"] * s.years for s in allsc)
print(f"total country-years: {cy}  ->  est {cy * 25 / 60 / 6:.0f} min at 6 workers")
for s in allsc[:5]:
    print(f"  {s.sid:28} seed={s.seed} {s.meta['n_countries']}c "
          f"{s.meta['n_channels']}ch {s.years}y events={s.meta['n_events']}")
PY
```
Expected: 100 scenarios and an estimate near 45 minutes. If the estimate exceeds 70 minutes, reduce the `n_countries` choices in `_shape` for the `mixed` and `step` families before generating.

- [ ] **Step 6: Commit**

```bash
cd /Users/user123/Desktop/projects/datascience-project
git add benchmark/spec/scenarios.py tests/benchmark/test_scenarios.py
git commit -m "feat(benchmark): assemble 100 deterministic dev/test scenarios"
```

---

### Task 7: Config writer

**Files:**
- Create: `benchmark/harness/config_writer.py`
- Test: `tests/benchmark/test_config_writer.py`

**Interfaces:**
- Consumes: `Scenario` from Task 6.
- Produces: `write_scenario_configs(scenario: Scenario, dest: Path) -> tuple[Path, Path]`, returning `(config_path, events_path)`. Task 8 consumes this.

- [ ] **Step 1: Write the failing test**

Create `tests/benchmark/test_config_writer.py`:

```python
import subprocess

import yaml

from benchmark.harness.config_writer import write_scenario_configs
from benchmark.spec import scenarios


def test_writes_two_yaml_files(tmp_path):
    s = scenarios.build_split("dev")[0]
    cfg_path, ev_path = write_scenario_configs(s, tmp_path)
    assert cfg_path.is_file() and ev_path.is_file()
    assert cfg_path.name == "config.yaml" and ev_path.name == "events_config.yaml"


def test_config_carries_every_field_the_generator_reads(tmp_path):
    s = scenarios.build_split("dev")[10]
    cfg_path, _ = write_scenario_configs(s, tmp_path)
    cfg = yaml.safe_load(cfg_path.read_text())

    assert cfg["years"] == s.years
    assert cfg["start_date"] == "2024/01/01"
    assert set(cfg) >= {"years", "start_date", "revenue_per_conv",
                        "customer_types", "sales_channels", "baseline",
                        "campaign_spend", "countries", "channels"}
    assert len(cfg["countries"]) == s.meta["n_countries"]
    assert len(cfg["channels"]) == s.meta["n_channels"]


def test_numbers_round_trip_as_plain_scalars_not_numpy_tags(tmp_path):
    """PyYAML serialises numpy scalars as !!python/object tags, which R's yaml
    reader cannot parse. Everything must come out as a plain number."""
    s = scenarios.build_split("test")[20]
    cfg_path, ev_path = write_scenario_configs(s, tmp_path)
    for p in [cfg_path, ev_path]:
        text = p.read_text()
        assert "!!python" not in text, f"{p.name} carries a python tag"
        assert "numpy" not in text, f"{p.name} carries a numpy tag"


def test_null_scenario_writes_an_empty_event_list(tmp_path):
    s = next(s for s in scenarios.build_all() if s.family == "null")
    _, ev_path = write_scenario_configs(s, tmp_path)
    assert yaml.safe_load(ev_path.read_text()) == []


def test_r_can_actually_read_the_generated_config(tmp_path):
    """The only test that proves the YAML dialect is compatible. R's yaml
    package is stricter than PyYAML about several constructs."""
    s = scenarios.build_split("dev")[5]
    cfg_path, ev_path = write_scenario_configs(s, tmp_path)
    script = (
        f'c <- yaml::read_yaml("{cfg_path}"); e <- yaml::read_yaml("{ev_path}"); '
        'cat(length(c$countries), length(c$channels), length(e))'
    )
    proc = subprocess.run(["Rscript", "-e", script], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    n_countries, n_channels, n_events = proc.stdout.split()
    assert int(n_countries) == s.meta["n_countries"]
    assert int(n_channels) == s.meta["n_channels"]
    assert int(n_events) == len(s.events)
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `synthetic_data_generator/.venv/bin/python -m pytest tests/benchmark/test_config_writer.py -v`
Expected: FAIL — `No module named 'benchmark.harness.config_writer'`.

- [ ] **Step 3: Write `benchmark/harness/config_writer.py`**

```python
"""Serialise a Scenario into the two YAML files the generator reads.

The only subtlety is numeric types: numpy scalars survive from the sampling
code, and PyYAML tags them as python objects, which R's yaml reader rejects.
Everything is coerced to a plain int, float or str on the way out.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from benchmark.spec.scenarios import Scenario

START_DATE = "2024/01/01"
REVENUE_PER_CONV = 40.0
CUSTOMER_TYPES = ["New", "Returning"]
SALES_CHANNELS = ["Ecom", "Stores"]


def _plain(obj):
    """Recursively strip numpy types so PyYAML emits plain scalars."""
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.str_,)):
        return str(obj)
    return obj


def write_scenario_configs(scenario: Scenario, dest: Path) -> tuple[Path, Path]:
    """Write config.yaml and events_config.yaml into `dest`."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    config = _plain({
        "years": scenario.years,
        "start_date": START_DATE,
        "revenue_per_conv": REVENUE_PER_CONV,
        "customer_types": CUSTOMER_TYPES,
        "sales_channels": SALES_CHANNELS,
        "baseline": scenario.baseline,
        "campaign_spend": scenario.campaign_spend,
        "countries": list(scenario.countries),
        "channels": list(scenario.channels),
    })

    cfg_path = dest / "config.yaml"
    ev_path = dest / "events_config.yaml"

    with cfg_path.open("w") as f:
        yaml.safe_dump(config, f, sort_keys=False, default_flow_style=False)
    with ev_path.open("w") as f:
        yaml.safe_dump(_plain(list(scenario.events)), f, sort_keys=False,
                       default_flow_style=False)

    return cfg_path, ev_path
```

- [ ] **Step 4: Run the tests and make sure they pass**

Run: `synthetic_data_generator/.venv/bin/python -m pytest tests/benchmark/test_config_writer.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
cd /Users/user123/Desktop/projects/datascience-project
git add benchmark/harness/config_writer.py tests/benchmark/test_config_writer.py
git commit -m "feat(benchmark): serialise scenarios to generator-compatible YAML"
```

---

### Task 8: Scenario runner

Runs one scenario end to end and files the output so ground truth never lands beside the data.

**Files:**
- Create: `benchmark/harness/runner.py`
- Test: `tests/benchmark/test_runner.py`

**Interfaces:**
- Consumes: `write_scenario_configs` (Task 7), the R and Python CLI contracts (Tasks 2, 3), `Scenario` (Task 6).
- Produces:
  - `DATASETS_DIR: Path` — `benchmark/datasets`
  - `dataset_dir(split, sid) -> Path`, `truth_dir(split, sid) -> Path`
  - `run_scenario(scenario, root=DATASETS_DIR, force=False) -> dict` — a result record with `sid`, `status` (`"generated"`, `"skipped"`, `"failed"`), `seconds`, and `error`
  - `run_many(scenarios, root=DATASETS_DIR, workers=6, force=False) -> list[dict]`

- [ ] **Step 1: Write the failing test**

Create `tests/benchmark/test_runner.py`:

```python
import pandas as pd
import pytest

from benchmark.harness import runner
from benchmark.spec import scenarios


def _tiny_scenario():
    """A one-country, two-channel, one-year scenario with a single holdout --
    the cheapest thing that still exercises the whole path."""
    s = next(s for s in scenarios.build_all() if s.family == "holdout")
    return scenarios.Scenario(
        sid="tiny_001_holdout", split="dev", seed=1, family="holdout", years=1,
        countries=s.countries[:1], channels=s.channels[:2],
        baseline=s.baseline, campaign_spend=s.campaign_spend,
        events=tuple(e for e in s.events if e["country"] == s.countries[0]["code"])
                or ({"pattern_id": "T_H", "pattern_type": "natural_holdout",
                     "country": s.countries[0]["code"],
                     "channel": s.channels[0]["name"], "start_day": 100,
                     "end_day": 142, "multiplier": 0, "description": "t"},),
        meta=dict(s.meta, n_countries=1, n_channels=2, years=1, n_days=365),
    )


def test_directory_helpers_separate_data_from_truth(tmp_path):
    d = runner.dataset_dir("test", "test_007_dark", root=tmp_path)
    t = runner.truth_dir("test", "test_007_dark", root=tmp_path)
    assert d.parent.name == "test"
    assert t.parent.name == "test_truth"
    assert d != t


@pytest.mark.slow
def test_run_scenario_files_outputs_on_the_correct_side(tmp_path):
    s = _tiny_scenario()
    result = runner.run_scenario(s, root=tmp_path)
    assert result["status"] == "generated", result.get("error")

    data = runner.dataset_dir(s.split, s.sid, root=tmp_path)
    truth = runner.truth_dir(s.split, s.sid, root=tmp_path)

    # THE black-box assertion: nothing but the two readable CSVs.
    assert sorted(p.name for p in data.iterdir()) == ["media.csv", "sales.csv"]
    assert {"ground_truth.csv", "meta.json"} <= {p.name for p in truth.iterdir()}

    media = pd.read_csv(data / "media.csv")
    assert len(media) > 0
    assert set(media.columns) == {
        "date", "ad_platform", "advertising_channel", "campaign_name",
        "campaign_id", "media_investment", "clicks", "impressions",
        "conversions", "conversion_value", "country_code"}

    gt = pd.read_csv(truth / "ground_truth.csv")
    assert len(gt) == len(s.events)


@pytest.mark.slow
def test_rerunning_skips_unless_forced(tmp_path):
    s = _tiny_scenario()
    assert runner.run_scenario(s, root=tmp_path)["status"] == "generated"
    assert runner.run_scenario(s, root=tmp_path)["status"] == "skipped"
    assert runner.run_scenario(s, root=tmp_path, force=True)["status"] == "generated"


@pytest.mark.slow
def test_failure_is_reported_not_raised(tmp_path):
    """A 45-minute batch must not die because one scenario is malformed."""
    s = _tiny_scenario()
    broken = scenarios.Scenario(
        **{**s.__dict__, "sid": "broken_001",
           "channels": ({"name": "Nonsense", "type": "impression"},)})
    result = runner.run_scenario(broken, root=tmp_path)
    assert result["status"] == "failed"
    assert result["error"]
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `synthetic_data_generator/.venv/bin/python -m pytest tests/benchmark/test_runner.py -v`
Expected: FAIL — `No module named 'benchmark.harness.runner'`.

- [ ] **Step 3: Write `benchmark/harness/runner.py`**

```python
"""Run scenarios through the R + Python generator and file the output.

The filing rule is the whole point: `datasets/<split>/<sid>/` receives only
media.csv and sales.csv. Ground truth, meta.json, true_roi.csv and the raw wide
output all go to `datasets/<split>_truth/<sid>/`, which no detector code is
permitted to import.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from benchmark.harness.config_writer import write_scenario_configs
from benchmark.spec.scenarios import Scenario

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GENERATOR_DIR = PROJECT_ROOT / "synthetic_data_generator"
R_SCRIPT = GENERATOR_DIR / "generate_with_simmmulator.R"
REFORMAT = GENERATOR_DIR / "reformat.py"
VENV_PYTHON = GENERATOR_DIR / ".venv" / "bin" / "python"
DATASETS_DIR = PROJECT_ROOT / "benchmark" / "datasets"

# What a detector is allowed to see.
DATA_FILES = ("media.csv", "sales.csv")
# Everything else the generator produces.
TRUTH_FILES = ("ground_truth.csv", "true_roi.csv")

R_TIMEOUT_S = 1800


def dataset_dir(split: str, sid: str, root: Path = DATASETS_DIR) -> Path:
    return Path(root) / split / sid


def truth_dir(split: str, sid: str, root: Path = DATASETS_DIR) -> Path:
    return Path(root) / f"{split}_truth" / sid


def _is_complete(split: str, sid: str, root: Path) -> bool:
    data, truth = dataset_dir(split, sid, root), truth_dir(split, sid, root)
    return (all((data / f).is_file() for f in DATA_FILES)
            and (truth / "ground_truth.csv").is_file()
            and (truth / "meta.json").is_file())


def run_scenario(scenario: Scenario, root: Path = DATASETS_DIR,
                 force: bool = False) -> dict:
    """Generate one scenario. Never raises -- failures come back as a record."""
    started = time.time()
    result = {"sid": scenario.sid, "split": scenario.split,
              "seed": scenario.seed, "status": "failed",
              "seconds": 0.0, "error": None}

    if not force and _is_complete(scenario.split, scenario.sid, root):
        result["status"] = "skipped"
        return result

    try:
        with tempfile.TemporaryDirectory(prefix=f"scn_{scenario.sid}_") as tmp:
            work = Path(tmp)
            cfg, evs = write_scenario_configs(scenario, work)

            subprocess.run(
                ["Rscript", str(R_SCRIPT), "--seed", str(scenario.seed),
                 "--config", str(cfg), "--events", str(evs),
                 "--outdir", str(work)],
                check=True, capture_output=True, text=True, timeout=R_TIMEOUT_S,
            )
            subprocess.run(
                [str(VENV_PYTHON), str(REFORMAT), "--config", str(cfg),
                 "--events", str(evs), "--outdir", str(work)],
                check=True, capture_output=True, text=True, timeout=600,
            )

            data = dataset_dir(scenario.split, scenario.sid, root)
            truth = truth_dir(scenario.split, scenario.sid, root)
            for d in (data, truth):
                if d.exists():
                    shutil.rmtree(d)
                d.mkdir(parents=True)

            for name in DATA_FILES:
                shutil.copy2(work / name, data / name)
            for name in TRUTH_FILES:
                if (work / name).is_file():
                    shutil.copy2(work / name, truth / name)

            payload = dict(scenario.meta)
            payload["sid"] = scenario.sid
            payload["split"] = scenario.split
            payload["seed"] = scenario.seed
            (truth / "meta.json").write_text(json.dumps(payload, indent=2,
                                                        sort_keys=True))
            (truth / "scenario.json").write_text(
                json.dumps(asdict(scenario), indent=2, sort_keys=True, default=str))

        result["status"] = "generated"
    except subprocess.CalledProcessError as exc:
        result["error"] = (exc.stderr or exc.stdout or str(exc))[-2000:]
    except Exception as exc:  # noqa: BLE001 -- a batch must survive one bad scenario
        result["error"] = f"{type(exc).__name__}: {exc}"

    result["seconds"] = round(time.time() - started, 1)
    return result


def run_many(scenarios, root: Path = DATASETS_DIR, workers: int = 6,
             force: bool = False, on_done=None) -> list[dict]:
    """Generate many scenarios in parallel. Subprocess-bound, so threads suffice."""
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_scenario, s, root, force): s for s in scenarios}
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            if on_done:
                on_done(res, len(results), len(futures))
    return sorted(results, key=lambda r: r["sid"])
```

- [ ] **Step 4: Run the fast tests**

Run: `synthetic_data_generator/.venv/bin/python -m pytest tests/benchmark/test_runner.py -v -m "not slow"`
Expected: 1 passed, 3 deselected.

- [ ] **Step 5: Run the slow tests**

Run: `synthetic_data_generator/.venv/bin/python -m pytest tests/benchmark/test_runner.py -v -m slow`
Expected: 3 passed, roughly 2 minutes.

- [ ] **Step 6: Commit**

```bash
cd /Users/user123/Desktop/projects/datascience-project
git add benchmark/harness/runner.py tests/benchmark/test_runner.py
git commit -m "feat(benchmark): run scenarios in parallel, filing truth separately"
```

---

### Task 9: Sealing

**Files:**
- Create: `benchmark/harness/seal.py`
- Test: `tests/benchmark/test_seal.py`

**Interfaces:**
- Consumes: `dataset_dir`, `truth_dir` (Task 8), `spec_hash` (Task 6).
- Produces:
  - `write_manifest(directory: Path) -> Path` — writes `manifest.sha256`
  - `verify_manifest(directory: Path) -> tuple[bool, list[str]]` — `(ok, problems)`
  - `seal_split(split: str, scenarios: list[Scenario], root=DATASETS_DIR) -> dict`
  - `verify_seal(split: str, root=DATASETS_DIR) -> tuple[bool, list[str]]`
  - `is_sealed(split: str, root=DATASETS_DIR) -> bool`
  - Plan 2's `run_final.py` calls `verify_seal`.

- [ ] **Step 1: Write the failing test**

Create `tests/benchmark/test_seal.py`:

```python
import json

import pytest

from benchmark.harness import seal


@pytest.fixture
def fake_split(tmp_path):
    """Two scenarios' worth of files, data and truth side."""
    for sid in ["test_001_dark", "test_002_step"]:
        d = tmp_path / "test" / sid
        t = tmp_path / "test_truth" / sid
        d.mkdir(parents=True)
        t.mkdir(parents=True)
        (d / "media.csv").write_text("date,media_investment\n2024-01-01,100\n")
        (d / "sales.csv").write_text("date,turnover\n2024-01-01,900\n")
        (t / "ground_truth.csv").write_text("pattern_id\nX\n")
        (t / "meta.json").write_text('{"family": "dark"}')
    return tmp_path


def test_manifest_covers_every_file(fake_split):
    path = seal.write_manifest(fake_split / "test")
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    assert len(lines) == 4  # two scenarios x two CSVs
    assert all(len(l.split("  ")[0]) == 64 for l in lines)


def test_verify_passes_on_an_untouched_directory(fake_split):
    seal.write_manifest(fake_split / "test")
    ok, problems = seal.verify_manifest(fake_split / "test")
    assert ok and problems == []


def test_verify_detects_a_modified_file(fake_split):
    seal.write_manifest(fake_split / "test")
    (fake_split / "test" / "test_001_dark" / "media.csv").write_text("tampered\n")
    ok, problems = seal.verify_manifest(fake_split / "test")
    assert not ok
    assert any("media.csv" in p and "changed" in p for p in problems)


def test_verify_detects_a_deleted_file(fake_split):
    seal.write_manifest(fake_split / "test")
    (fake_split / "test" / "test_002_step" / "sales.csv").unlink()
    ok, problems = seal.verify_manifest(fake_split / "test")
    assert not ok
    assert any("missing" in p for p in problems)


def test_verify_detects_an_added_file(fake_split):
    seal.write_manifest(fake_split / "test")
    (fake_split / "test" / "test_001_dark" / "extra.csv").write_text("surprise\n")
    ok, problems = seal.verify_manifest(fake_split / "test")
    assert not ok
    assert any("unexpected" in p for p in problems)


def test_seal_split_records_spec_hash_and_counts(fake_split):
    from benchmark.spec import scenarios
    scns = [s for s in scenarios.build_split("test")][:2]
    info = seal.seal_split("test", scns, root=fake_split)

    marker = json.loads((fake_split / "test" / "SEALED").read_text())
    assert marker["spec_hash"] == scenarios.spec_hash(scns)
    assert marker["n_scenarios"] == 2
    assert "sealed_at" in marker
    assert info == marker
    assert seal.is_sealed("test", root=fake_split)


def test_verify_seal_checks_both_data_and_truth(fake_split):
    from benchmark.spec import scenarios
    scns = scenarios.build_split("test")[:2]
    seal.seal_split("test", scns, root=fake_split)
    assert seal.verify_seal("test", root=fake_split)[0]

    (fake_split / "test_truth" / "test_001_dark" / "ground_truth.csv").write_text("X\n")
    ok, problems = seal.verify_seal("test", root=fake_split)
    assert not ok and problems


def test_sealing_twice_is_refused(fake_split):
    from benchmark.spec import scenarios
    scns = scenarios.build_split("test")[:2]
    seal.seal_split("test", scns, root=fake_split)
    with pytest.raises(RuntimeError, match="already sealed"):
        seal.seal_split("test", scns, root=fake_split)


def test_unsealed_split_reports_false(fake_split):
    assert not seal.is_sealed("dev", root=fake_split)
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `synthetic_data_generator/.venv/bin/python -m pytest tests/benchmark/test_seal.py -v`
Expected: FAIL — `No module named 'benchmark.harness.seal'`.

- [ ] **Step 3: Write `benchmark/harness/seal.py`**

```python
"""Freeze a split so that later claims about it are provable.

A sealed split carries a manifest.sha256 over every file on both the data and
truth sides, plus a SEALED marker recording when it was sealed and the hash of
the scenario definitions it was built from. Regenerating the split with any
parameter changed produces a different spec_hash; editing any file produces a
different manifest. Both are detectable by verify_seal().
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

from benchmark.harness.runner import DATASETS_DIR
from benchmark.spec.scenarios import Scenario, spec_hash

MANIFEST_NAME = "manifest.sha256"
SEAL_NAME = "SEALED"
_EXCLUDED = {MANIFEST_NAME, SEAL_NAME}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.rglob("*")
                  if p.is_file() and p.name not in _EXCLUDED)


def write_manifest(directory: Path) -> Path:
    directory = Path(directory)
    lines = [f"{_sha256(p)}  {p.relative_to(directory).as_posix()}"
             for p in _files(directory)]
    out = directory / MANIFEST_NAME
    out.write_text("\n".join(lines) + "\n")
    return out


def verify_manifest(directory: Path) -> tuple[bool, list[str]]:
    directory = Path(directory)
    manifest = directory / MANIFEST_NAME
    if not manifest.is_file():
        return False, [f"{directory}: no manifest"]

    expected = {}
    for line in manifest.read_text().splitlines():
        if line.strip():
            digest, rel = line.split("  ", 1)
            expected[rel] = digest

    actual = {p.relative_to(directory).as_posix(): p for p in _files(directory)}
    problems = []

    for rel, digest in sorted(expected.items()):
        if rel not in actual:
            problems.append(f"missing: {rel}")
        elif _sha256(actual[rel]) != digest:
            problems.append(f"changed: {rel}")
    for rel in sorted(set(actual) - set(expected)):
        problems.append(f"unexpected: {rel}")

    return not problems, problems


def seal_split(split: str, scenarios: list[Scenario],
               root: Path = DATASETS_DIR) -> dict:
    """Write manifests for both sides and stamp the SEALED marker."""
    root = Path(root)
    data_root, truth_root = root / split, root / f"{split}_truth"
    marker = data_root / SEAL_NAME

    if marker.is_file():
        raise RuntimeError(
            f"{split} is already sealed (marker at {marker}). Sealing again "
            f"would destroy the record of what the original benchmark was. "
            f"Delete the marker deliberately if you really mean to reseal.")

    write_manifest(data_root)
    write_manifest(truth_root)

    info = {
        "split": split,
        "sealed_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "spec_hash": spec_hash(scenarios),
        "n_scenarios": len(scenarios),
    }
    marker.write_text(json.dumps(info, indent=2, sort_keys=True))
    return info


def is_sealed(split: str, root: Path = DATASETS_DIR) -> bool:
    return (Path(root) / split / SEAL_NAME).is_file()


def verify_seal(split: str, root: Path = DATASETS_DIR) -> tuple[bool, list[str]]:
    root = Path(root)
    if not is_sealed(split, root):
        return False, [f"{split} is not sealed"]

    problems = []
    for directory in (root / split, root / f"{split}_truth"):
        ok, found = verify_manifest(directory)
        if not ok:
            problems += [f"{directory.name}: {p}" for p in found]
    return not problems, problems
```

- [ ] **Step 4: Run the tests and make sure they pass**

Run: `synthetic_data_generator/.venv/bin/python -m pytest tests/benchmark/test_seal.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
cd /Users/user123/Desktop/projects/datascience-project
git add benchmark/harness/seal.py tests/benchmark/test_seal.py
git commit -m "feat(benchmark): seal splits with SHA-256 manifests and a spec hash"
```

---

### Task 10: Generation CLI

**Files:**
- Create: `benchmark/harness/generate.py`
- Test: `tests/benchmark/test_generate_cli.py`

**Interfaces:**
- Consumes: everything from Tasks 6–9.
- Produces: `python -m benchmark.harness.generate [--split dev|test|all] [--workers N] [--force] [--seal] [--dry-run] [--limit N]`.

- [ ] **Step 1: Write the failing test**

Create `tests/benchmark/test_generate_cli.py`:

```python
import subprocess


def _run(venv_python, project_root, *args):
    return subprocess.run(
        [str(venv_python), "-m", "benchmark.harness.generate", *args],
        cwd=project_root, capture_output=True, text=True,
    )


def test_dry_run_reports_the_plan_without_generating(venv_python, project_root):
    proc = _run(venv_python, project_root, "--split", "all", "--dry-run")
    assert proc.returncode == 0, proc.stderr
    assert "100 scenarios" in proc.stdout
    assert "country-years" in proc.stdout
    assert "estimated" in proc.stdout.lower()


def test_dry_run_can_be_limited_to_one_split(venv_python, project_root):
    proc = _run(venv_python, project_root, "--split", "dev", "--dry-run")
    assert proc.returncode == 0, proc.stderr
    assert "45 scenarios" in proc.stdout


def test_seal_only_on_an_ungenerated_split_fails_loudly(venv_python, project_root,
                                                        tmp_path):
    """--seal-only must refuse immediately rather than starting a 45-minute
    generation run."""
    proc = _run(venv_python, project_root, "--split", "test", "--seal-only",
                "--root", str(tmp_path))
    assert proc.returncode != 0
    assert "not generated" in (proc.stdout + proc.stderr)
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `synthetic_data_generator/.venv/bin/python -m pytest tests/benchmark/test_generate_cli.py -v`
Expected: FAIL — `No module named benchmark.harness.generate`.

- [ ] **Step 3: Write `benchmark/harness/generate.py`**

```python
"""CLI for generating and sealing the benchmark.

    python -m benchmark.harness.generate --split all --dry-run
    python -m benchmark.harness.generate --split dev
    python -m benchmark.harness.generate --split test
    python -m benchmark.harness.generate --split test --seal

Generation is resumable: a scenario whose outputs already exist is skipped
unless --force is passed, so an interrupted 45-minute run can be restarted.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from benchmark.harness import runner, seal
from benchmark.spec import scenarios

SECONDS_PER_COUNTRY_YEAR = 25.0


def _plan(scns):
    country_years = sum(s.meta["n_countries"] * s.years for s in scns)
    return country_years, country_years * SECONDS_PER_COUNTRY_YEAR


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", choices=["dev", "test", "all"], default="all")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--force", action="store_true",
                   help="regenerate scenarios that already exist")
    p.add_argument("--seal", action="store_true",
                   help="seal the split after generating (test split only in practice)")
    p.add_argument("--seal-only", action="store_true",
                   help="seal an already-generated split without regenerating; "
                        "fails immediately if the split is incomplete")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=None,
                   help="generate only the first N scenarios, for smoke tests")
    p.add_argument("--root", default=str(runner.DATASETS_DIR))
    args = p.parse_args(argv)

    root = Path(args.root)
    splits = ["dev", "test"] if args.split == "all" else [args.split]
    scns = [s for sp in splits for s in scenarios.build_split(sp)]
    if args.limit:
        scns = scns[: args.limit]

    country_years, est_seconds = _plan(scns)
    print(f"{len(scns)} scenarios, {country_years} country-years, "
          f"estimated {est_seconds / 60 / args.workers:.0f} min "
          f"at {args.workers} workers")
    print(f"spec_hash: {scenarios.spec_hash(scns)}")

    if args.dry_run:
        return 0

    if args.seal_only:
        for sp in splits:
            sp_scns = scenarios.build_split(sp)
            missing = [s.sid for s in sp_scns
                       if not (root / sp / s.sid / "media.csv").is_file()]
            if missing:
                print(f"ERROR: {sp} is not generated ({len(missing)} of "
                      f"{len(sp_scns)} scenarios missing); generate before "
                      f"sealing", file=sys.stderr)
                return 2
            info = seal.seal_split(sp, sp_scns, root=root)
            print(f"sealed {sp}: {info['n_scenarios']} scenarios, "
                  f"spec_hash {info['spec_hash'][:16]}")
        return 0

    started = time.time()

    def progress(res, done, total):
        flag = {"generated": "ok", "skipped": "--", "failed": "FAIL"}[res["status"]]
        print(f"[{done:3}/{total}] {flag:4} {res['sid']:32} {res['seconds']:6.1f}s")
        if res["status"] == "failed":
            print(f"        {res['error'][:400]}", file=sys.stderr)

    results = runner.run_many(scns, root=root, workers=args.workers,
                              force=args.force, on_done=progress)

    failed = [r for r in results if r["status"] == "failed"]
    print(f"\ndone in {(time.time() - started) / 60:.1f} min: "
          f"{sum(r['status'] == 'generated' for r in results)} generated, "
          f"{sum(r['status'] == 'skipped' for r in results)} skipped, "
          f"{len(failed)} failed")

    if failed:
        print("failed scenarios: " + ", ".join(r["sid"] for r in failed),
              file=sys.stderr)
        return 1

    if args.seal:
        for sp in splits:
            sp_scns = scenarios.build_split(sp)
            if not (root / sp).is_dir():
                print(f"ERROR: {sp} is not generated", file=sys.stderr)
                return 2
            info = seal.seal_split(sp, sp_scns, root=root)
            print(f"sealed {sp}: {info['n_scenarios']} scenarios, "
                  f"spec_hash {info['spec_hash'][:16]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests and make sure they pass**

Run: `synthetic_data_generator/.venv/bin/python -m pytest tests/benchmark/test_generate_cli.py -v`
Expected: 3 passed.

- [ ] **Step 5: Smoke-test the real path on four scenarios**

```bash
cd /Users/user123/Desktop/projects/datascience-project
rm -rf /tmp/bench_smoke
synthetic_data_generator/.venv/bin/python -m benchmark.harness.generate \
  --split dev --limit 4 --workers 4 --root /tmp/bench_smoke
ls /tmp/bench_smoke/dev/*/ /tmp/bench_smoke/dev_truth/*/
```
Expected: 4 generated, 0 failed. Each `dev/<sid>/` holds exactly `media.csv` and `sales.csv`; each `dev_truth/<sid>/` holds `ground_truth.csv`, `meta.json`, `scenario.json`, `true_roi.csv`.

- [ ] **Step 6: Commit**

```bash
cd /Users/user123/Desktop/projects/datascience-project
git add benchmark/harness/generate.py tests/benchmark/test_generate_cli.py
git commit -m "feat(benchmark): add generation CLI with resume, progress, and sealing"
```

---

### Task 11: Generate and seal the real benchmark

The one task that is an operation rather than a code change. Roughly 45 minutes of wall-clock.

**Files:**
- Create: `benchmark/datasets/**` (generated; CSVs are gitignored, manifests and seals are not)
- Create: `benchmark/BENCHMARK.md`
- Test: `tests/benchmark/test_frozen_benchmark.py`

**Interfaces:**
- Consumes: the CLI from Task 10.
- Produces: the frozen benchmark on disk, and `benchmark/BENCHMARK.md` describing it. Plans 2 and 3 both read from `benchmark/datasets/`.

- [ ] **Step 1: Run the full fast test suite first**

Run: `synthetic_data_generator/.venv/bin/python -m pytest tests -v -m "not slow"`
Expected: all pass. Do not start a 45-minute job on a red suite.

- [ ] **Step 2: Print the plan and sanity-check the estimate**

```bash
cd /Users/user123/Desktop/projects/datascience-project
synthetic_data_generator/.venv/bin/python -m benchmark.harness.generate \
  --split all --dry-run
```
Expected: `100 scenarios`, and an estimate under 70 minutes. Record the `spec_hash` — it must match what ends up in the SEALED marker.

- [ ] **Step 3: Generate the development split**

```bash
cd /Users/user123/Desktop/projects/datascience-project
synthetic_data_generator/.venv/bin/python -m benchmark.harness.generate \
  --split dev --workers 6 2>&1 | tee /tmp/gen_dev.log
```
Expected: 45 generated, 0 failed. If any scenario fails, read its error in the log, fix the cause in `benchmark/spec/`, and rerun — the run is resumable and will skip what already succeeded.

- [ ] **Step 4: Generate the test split and seal it**

```bash
cd /Users/user123/Desktop/projects/datascience-project
synthetic_data_generator/.venv/bin/python -m benchmark.harness.generate \
  --split test --workers 6 2>&1 | tee /tmp/gen_test.log
synthetic_data_generator/.venv/bin/python -m benchmark.harness.generate \
  --split test --seal-only --root benchmark/datasets
```
Expected: 55 generated, 0 failed, then `sealed test: 55 scenarios, spec_hash …`.

**From this point the test split is frozen.** Do not regenerate it, do not open any file under `benchmark/datasets/test_truth/`, and do not read `ground_truth.csv` for any test scenario until Plan 3's final evaluation.

- [ ] **Step 5: Write the frozen-benchmark test**

Create `tests/benchmark/test_frozen_benchmark.py`:

```python
"""Standing guarantees about the benchmark on disk. These run on every suite
invocation from here on, so a later change that quietly breaks the seal or
leaks ground truth into a dataset directory fails immediately."""
import json
from pathlib import Path

import pytest

from benchmark.harness import runner, seal
from benchmark.spec import scenarios

DATASETS = runner.DATASETS_DIR
pytestmark = pytest.mark.skipif(not (DATASETS / "test" / "SEALED").is_file(),
                                reason="benchmark not generated yet")


@pytest.mark.parametrize("split,expected", [("dev", 45), ("test", 55)])
def test_every_scenario_was_generated(split, expected):
    dirs = [p for p in (DATASETS / split).iterdir() if p.is_dir()]
    assert len(dirs) == expected


@pytest.mark.parametrize("split", ["dev", "test"])
def test_dataset_directories_leak_nothing(split):
    """The black-box guarantee, checked against what is actually on disk."""
    for d in (DATASETS / split).iterdir():
        if d.is_dir():
            assert sorted(p.name for p in d.iterdir()) == ["media.csv", "sales.csv"], d


@pytest.mark.parametrize("split", ["dev", "test"])
def test_truth_directories_are_complete(split):
    for s in scenarios.build_split(split):
        t = runner.truth_dir(split, s.sid)
        assert (t / "ground_truth.csv").is_file(), s.sid
        assert (t / "meta.json").is_file(), s.sid


def test_test_split_seal_still_verifies():
    ok, problems = seal.verify_seal("test")
    assert ok, problems


def test_seal_records_the_current_spec_hash():
    marker = json.loads((DATASETS / "test" / "SEALED").read_text())
    assert marker["spec_hash"] == scenarios.spec_hash(scenarios.build_split("test"))
    assert marker["n_scenarios"] == 55


def test_dev_split_is_not_sealed():
    """Dev must stay writable -- it is where iteration is allowed to happen."""
    assert not seal.is_sealed("dev")
```

- [ ] **Step 6: Run it**

Run: `synthetic_data_generator/.venv/bin/python -m pytest tests/benchmark/test_frozen_benchmark.py -v`
Expected: 9 passed.

If `test_seal_records_the_current_spec_hash` fails, the scenario code changed after sealing. **Do not reseal.** Restore the scenario code to the sealed state; the seal is the record of what the benchmark actually is.

- [ ] **Step 7: Write `benchmark/BENCHMARK.md`**

```bash
cd /Users/user123/Desktop/projects/datascience-project
synthetic_data_generator/.venv/bin/python - > benchmark/BENCHMARK.md <<'PY'
import json
from collections import Counter
from benchmark.harness import runner
from benchmark.spec import scenarios

marker = json.loads((runner.DATASETS_DIR / "test" / "SEALED").read_text())
allsc = scenarios.build_all()

print("# The Frozen Benchmark\n")
print(f"100 scenarios: 45 development, 55 hold-out test. Test split sealed "
      f"{marker['sealed_at']}, spec_hash `{marker['spec_hash']}`.\n")
print("Dataset directories hold only `media.csv` and `sales.csv`. Ground truth, "
      "`meta.json`, `scenario.json` and `true_roi.csv` live under "
      "`<split>_truth/`, which detector code must never import.\n")
print("## Regenerating\n")
print("```bash\npython -m benchmark.harness.generate --split dev\n"
      "python -m benchmark.harness.generate --split test --seal\n```\n")
print("Generation is deterministic: the same code produces the same 100 "
      "scenarios, so `spec_hash` is a full description of the benchmark.\n")
print("## Families\n")
print("| family | dev | test |\n|---|---|---|")
for fam, (d, t) in scenarios.FAMILY_COUNTS.items():
    print(f"| {fam} | {d} | {t} |")
print(f"| **total** | **45** | **55** |\n")
print("## Coverage\n")
for axis in ["noise_level", "market_spread", "n_countries", "n_channels",
             "trend_p", "years"]:
    counts = Counter(s.meta[axis] for s in allsc)
    print(f"- **{axis}**: " + ", ".join(f"`{k}`x{v}" for k, v in sorted(
        counts.items(), key=lambda kv: str(kv[0]))))
print("\n## Negative controls\n")
print("Two pattern types change the data but must never be reported as "
      "detections; anything found inside their windows counts as a false "
      "positive:\n")
print("- `ramp_block` — gradual drift, not a step change")
print("- `intermittent_baseline` — a flighting channel whose short gaps are normal\n")
print("Plus 9 `null` scenarios carrying no events at all, which are the only "
      "way to measure a false-positive rate.")
PY
head -40 benchmark/BENCHMARK.md
```

- [ ] **Step 8: Commit the frozen benchmark**

```bash
cd /Users/user123/Desktop/projects/datascience-project
git add benchmark/BENCHMARK.md tests/benchmark/test_frozen_benchmark.py
git add -f benchmark/datasets/test/SEALED \
           benchmark/datasets/test/manifest.sha256 \
           benchmark/datasets/test_truth/manifest.sha256
git commit -m "feat(benchmark): freeze and seal the 100-scenario benchmark

Test split sealed; ground truth is not to be read until the algorithms are
final and Plan 3's gated final evaluation runs."
```

---

## Verification

Run the whole suite, fast and slow, and confirm the benchmark is intact:

```bash
cd /Users/user123/Desktop/projects/datascience-project
synthetic_data_generator/.venv/bin/python -m pytest tests -v
synthetic_data_generator/.venv/bin/python -c "
from benchmark.harness import seal
ok, problems = seal.verify_seal('test')
print('SEAL OK' if ok else f'SEAL BROKEN: {problems}')
"
```

Expected: every test passes and `SEAL OK`.

## What Plan 2 picks up

Plan 2 builds `benchmark/eval/` — the truth loader (normalising the off-by-one
`end_date` and dropping `NON_EVENT_TYPES` from the matchable set), interval
matching, the ten metrics from spec §9, and the `run_dev` / `run_final`
entry points, all validated against a deliberately trivial detector before any
real algorithm exists.
