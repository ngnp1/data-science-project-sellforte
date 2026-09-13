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
    # Built with getattr so that even a malformed `scenario` can't raise here --
    # every path below must return a record, not propagate an exception.
    result = {"sid": getattr(scenario, "sid", None),
              "split": getattr(scenario, "split", None),
              "seed": getattr(scenario, "seed", None),
              "status": "failed", "seconds": 0.0, "error": None}

    try:
        if not force and _is_complete(scenario.split, scenario.sid, root):
            result["status"] = "skipped"
        else:
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
    """Generate many scenarios in parallel. Subprocess-bound, so threads suffice.

    No exception from an individual future -- however unexpected -- is allowed
    to abort the loop or discard results already collected. Anything that
    escapes `run_scenario` (it shouldn't, but see its own docstring caveat) is
    converted into the same kind of failed record here instead.
    """
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_scenario, s, root, force): s for s in scenarios}
        for fut in as_completed(futures):
            scenario = futures[fut]
            try:
                res = fut.result()
            except Exception as exc:  # noqa: BLE001 -- one bad worker must not sink the batch
                res = {"sid": getattr(scenario, "sid", None),
                       "split": getattr(scenario, "split", None),
                       "seed": getattr(scenario, "seed", None),
                       "status": "failed", "seconds": 0.0,
                       "error": f"{type(exc).__name__}: {exc}"}
            results.append(res)
            if on_done:
                on_done(res, len(results), len(futures))
    return sorted(results, key=lambda r: r["sid"])
