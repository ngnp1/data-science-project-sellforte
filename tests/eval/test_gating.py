import json
import subprocess
from pathlib import Path

PY = "synthetic_data_generator/.venv/bin/python"
ROOT = Path(__file__).resolve().parents[2]


def run(module, *args):
    return subprocess.run([PY, "-m", module, *args], cwd=ROOT,
                          capture_output=True, text=True)


def test_final_refuses_without_the_finalize_flag():
    """The gate. Running the sealed test split must be a deliberate act."""
    p = run("benchmark.eval.run_final",
            "--detector", "benchmark.eval.detectors_for_testing:never_detect")
    assert p.returncode != 0
    assert "--finalize" in (p.stdout + p.stderr)


def test_final_names_the_seal_check_in_its_help():
    p = run("benchmark.eval.run_final", "--help")
    assert p.returncode == 0
    assert "seal" in (p.stdout + p.stderr).lower()


def test_dev_runs_freely_and_appends_history():
    hist_before = 0
    hist = ROOT / "benchmark/eval/dev_history.jsonl"
    if hist.is_file():
        hist_before = len(hist.read_text().splitlines())

    p = run("benchmark.eval.run_dev",
            "--detector", "benchmark.eval.detectors_for_testing:perfect_oracle",
            "--limit", "5", "--label", "harness-selftest")
    assert p.returncode == 0, p.stderr
    assert "f1" in (p.stdout.lower())

    after = len(hist.read_text().splitlines())
    assert after == hist_before + 1
    last = json.loads(hist.read_text().splitlines()[-1])
    for key in ["timestamp", "label", "detector", "source_hash", "metrics"]:
        assert key in last
    assert last["metrics"]["f1"] == 1.0


def test_detection_package_has_no_import_path_to_truth():
    """Spec section 4: detector code must never be able to import a truth
    loader. detection/ does not exist yet -- when Plan 3 creates it, this test
    is what keeps the boundary real."""
    det = ROOT / "detection"
    if not det.is_dir():
        return
    offenders = []
    for py in det.rglob("*.py"):
        text = py.read_text()
        if "benchmark.eval" in text or "_truth" in text or "ground_truth" in text:
            offenders.append(str(py.relative_to(ROOT)))
    assert not offenders, f"detection/ reaches for truth: {offenders}"
