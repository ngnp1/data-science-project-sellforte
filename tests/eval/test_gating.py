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


def test_dev_runs_freely_and_appends_history(tmp_path):
    """The dev split is free to run as often as you like, but the TEST must
    never write to the tracked benchmark/eval/dev_history.jsonl -- that file
    is a real development trajectory, appended to by actual dev-loop runs
    (Plan 3 onward), and a test dirtying it on every invocation would pollute
    the working tree and, eventually, the commit history. --history redirects
    the append target, so this test starts from a known-empty file rather
    than counting deltas on a file other things may also be writing to."""
    hist = tmp_path / "dev_history.jsonl"
    assert not hist.exists()

    p = run("benchmark.eval.run_dev",
            "--detector", "benchmark.eval.detectors_for_testing:perfect_oracle",
            "--limit", "5", "--label", "harness-selftest",
            "--history", str(hist))
    assert p.returncode == 0, p.stderr
    assert "f1" in (p.stdout.lower())

    lines = hist.read_text().splitlines()
    assert len(lines) == 1
    last = json.loads(lines[0])
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


def test_repeat_final_run_banners_the_report_and_increments_run_index(
        tmp_path, monkeypatch):
    """run_final.py's own docstring says a second run is 'permanently visible
    in final_runs.jsonl, and the report is expected to say so' -- but the
    original code only ever wrote that notice to stderr, which never reaches
    the rendered Markdown, the --out file, or the JSONL record itself. This
    exercises the fix directly against the extracted helpers
    (_prior_run_count, _repeat_run_banner) and the run_index field, WITHOUT
    ever invoking run_final.main() with --finalize: that path verifies the
    seal and reads the real sealed test split, which must never run as a side
    effect of a test. FINAL_RUNS is monkeypatched to a tmp_path file that is
    never near benchmark/eval/final_runs.jsonl.
    """
    from benchmark.eval import run_final

    fake_final_runs = tmp_path / "final_runs.jsonl"
    monkeypatch.setattr(run_final, "FINAL_RUNS", fake_final_runs)

    # No prior records: this would be run #1, and there is nothing to
    # announce.
    assert run_final._prior_run_count() == 0
    assert run_final._repeat_run_banner(1, 0) == ""
    report = run_final._repeat_run_banner(1, 0) + "# Evaluation body"
    assert report == "# Evaluation body"

    # Seed one fake prior record -- never touches the real final_runs.jsonl.
    fake_final_runs.write_text(json.dumps({"run_index": 1}) + "\n")

    prior_count = run_final._prior_run_count()
    assert prior_count == 1
    run_index = prior_count + 1
    assert run_index == 2

    banner = run_final._repeat_run_banner(run_index, prior_count)
    assert "#2" in banner
    assert "1 prior" in banner
    assert "final_runs.jsonl" in banner

    # The banner must sit at the very top of the composed report.
    report = banner + "# Evaluation body"
    assert report.startswith(banner)
    assert report.index(banner) == 0
    assert "# Evaluation body" in report

    # And the record itself must be self-describing -- run_index, not just a
    # line count a reader has to go compute.
    record = {"run_index": run_index}
    assert record["run_index"] == 2

    # A third record makes it #3, not stuck at #2.
    fake_final_runs.write_text(
        json.dumps({"run_index": 1}) + "\n" + json.dumps({"run_index": 2}) + "\n")
    prior_count = run_final._prior_run_count()
    assert prior_count == 2
    assert "#3" in run_final._repeat_run_banner(prior_count + 1, prior_count)

    # The real final_runs.jsonl must still not exist as a result of this test.
    assert not (ROOT / "benchmark/eval/final_runs.jsonl").exists()
