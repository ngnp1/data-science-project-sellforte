import json
import pytest
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


# Every import path by which detector code could reach the answer key.
#
# "benchmark.spec" is here because it is the answer key: it builds the events
# the generator injects, so importing it hands the detector the event families
# and their parameters directly. It also needs NO file access, which means the
# runtime open() guard in tests/detection/test_pipeline.py cannot see it --
# this static scan is the only thing that can. "benchmark.harness" carries
# runner.truth_dir, which resolves a truth path without either marker string
# appearing in the caller.
FORBIDDEN_IMPORTS = (
    "benchmark.eval",
    "benchmark.spec",
    "benchmark.harness",
    "_truth",
    "ground_truth",
)


def _truth_reaching_imports(text: str) -> list[str]:
    """The needles present in `text`. Shared by the scan and its control."""
    return [needle for needle in FORBIDDEN_IMPORTS if needle in text]


def test_detection_package_has_no_import_path_to_truth():
    """Spec section 4: detector code must never be able to import a truth
    loader, or anything that is equivalent to one."""
    det = ROOT / "detection"
    if not det.is_dir():
        return
    offenders = []
    for f in det.rglob("*"):
        if not f.is_file() or "__pycache__" in f.parts:
            continue
        try:
            text = f.read_text()
        except UnicodeDecodeError:
            # A binary file under detection/ cannot be source, and a data blob
            # is exactly what the companion check below exists to catch.
            continue
        found = _truth_reaching_imports(text)
        if found:
            offenders.append(f"{f.relative_to(ROOT)}: {found}")
    assert not offenders, f"detection/ reaches for truth: {offenders}"


# One file, by name, is allowed to be Markdown instead of Python: a README
# a reader opens to find the pipeline and the primitives without opening the
# code. It holds no data and nothing here can import it as a smuggled answer
# table, so it is a narrow exemption, not a loophole -- the test below proves
# the guard still catches everything else.
NON_SOURCE_EXEMPT_FILENAMES = {"README.md"}


def test_the_detection_package_contains_only_source():
    """The import scan and the final-run audit hash both used to look at *.py
    alone. A precomputed answer table dropped under detection/ as JSON, CSV or
    Parquet would have been read by the detector, scanned by neither, and
    covered by no hash -- a way to smuggle in the answers that leaves no trace
    in the audit record. detection/ is source; nothing else belongs there,
    except the one exempt file named above."""
    det = ROOT / "detection"
    if not det.is_dir():
        return
    intruders = [str(f.relative_to(ROOT)) for f in det.rglob("*")
                 if f.is_file() and f.suffix != ".py"
                 and f.name not in NON_SOURCE_EXEMPT_FILENAMES
                 and "__pycache__" not in f.parts]
    assert not intruders, f"non-source files under detection/: {intruders}"


def test_the_non_source_exemption_is_exactly_one_file():
    """A broad exemption would let a real answer table hide behind it. This
    pins the set to exactly the one file it names."""
    assert NON_SOURCE_EXEMPT_FILENAMES == {"README.md"}


def test_the_only_source_guard_still_catches_a_real_intruder():
    """Positive control: a JSON file under detection/ must still fail the
    guard above. Proves the exemption is narrow rather than a hole that
    happens to admit only what exists today."""
    det = ROOT / "detection"
    intruder = det / "_gating_test_intruder.json"
    intruder.write_text("{}")
    try:
        offenders = [str(f.relative_to(ROOT)) for f in det.rglob("*")
                     if f.is_file() and f.suffix != ".py"
                     and f.name not in NON_SOURCE_EXEMPT_FILENAMES
                     and "__pycache__" not in f.parts]
        assert "detection/_gating_test_intruder.json" in offenders
    finally:
        intruder.unlink()


@pytest.mark.parametrize("line", [
    "from benchmark.eval.truth import load_truth",
    "from benchmark.spec.events import dark",
    "from benchmark.harness.runner import truth_dir",
    "path = DATASETS / 'dev_truth' / sid",
    "df = pd.read_csv('ground_truth.csv')",
])
def test_the_import_scan_would_catch_a_violation(line):
    """Positive control. A scan that cannot fail is not a gate, and this one
    silently had a hole: benchmark.spec was unbanned for the whole of Plan 3.
    Each line below must be caught by the scan the test above runs."""
    assert _truth_reaching_imports(line), f"scan misses: {line}"


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

    # Capture the real final_runs.jsonl state before the test, to verify
    # the test does not modify it (immutability check).
    real_final_runs_path = ROOT / "benchmark/eval/final_runs.jsonl"
    real_final_runs_state = real_final_runs_path.read_text() if real_final_runs_path.exists() else None

    fake_final_runs = tmp_path / "final_runs.jsonl"
    monkeypatch.setattr(run_final, "FINAL_RUNS", fake_final_runs)

    # No prior records: this would be run #1, and there is nothing to
    # announce.
    assert run_final._prior_run_count() == 0
    assert run_final._repeat_run_banner(1, 0) == ""

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

    # Where the banner ends up in the composed report, and that the record is
    # self-describing, are properties of run_final itself -- see
    # test_compose_report_puts_the_repeat_banner_above_the_report and
    # test_build_record_is_self_describing below. Asserting them against
    # locally built strings and dicts here (the earlier
    # `assert (banner + body).startswith(banner)` and
    # `assert {"run_index": 2}["run_index"] == 2`) tested Python, not this
    # program, and could not fail whatever run_final did.

    # A third record makes it #3, not stuck at #2.
    fake_final_runs.write_text(
        json.dumps({"run_index": 1}) + "\n" + json.dumps({"run_index": 2}) + "\n")
    prior_count = run_final._prior_run_count()
    assert prior_count == 2
    assert "#3" in run_final._repeat_run_banner(prior_count + 1, prior_count)

    # The real final_runs.jsonl must not be modified by this test. It may exist
    # (after a prior final run), but its contents must be unchanged. This leak
    # check catches if a test ever silently appended to the audit record.
    assert (real_final_runs_path.read_text() if real_final_runs_path.exists() else None) == real_final_runs_state


def _fake_results() -> dict:
    """The smallest results dict `compose_report` and `build_record` accept."""
    return {
        "split": "test",
        "n_scenarios": 3,
        "overall": {"precision": 1.0, "recall": 1.0, "f1": 1.0,
                    "n_tp": 3, "n_fp": 0, "n_fn": 0},
        "per_type": {},
        "iou": {"mean_iou": 1.0, "median_iou": 1.0, "n": 3},
        "boundary": {"start_median": 0.0, "start_p90": 0.0,
                     "end_median": 0.0, "end_p90": 0.0, "n": 3},
        "accuracy": {"channel_accuracy": 1.0, "market_accuracy": 1.0,
                     "n_relaxed_matches": 3},
        "day_level": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
        "confusion": {}, "reliability": [], "operating": [],
        "null_fp_rate": 0.0, "null_country_years": 4.0,
        "per_scenario": {}, "breakdowns": {}, "event_breakdowns": {},
    }


def _stub_the_gated_path(monkeypatch, tmp_path, results):
    """Replace everything in run_final.main that would touch the sealed split.

    The seal check, the SEALED marker and evaluate_split are all stubbed, and
    PROJECT_ROOT is redirected into tmp_path, so nothing here reads
    benchmark/datasets/test/ or its truth. The point of the test below is the
    ORDER of the tail of main(), which needs none of that to be real.
    """
    from benchmark.eval import run_final

    fake_root = tmp_path / "root"
    (fake_root / "benchmark/datasets/test").mkdir(parents=True)
    (fake_root / "benchmark/datasets/test/SEALED").write_text(
        json.dumps({"spec_hash": "deadbeef", "sealed_at": "2026-01-01T00:00:00Z"}))

    monkeypatch.setattr(run_final, "PROJECT_ROOT", fake_root)
    monkeypatch.setattr(run_final.seal, "verify_seal", lambda split: (True, []))
    monkeypatch.setattr(run_final, "load_detector", lambda spec: (lambda *a: []))
    monkeypatch.setattr(run_final, "evaluate_split",
                        lambda *a, **k: results)

    fake_final_runs = tmp_path / "final_runs.jsonl"
    monkeypatch.setattr(run_final, "FINAL_RUNS", fake_final_runs)
    return run_final, fake_final_runs


def test_a_failing_out_path_still_records_the_final_run(tmp_path, monkeypatch):
    """C1: the audit record must survive a mistyped --out.

    By the time --out is written the SEALED SPLIT HAS ALREADY BEEN READ and
    the metrics have already been printed to the terminal. If the write raises
    before final_runs.jsonl is appended to, the seal has been spent with no
    trace: the next run reads a prior count of 0, records run_index 1, and its
    report claims to be the first final evaluation of the hold-out split.
    Spec sections 4.3 and 10 exist to make that impossible.

    Never goes near the real gated path -- seal verification, the SEALED
    marker, the detector and evaluate_split are all stubbed, and FINAL_RUNS
    points into tmp_path.
    """
    import pytest

    # Capture the real final_runs.jsonl state before the test, to verify
    # the test does not modify it (immutability check).
    real_final_runs_path = ROOT / "benchmark/eval/final_runs.jsonl"
    real_final_runs_state = real_final_runs_path.read_text() if real_final_runs_path.exists() else None

    run_final, fake_final_runs = _stub_the_gated_path(
        monkeypatch, tmp_path, _fake_results())

    # A directory that does not exist: write_text raises.
    bad_out = tmp_path / "no" / "such" / "dir" / "report.md"

    with pytest.raises(OSError):
        run_final.main(["--detector", "x:y", "--finalize", "--out", str(bad_out)])

    assert not bad_out.exists()
    assert fake_final_runs.is_file(), (
        "the sealed split was read and the report printed, but no audit "
        "record was written -- the next final run would call itself the first")
    lines = fake_final_runs.read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["run_index"] == 1

    # And the NEXT run therefore knows it is a repeat.
    assert run_final._prior_run_count() == 1
    # The real final_runs.jsonl must not be modified by this test. It may exist
    # (after a prior final run), but its contents must be unchanged. This leak
    # check catches if a test ever silently appended to the audit record.
    assert (real_final_runs_path.read_text() if real_final_runs_path.exists() else None) == real_final_runs_state


def test_compose_report_puts_the_repeat_banner_above_the_report(
        tmp_path, monkeypatch):
    """The composition itself, not a restatement of string concatenation.

    Replaces the earlier tautologies (`assert (a + b).startswith(a)`), which
    tested Python rather than this program: they passed whether or not
    main() actually used the banner.
    """
    from benchmark.eval import run_final

    monkeypatch.setattr(run_final, "FINAL_RUNS", tmp_path / "final_runs.jsonl")
    results = _fake_results()

    first = run_final.compose_report(results, run_index=1, prior_count=0)
    assert first.startswith("# Evaluation")
    assert "REPEAT FINAL RUN" not in first

    repeat = run_final.compose_report(results, run_index=2, prior_count=1)
    assert repeat.startswith("> **REPEAT FINAL RUN")
    assert "#2" in repeat.splitlines()[0]
    assert "1 prior" in repeat.splitlines()[0]
    assert "final_runs.jsonl" in repeat.splitlines()[0]
    # The body is still all there, below the banner.
    assert repeat.endswith(first)


def test_build_record_is_self_describing(tmp_path, monkeypatch):
    """The record must carry run_index itself, rather than leaving a reader to
    go count lines in final_runs.jsonl."""
    import argparse

    run_final, fake_final_runs = _stub_the_gated_path(
        monkeypatch, tmp_path, _fake_results())
    args = argparse.Namespace(detector="pkg.mod:fn")
    marker = {"spec_hash": "deadbeef", "sealed_at": "2026-01-01T00:00:00Z"}

    record = run_final.build_record(_fake_results(), args, 2, marker)
    assert record["run_index"] == 2
    assert record["detector"] == "pkg.mod:fn"
    assert record["sealed_spec_hash"] == "deadbeef"
    assert record["metrics"]["f1"] == 1.0
    assert "detection_source_hash" in record and "timestamp" in record
