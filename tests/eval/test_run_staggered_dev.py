from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from benchmark.eval import run_staggered_dev as command
from benchmark.eval.adapter import to_event
from detection.model import DetectedEvent


@pytest.fixture
def configured(monkeypatch):
    def build(split):
        assert split == "dev"
        return [SimpleNamespace(sid="dev_001"), SimpleNamespace(sid="dev_002")]
    monkeypatch.setattr(command, "build_split", build)


def test_preflight_detects_missing_scenario_even_when_directory_is_absent(configured, monkeypatch, tmp_path):
    checks = []
    def complete(split, sid, root):
        assert split == "dev"
        checks.append(sid)
        return sid == "dev_001"
    monkeypatch.setattr(command, "_is_complete", complete)
    monkeypatch.setattr(command, "evaluate_split", lambda *a, **k: pytest.fail("must not evaluate"))
    out = tmp_path / "report"
    with pytest.raises(SystemExit) as error:
        command.main(["--out", str(out)])
    assert error.value.code == 2
    assert checks == ["dev_001", "dev_002"]
    assert not out.exists()


@pytest.mark.parametrize("excluded", [["test_001"], ["typo"], ["dev_001", "dev_002"]])
def test_invalid_or_empty_selection_is_rejected(configured, monkeypatch, tmp_path, excluded):
    monkeypatch.setattr(command, "evaluate_split", lambda *a, **k: pytest.fail("must not evaluate"))
    with pytest.raises(SystemExit) as error:
        command.main(["--exclude", *excluded, "--out", str(tmp_path / "report")])
    assert error.value.code == 2


def test_report_uses_adapter_exports_ranked_launches_and_preserves_tracked_history(
        configured, monkeypatch, tmp_path):
    ready = DetectedEvent(
        "dev_001", "AT", "TV", "staggered_launch",
        pd.Timestamp("2024-01-01"), pd.Timestamp("2024-03-10"),
        detection_confidence=0.9, informativeness=0.6,
        evidence={"onset": "2024-03-11", "confidence_kind": "heuristic",
                  "launch_verification": {"status": "analysis_ready", "reason_codes": [],
                                          "qualified_peers": ["CH"], "peers": []}})
    rejected = replace(ready, country_code="DE", informativeness=1.0,
                       evidence={**ready.evidence, "launch_verification": {
                           "status": "not_analysis_ready", "reason_codes": ["insufficient_history"],
                           "qualified_peers": [], "peers": []}})
    holdout = replace(ready, event_type="natural_holdout")
    calls = []
    def complete(split, sid, root):
        assert split == "dev" and sid == "dev_001"
        return True
    def evaluate(detector, split, **kwargs):
        assert split == "dev"
        assert kwargs["sids"] == ["dev_001"]
        assert kwargs["load_data"] is True
        predictions = detector(None, None, "dev_001")
        assert predictions == [to_event(rejected), to_event(holdout), to_event(ready)]
        assert all(event.detection_confidence is None for event in predictions)
        calls.append(split)
        return {"overall": {"n_tp": 1}, "confusion": {("staggered_launch", "staggered_launch"): 1}}
    def truth(split, sid, root):
        assert split == "dev" and sid == "dev_001"
        return [to_event(ready)]
    monkeypatch.setattr(command, "_is_complete", complete)
    monkeypatch.setattr(command, "evaluate_split", evaluate)
    monkeypatch.setattr(command, "run_detection", lambda *a: [rejected, holdout, ready])
    monkeypatch.setattr(command.T, "load_truth", truth)

    protected = [command.ROOT / "benchmark/eval/dev_history.jsonl",
                 command.ROOT / "benchmark/eval/final_runs.jsonl"]
    protected += list((command.ROOT / "benchmark/datasets").glob("test*/manifest.sha256"))
    protected += [command.ROOT / "benchmark/datasets/test/SEALED"]
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected if p.is_file()}
    out = tmp_path / "report"
    args = ["--exclude", "dev_002", "--out", str(out)]
    assert command.main(args) == 0
    assert command.main(args) == 0
    assert calls == ["dev", "dev"]
    assert {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in before} == before
    assert {p.name for p in out.iterdir()} == {
        "metrics.json", "launch-summary.json", "run-metadata.json", "predictions.jsonl",
        "report.md", "run-history.jsonl"}
    metadata = json.loads((out / "run-metadata.json").read_text())
    assert metadata["selected_scenarios"] == ["dev_001"]
    assert metadata["excluded_scenarios"] == ["dev_002"]
    assert metadata["evaluation_status"] == "completed"
    assert metadata["source_hash"] and metadata["timestamp"]
    summary = json.loads((out / "launch-summary.json").read_text())
    assert [row["readiness"] for row in summary] == ["analysis_ready", "not_analysis_ready"]
    assert summary[0]["launch_date"] == "2024-03-11"
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["detection"]["metrics"] == {
        "precision": 0.5, "recall": 1.0, "f1": 2 / 3, "n_tp": 1, "n_fp": 1, "n_fn": 0}
    assert metrics["readiness"]["reason_codes"] == {"insufficient_history": 1}
    assert len((out / "run-history.jsonl").read_text().splitlines()) == 2
    assert len((out / "predictions.jsonl").read_text().splitlines()) == 3
    report = (out / "report.md").read_text()
    assert "not readiness accuracy" in report
    assert "Every launch is retained" in report
