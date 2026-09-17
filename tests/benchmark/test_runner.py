from types import SimpleNamespace

import pandas as pd
import pytest

from benchmark.harness import runner
from benchmark.spec import scenarios


def _fake_scenario(sid, split="dev", seed=1):
    """A stand-in for a Scenario that only carries what run_scenario/run_many
    touch on the failure paths (sid, split, seed) -- cheap enough that these
    tests need no real generator run and no `slow` marker."""
    return SimpleNamespace(sid=sid, split=split, seed=seed)


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


def test_run_scenario_reports_failure_instead_of_raising_when_is_complete_blows_up():
    """Reviewer's reproduction: a bad `root` makes the internal `_is_complete`
    path arithmetic raise (Path(12345) is a TypeError) before any subprocess
    is ever launched. run_scenario must still come back as a record, not
    propagate the exception -- this is fast because it never reaches R."""
    s = _fake_scenario("bad_root_001")
    result = runner.run_scenario(s, root=12345)
    assert result["status"] == "failed"
    assert result["sid"] == "bad_root_001"
    assert result["error"]
    assert "seconds" in result and result["seconds"] >= 0.0


def test_run_many_survives_a_worker_raising_and_keeps_the_rest(monkeypatch):
    """No exception from one future may propagate out of run_many, and no
    already-collected result may be discarded because of it."""
    def fake_run_scenario(scenario, root, force):
        if scenario.sid == "bad":
            raise RuntimeError("boom")
        return {"sid": scenario.sid, "split": scenario.split, "seed": scenario.seed,
                "status": "generated", "seconds": 0.0, "error": None}

    monkeypatch.setattr(runner, "run_scenario", fake_run_scenario)

    scenarios_in = [_fake_scenario(sid) for sid in ["c", "a", "bad", "b"]]
    results = runner.run_many(scenarios_in, workers=2)  # must not raise

    assert {r["sid"] for r in results} == {"a", "b", "bad", "c"}
    assert len(results) == 4  # one result per input, none dropped


def test_run_many_converts_the_raised_exception_into_a_failed_record(monkeypatch):
    def fake_run_scenario(scenario, root, force):
        if scenario.sid == "bad":
            raise RuntimeError("boom")
        return {"sid": scenario.sid, "split": scenario.split, "seed": scenario.seed,
                "status": "generated", "seconds": 0.0, "error": None}

    monkeypatch.setattr(runner, "run_scenario", fake_run_scenario)

    results = runner.run_many([_fake_scenario("ok"), _fake_scenario("bad")], workers=2)

    bad = next(r for r in results if r["sid"] == "bad")
    assert bad["status"] == "failed"
    assert "boom" in bad["error"]
    ok = next(r for r in results if r["sid"] == "ok")
    assert ok["status"] == "generated"


def test_run_many_invokes_on_done_once_per_completion_including_failures(monkeypatch):
    def fake_run_scenario(scenario, root, force):
        if scenario.sid == "boom":
            raise RuntimeError("nope")
        return {"sid": scenario.sid, "split": scenario.split, "seed": scenario.seed,
                "status": "generated", "seconds": 0.0, "error": None}

    monkeypatch.setattr(runner, "run_scenario", fake_run_scenario)

    seen = []
    scenarios_in = [_fake_scenario(sid) for sid in ["ok1", "boom", "ok2"]]
    runner.run_many(
        scenarios_in, workers=3,
        on_done=lambda res, done, total: seen.append((res["sid"], done, total)))

    assert len(seen) == 3  # once per completion, converted failures included
    assert {sid for sid, _, _ in seen} == {"ok1", "boom", "ok2"}
    assert {done for _, done, _ in seen} == {1, 2, 3}
    assert all(total == 3 for _, _, total in seen)


def test_run_many_returns_results_sorted_by_sid(monkeypatch):
    def fake_run_scenario(scenario, root, force):
        return {"sid": scenario.sid, "split": scenario.split, "seed": scenario.seed,
                "status": "generated", "seconds": 0.0, "error": None}

    monkeypatch.setattr(runner, "run_scenario", fake_run_scenario)

    scenarios_in = [_fake_scenario(sid) for sid in ["zeta", "alpha", "mu"]]
    results = runner.run_many(scenarios_in, workers=3)

    assert [r["sid"] for r in results] == ["alpha", "mu", "zeta"]


def test_run_many_propagates_root_and_force_to_each_call(monkeypatch):
    calls = []

    def fake_run_scenario(scenario, root, force):
        calls.append((scenario.sid, root, force))
        return {"sid": scenario.sid, "split": scenario.split, "seed": scenario.seed,
                "status": "generated", "seconds": 0.0, "error": None}

    monkeypatch.setattr(runner, "run_scenario", fake_run_scenario)

    scenarios_in = [_fake_scenario(sid) for sid in ["a", "b"]]
    runner.run_many(scenarios_in, root="SOME/ROOT", workers=2, force=True)

    assert len(calls) == 2
    for sid, root, force in calls:
        assert root == "SOME/ROOT"
        assert force is True
    assert {sid for sid, _, _ in calls} == {"a", "b"}
