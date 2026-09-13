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
