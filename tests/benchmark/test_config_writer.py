import dataclasses
import subprocess

import pytest
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


def _max_min_proportion(channels):
    """Replay generate_with_simmmulator.R's exact regrouping logic (lines
    53-58 and 76-79 of generate_with_simmmulator.R) in Python: split channels
    into impression-type and click-type, concatenate impressions first, drop
    the last element of THAT order, then flatten (spend_share_min,
    spend_share_max) pairs from what remains -- exactly what R's
    MAX_MIN_PROPORTION vector is built from."""
    impression_channels = [c for c in channels if c["type"] == "impression"]
    click_channels = [c for c in channels if c["type"] == "click"]
    channels_ordered = impression_channels + click_channels
    all_but_last = channels_ordered[:-1]
    values = []
    for ch in all_but_last:
        values.append(ch.get("spend_share_min"))
        values.append(ch.get("spend_share_max"))
    return values


def test_max_min_proportion_vector_matches_r_for_every_scenario():
    """This is the test that would have caught the channel-ordering defect:
    for every one of the 100 frozen scenarios, replay R's exact regrouping
    logic and assert the resulting MAX_MIN_PROPORTION vector has length
    2*n_channels - 2 with no missing values. Before the fix to
    axes.pick_channels, 49 of 100 scenarios produced a short vector here
    because R's "last channel after impressions-then-clicks regrouping"
    didn't match Python's "last channel drawn"."""
    all_scenarios = scenarios.build_all()
    assert len(all_scenarios) == 100
    for s in all_scenarios:
        n = len(s.channels)
        values = _max_min_proportion(s.channels)
        assert len(values) == 2 * n - 2, \
            f"{s.sid}: MAX_MIN_PROPORTION length {len(values)} != {2 * n - 2}"
        assert all(v is not None for v in values), \
            f"{s.sid}: MAX_MIN_PROPORTION has a missing value"


@pytest.mark.slow
def test_generator_accepts_the_fixed_channel_ordering(tmp_path, generator_dir):
    """dev_006 was one of the 49 scenarios broken by the channel-ordering
    defect in axes.pick_channels: R's regrouped "last channel" (last after
    sorting impressions-then-clicks) didn't match Python's "last drawn"
    channel, so R's MAX_MIN_PROPORTION vector came up short and
    step_2_ads_spend aborted mid-run. Unlike
    test_r_can_actually_read_the_generated_config above -- which only proves
    R can parse the YAML dialect, and passed on this exact scenario even
    while it was fatally broken -- this test runs the real simulator
    end-to-end and proves siMMMulator actually accepts the channel contract
    we emit.

    Reduced to one country and one year (keeping the scenario's own channel
    list, which is what is under test) to keep the run to roughly 25
    seconds."""
    s = next(x for x in scenarios.build_all() if x.sid == "dev_006")
    minimal = dataclasses.replace(s, countries=(s.countries[0],), years=1)

    cfg_path, ev_path = write_scenario_configs(minimal, tmp_path)
    outdir = tmp_path / "out"

    proc = subprocess.run(
        ["Rscript", str(generator_dir / "generate_with_simmmulator.R"),
         "--seed", str(minimal.seed), "--config", str(cfg_path),
         "--events", str(ev_path), "--outdir", str(outdir)],
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    assert (outdir / "raw_daily_wide.csv").is_file()
