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
