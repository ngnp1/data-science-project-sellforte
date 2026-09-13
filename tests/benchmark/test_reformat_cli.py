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
