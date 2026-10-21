"""The teammates' algorithms combined behind the Detector interface."""
from dataclasses import replace

import pandas as pd
import pytest

from detection.model import DetectedEvent, EVENT_TYPES
from detection.pipeline import run_detection
from detectors import registry
from tests.conftest import GENERATOR_DIR

SAMPLE = GENERATOR_DIR / "data"
OWNERS = {"single_channel": "Nam Phu", "natural_holdout": "Triet", "staggered_launch": "tvh-ds",
          "channel_pulse": "Triet", "dark_period": "Triet", "step_change": "Triet"}


@pytest.fixture(scope="module")
def frames():
    return pd.read_csv(SAMPLE / "media.csv"), pd.read_csv(SAMPLE / "sales.csv")


@pytest.fixture(scope="module")
def team_findings(frames):
    return registry.get("team").detect(*frames)


def test_the_team_detector_is_registered_after_ours():
    assert [d.key for d in registry.available()][:2] == ["rule_based", "team"]
    assert registry.get("team").event_types == EVENT_TYPES


def test_each_pattern_comes_from_its_owner(team_findings):
    assert team_findings and all(isinstance(f, DetectedEvent) for f in team_findings)
    for f in team_findings:
        assert f.evidence["source"] == OWNERS[f.event_type]
        assert OWNERS[f.event_type] not in f.explanation


def test_on_the_sample_it_finds_the_same_periods_as_our_library(frames, team_findings):
    ours = {(e.event_type, e.country_code, e.channel) for e in run_detection(*frames)}
    assert {(f.event_type, f.country_code, f.channel) for f in team_findings} == ours


def test_pulse_windows_and_step_size_survive_the_conversion(team_findings):
    pulse = next(f for f in team_findings if f.event_type == "channel_pulse")
    assert len(pulse.components) == 3
    step = next(f for f in team_findings if f.event_type == "step_change")
    assert step.country_code == "CH" and step.channel == "Google Search"


def test_the_harness_can_score_it(frames):
    from benchmark.eval.adapter import for_detector
    events = for_detector("team")(*frames, "sample")
    assert len(events) == 6 and {e.sid for e in events} == {"sample"}


def test_staggered_launches_keep_tvh_ds_readiness_checks(frames, team_findings):
    import app as viewer
    from detection.io.panel import build_panel
    from ui.readout import readout
    launch = next(f for f in team_findings if f.event_type == "staggered_launch")
    assert launch.evidence["source"] == "tvh-ds"
    assert launch.evidence["launch_verification"]["status"]
    panel = build_panel(*frames)
    lines = viewer.caveats(launch, readout(launch, panel, team_findings))
    assert any(line.startswith("Launch analysis readiness:") for line in lines)


def test_a_holdout_on_the_same_window_as_a_single_channel_period_is_dropped():
    from detectors.team import _drop_renamed_holdouts
    day = pd.Timestamp
    single = DetectedEvent(sid="", country_code="NL", channel="TikTok", event_type="single_channel",
                           start=day("2025-07-23"), end=day("2025-09-02"))
    same = DetectedEvent(sid="", country_code="NL", channel="Instagram", event_type="natural_holdout",
                         start=day("2025-07-23"), end=day("2025-09-02"))
    other = replace(same, start=day("2025-07-24"))
    elsewhere = replace(same, country_code="SE")
    kept = _drop_renamed_holdouts([single, same, other, elsewhere])
    assert same not in kept and all(f in kept for f in (single, other, elsewhere))


def test_team_findings_are_scored_like_ours(team_findings):
    for f in team_findings:
        assert f.informativeness is not None and f.detection_confidence is not None
        assert f.evidence["confidence_sub_scores"] and f.evidence["informativeness_drivers"]
        assert f.explanation and "Found by" not in f.explanation
    launch = next(f for f in team_findings if f.event_type == "staggered_launch")
    assert launch.evidence["source"] == "tvh-ds" and launch.evidence["launch_verification"]


def test_scoring_is_shared_with_our_library(frames):
    from detection.io.panel import build_panel
    from detection.scoring import score_findings
    media, sales = frames
    ours = run_detection(media, sales)
    raw = [type(e)(sid=e.sid, country_code=e.country_code, channel=e.channel,
                   event_type=e.event_type, start=e.start, end=e.end,
                   magnitude_ratio=e.magnitude_ratio, components=e.components)
           for e in ours]
    rescored = score_findings(raw, build_panel(media, sales))
    assert [(e.event_type, e.country_code, e.informativeness is not None) for e in rescored] == \
        [(e.event_type, e.country_code, True) for e in ours]
