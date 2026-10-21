"""The teammates' algorithms combined behind the Detector interface."""
import pandas as pd
import pytest

from detection.model import DetectedEvent, EVENT_TYPES
from detection.pipeline import run_detection
from detectors import registry
from tests.conftest import GENERATOR_DIR

SAMPLE = GENERATOR_DIR / "data"
OWNERS = {"single_channel": "Nam Phu", "natural_holdout": "Thai", "staggered_launch": "Thai",
          "channel_pulse": "Triet", "dark_period": "Triet", "step_change": "David"}


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
        assert f"Found by {OWNERS[f.event_type]}" in f.explanation


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
