"""Every registered detector honours the output contract on the sample."""
import pandas as pd
import pytest

from detection.model import EVENT_TYPES, DetectedEvent
from detectors import registry
from tests.conftest import GENERATOR_DIR

SAMPLE = GENERATOR_DIR / "data"


@pytest.fixture(scope="module")
def frames():
    return pd.read_csv(SAMPLE / "media.csv"), pd.read_csv(SAMPLE / "sales.csv")


@pytest.mark.parametrize("detector", registry.available(), ids=lambda d: d.key)
def test_findings_honour_the_contract(detector, frames):
    media, sales = frames
    findings = detector.detect(media, sales)
    assert findings and all(isinstance(f, DetectedEvent) for f in findings)
    dates = pd.to_datetime(media["date"])
    markets = set(media["country_code"])
    channels = set(media["advertising_channel"])
    for f in findings:
        assert f.event_type in EVENT_TYPES and f.event_type in detector.event_types
        assert f.country_code in markets
        assert dates.min() <= f.start <= f.end <= dates.max()
        assert (f.channel is None) == (f.event_type == "dark_period")
        assert f.channel is None or f.channel in channels
        if f.event_type == "channel_pulse":
            assert f.components


def test_the_rule_based_detector_is_our_library(frames):
    from detection.pipeline import run_detection
    media, sales = frames
    assert registry.get("rule_based").detect(media, sales) == run_detection(media, sales, "")


def test_subject_channels_lives_with_the_contract():
    import detection.model as model
    import detection.scoring.score as score
    assert model.subject_channels is score.subject_channels
