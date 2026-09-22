"""The harness-side adapter: DetectedEvent -> Event."""
import pathlib

import pandas as pd
import pytest

from benchmark.eval.adapter import detect, to_event
from benchmark.eval.model import Event
from detection.model import DetectedEvent

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEV = ROOT / "benchmark" / "datasets" / "dev"


def load(sid):
    media = pd.read_csv(DEV / sid / "media.csv", parse_dates=["date"])
    sales = pd.read_csv(DEV / sid / "sales.csv", parse_dates=["date"])
    return media, sales


def test_to_event_maps_every_field():
    d = DetectedEvent(sid="dev_001", country_code="DE", channel="TV",
                      event_type="natural_holdout",
                      start=pd.Timestamp("2024-03-01"),
                      end=pd.Timestamp("2024-03-10"),
                      magnitude_ratio=3.0,
                      components=((pd.Timestamp("2024-03-01"),
                                   pd.Timestamp("2024-03-05")),),
                      tags=("cross_market_holdout",),
                      evidence={"z": 4.2})
    e = to_event(d)
    assert isinstance(e, Event)
    assert (e.sid, e.country_code, e.channel, e.event_type) == \
           ("dev_001", "DE", "TV", "natural_holdout")
    assert e.start == d.start and e.end == d.end
    assert e.components == d.components
    assert e.tags == d.tags
    # magnitude_ratio is the harness's `multiplier`; the brief's version of
    # this test claimed to map every field but never asserted it.
    assert e.multiplier == d.magnitude_ratio
    # Spec section 8's scores are Plan 4's work; until then they must be None
    # rather than a default the reliability curve would silently believe.
    assert e.detection_confidence is None
    assert e.informativeness is None


def test_to_event_carries_a_panel_wide_event_with_no_channel():
    """A dark period names no channel, and truth records it as channel None.
    A mapper that coerced None to a string would fail every dark match."""
    d = DetectedEvent(sid="dev_005", country_code="FR", channel=None,
                      event_type="dark_period",
                      start=pd.Timestamp("2024-12-18"),
                      end=pd.Timestamp("2024-12-31"))
    e = to_event(d)
    assert e.channel is None
    assert e.multiplier is None
    assert e.n_days == 14


@pytest.mark.benchmark_data
def test_detect_returns_harness_events_for_a_real_scenario():
    media, sales = load("dev_005")
    events = detect(media, sales, "dev_005")
    assert events and all(isinstance(e, Event) for e in events)


@pytest.mark.benchmark_data
def test_detect_preserves_pulse_grouping():
    """One Event per pulse train, with its windows attached. Re-splitting a
    train here would push every pulse IoU below the matcher's threshold."""
    media, sales = load("dev_019")
    pulses = [e for e in detect(media, sales, "dev_019")
              if e.event_type == "channel_pulse"
              and e.country_code == "DE" and e.channel == "Affiliate"]
    assert len(pulses) == 1
    assert len(pulses[0].components) >= 2


def test_detect_requires_the_media_frame():
    """run_dev and run_final pass None unless --load-data is given. A detector
    that silently returned [] there would burn the one-shot final gate and
    record a 0.0 audit line."""
    with pytest.raises(ValueError, match="--load-data"):
        detect(None, None, "dev_005")


def test_detect_is_the_callable_run_dev_loads():
    """The adapter is only useful if run_dev's --detector spec resolves to it.
    Checks the loader, not the string."""
    from benchmark.eval.run_dev import load_detector
    assert load_detector("benchmark.eval.adapter:detect") is detect


def test_the_adapter_carries_both_scores_across():
    """The harness's reliability and operating curves read
    Event.detection_confidence. Dropping it here renders both degenerate --
    which is exactly what happened before scoring existed."""
    from detection.model import DetectedEvent
    d = DetectedEvent(
        sid="dev_001", country_code="DE", channel="TV",
        event_type="natural_holdout",
        start=pd.Timestamp("2024-03-01"), end=pd.Timestamp("2024-04-01"),
        detection_confidence=0.83, informativeness=0.41)
    e = to_event(d)
    assert e.detection_confidence == 0.83
    assert e.informativeness == 0.41
