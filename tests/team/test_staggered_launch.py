import pandas as pd

from detection.model import DetectedEvent
from tests.conftest import GENERATOR_DIR
from team.staggered_launch import run_detection


def test_finds_the_late_us_instagram_launch_on_the_sample():
    data = GENERATOR_DIR / "data"
    found = run_detection(pd.read_csv(data / "media.csv"), pd.read_csv(data / "sales.csv"))
    assert [(e.event_type, e.country_code, e.channel) for e in found] == \
        [("staggered_launch", "US", "Instagram")]
    assert all(isinstance(e, DetectedEvent) and "launch_verification" in e.evidence for e in found)


def test_empty_media_finds_nothing():
    assert run_detection(pd.DataFrame(columns=["date", "country_code",
                                               "advertising_channel", "media_investment"])) == []
