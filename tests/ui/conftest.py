import pandas as pd
import pytest

from detection.io.panel import build_panel
from detection.pipeline import run_detection
from tests.conftest import GENERATOR_DIR

SAMPLE = GENERATOR_DIR / "data"


@pytest.fixture(scope="session")
def sample_frames():
    return pd.read_csv(SAMPLE / "media.csv"), pd.read_csv(SAMPLE / "sales.csv")


@pytest.fixture(scope="session")
def sample_events(sample_frames):
    return run_detection(*sample_frames, sid="sample")


@pytest.fixture(scope="session")
def sample_panel(sample_frames):
    return build_panel(*sample_frames, sid="sample")
