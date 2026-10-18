import pytest

from detection.model import EVENT_TYPES
from detectors import registry
from detectors.base import Detector
from detectors.rule_based import RuleBasedDetector


class Stub(Detector):
    key = "stub-registry"
    name = "Stub"
    event_types = frozenset({"dark_period"})

    def detect(self, media, sales):
        return []


@pytest.fixture
def clean_registry(monkeypatch):
    monkeypatch.setattr(registry, "_DETECTORS", dict(registry._DETECTORS))


def test_our_library_is_the_default_detector():
    assert isinstance(registry.default(), RuleBasedDetector)
    assert registry.default().key == "rule_based"
    assert registry.default().event_types == EVENT_TYPES


def test_register_get_and_available_keep_order(clean_registry):
    stub = Stub()
    registry.register(stub)
    assert registry.get("stub-registry") is stub
    assert [d.key for d in registry.available()] == ["rule_based", "stub-registry"]
    assert registry.default().key == "rule_based"


def test_a_key_can_only_be_registered_once(clean_registry):
    registry.register(Stub())
    with pytest.raises(ValueError, match="stub-registry"):
        registry.register(Stub())


def test_an_unknown_key_names_the_available_ones():
    with pytest.raises(KeyError, match="rule_based"):
        registry.get("nope")


def test_a_detector_must_implement_detect():
    class Incomplete(Detector):
        key, name, event_types = "x", "X", frozenset()
    with pytest.raises(TypeError):
        Incomplete()
