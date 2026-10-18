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
    """Only our detector registered, whatever else a later change adds, and
    restored afterwards."""
    monkeypatch.setattr(registry, "_DETECTORS", {"rule_based": registry.get("rule_based")})


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



def test_every_module_in_detectors_is_listed_for_registration():
    """A detector module nobody imports never registers -- the README's steps
    would silently do nothing. Every module in the package must be listed."""
    import pathlib
    import detectors
    pkg = pathlib.Path(detectors.__file__).parent
    modules = {f"detectors.{f.stem}" for f in pkg.glob("*.py")
               if f.stem not in {"__init__", "base", "registry"}}
    assert modules == set(registry.MODULES)


def test_every_listed_module_registers_its_detectors():
    import importlib
    for name in registry.MODULES:
        found = importlib.import_module(name).DETECTORS
        assert found and all(isinstance(d, Detector) for d in found)
        assert all(registry.get(d.key) is d for d in found)
