"""Pins pulse/stress.py per on/off layer: cases handled, and known failures as xfail."""
import pytest

from pulse.stress import CASES, NEGATIVES, score

KNOWN_FAILURES = {
    "A": {
        "1-day spend spike inside each pause": "the spike splits each pause; 7-day boundary error",
        "2-day spend spike inside each pause": "the spike splits each pause; 7-day boundary error",
        "3-day spend spike inside each pause": "the spike splits each pause; 7-day boundary error",
        "2 missing days inside a pause's flank": "missing days extend the pause by 2 days",
        "short 5-day pauses (below MIN_DAYS)": "below MIN_DAYS = 7 by design",
    },
    "B": {
        "2-day spend spike inside each pause": "two switches beat two wrong verdicts; 7-day boundary error",
        "3-day spend spike inside each pause": "two switches beat three wrong verdicts; 7-day boundary error",
        "short 5-day pauses (below MIN_DAYS)": "below MIN_DAYS = 7 by design",
    },
    "C": {
        "short 5-day pauses (below MIN_DAYS)": "below MIN_DAYS = 7 by design",
    },
}


def _cases(layer):
    known = KNOWN_FAILURES[layer]
    return [pytest.param(layer, *c, id=f"{layer}-{c[0]}",
                         marks=[pytest.mark.xfail(reason=known[c[0]], strict=True)] if c[0] in known else [])
            for c in CASES]


@pytest.mark.parametrize("layer,name,pauses,kw", _cases("A") + _cases("B") + _cases("C"))
def test_every_pause_found_exactly(layer, name, pauses, kw):
    n, matched, err, _ = score(pauses, layer, **kw)
    assert n == 1 and matched == len(pauses) and err == 0


@pytest.mark.parametrize("layer", "ABC")
@pytest.mark.parametrize("name,pauses,kw", NEGATIVES, ids=[c[0] for c in NEGATIVES])
def test_no_pulse_reported(layer, name, pauses, kw):
    assert score(pauses, layer, **kw)[0] == 0
