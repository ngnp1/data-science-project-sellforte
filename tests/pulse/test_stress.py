"""Pins pulse/stress.py: cases the detector handles, and known failures as xfail."""
import pytest

from pulse.stress import CASES, NEGATIVES, score

KNOWN_FAILURES = {
    "partial: spend at 25% in pauses": "above the 15% cutoff, so treated as on",
    "1-day spend spike inside each pause": "the spike splits each pause; 7-day boundary error",
    "2 missing days inside a pause's flank": "missing days extend the pause by 2 days",
    "short 5-day pauses (below MIN_DAYS)": "below MIN_DAYS = 7 by design",
}


@pytest.mark.parametrize("name,pauses,kw", [
    pytest.param(*c, marks=pytest.mark.xfail(reason=KNOWN_FAILURES[c[0]], strict=True))
    if c[0] in KNOWN_FAILURES else c for c in CASES], ids=[c[0] for c in CASES])
def test_every_pause_found_exactly(name, pauses, kw):
    n, matched, err, _ = score(pauses, **kw)
    assert n == 1 and matched == len(pauses) and err == 0


@pytest.mark.parametrize("name,pauses,kw", NEGATIVES, ids=[c[0] for c in NEGATIVES])
def test_no_pulse_reported(name, pauses, kw):
    assert score(pauses, **kw)[0] == 0
