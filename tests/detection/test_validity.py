import pandas as pd
import pytest

from detection import params
from detection.validity import assess
from tests.detection.test_score import build, event


def test_a_clean_event_is_ok_with_no_reasons():
    p = build({"TV": [100.0] * 60 + [0.0] * 30 + [100.0] * 60,
               "Radio": [50.0] * 150})
    verdict, reasons = assess(event(p, "TV", 60, 89), p)
    assert verdict == "ok"
    assert reasons == ()


def test_missing_rows_are_a_data_gap_not_a_holdout():
    """The single largest real-data risk in spec section 11: an export that
    omits a row rather than writing a zero. The shape is identical; the meaning
    is the opposite."""
    n = 150
    dates = pd.date_range("2024-01-01", periods=n)
    rows = []
    for i, d in enumerate(dates):
        if 60 <= i < 90:
            continue                      # rows ABSENT, not zero
        rows.append([d, "P", "TV", "c", 1, 100.0, 1.0, 100.0, 0, 0.0, "DE"])
        rows.append([d, "P", "Radio", "c", 1, 50.0, 1.0, 50.0, 0, 0.0, "DE"])
    from detection.io.panel import build_panel
    from tests.detection.test_score import COLS
    p = build_panel(pd.DataFrame(rows, columns=COLS), sid="dev_test")
    verdict, reasons = assess(event(p, "TV", 60, 89), p)
    assert verdict == "suspect_data_gap"
    assert any("missing" in r for r in reasons)


def test_spend_zero_with_impressions_flowing_is_tracking_loss():
    p = build({"TV": [100.0] * 60 + [0.0] * 30 + [100.0] * 60,
               "Radio": [50.0] * 150},
              impressions_by_channel={"TV": [100.0] * 150})
    verdict, reasons = assess(event(p, "TV", 60, 89), p)
    assert verdict == "suspect_tracking_loss"
    assert any("impression" in r for r in reasons)


def test_every_channel_in_every_country_off_at_once_is_suspect():
    """A panel-wide stop is far more likely to be a feed outage than a
    coordinated global marketing pause."""
    p = build({"TV": [100.0] * 60 + [0.0] * 30 + [100.0] * 60,
               "Radio": [100.0] * 60 + [0.0] * 30 + [100.0] * 60})
    verdict, reasons = assess(
        event(p, None, 60, 89, event_type="dark_period"), p)
    assert verdict != "ok"
    assert any("every channel" in r for r in reasons)


def test_a_verdict_lists_every_reason_that_fired():
    """Reasons accumulate: an analyst triaging needs all of them, not the
    first one found."""
    p = build({"TV": [100.0] * 60 + [0.0] * 30 + [100.0] * 60,
               "Radio": [100.0] * 60 + [0.0] * 30 + [100.0] * 60},
              impressions_by_channel={"TV": [100.0] * 150,
                                      "Radio": [100.0] * 150})
    _, reasons = assess(event(p, None, 60, 89, event_type="dark_period"), p)
    assert len(reasons) >= 2


def test_a_step_change_is_not_assessed_for_missing_rows():
    """A step change never stops the channel, so there is no absence to
    explain and the gate must not invent one."""
    p = build({"TV": [100.0] * 75 + [300.0] * 75, "Radio": [50.0] * 150})
    verdict, _ = assess(event(p, "TV", 75, 149, event_type="step_change",
                              evidence={"z": 6.0}), p)
    assert verdict == "ok"


def test_tracking_loss_is_not_claimed_about_a_channel_that_never_stopped():
    """The trigger's claim is "spend stopped but impressions did not", so it may
    only fire where spend actually stopped.

    A dark_period's subject channels are EVERY channel in the market. Before
    this was pinned, a market where only TV went dark while Radio spent 50/day
    throughout reported: "Radio: spend is zero but impressions continue" -- a
    false statement about a channel spending normally, in output an analyst is
    meant to trust. Wrong verdicts are worse than no verdict here, because the
    whole point of the gate is to be believed.
    """
    p = build({"TV": [100.0] * 60 + [0.0] * 30 + [100.0] * 60,
               "Radio": [50.0] * 150})
    verdict, reasons = assess(
        event(p, None, 60, 89, event_type="dark_period"), p)
    assert not any("Radio" in r for r in reasons), reasons
    assert verdict == "ok", verdict
