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


def test_a_grouped_pulse_is_not_called_tracking_loss_for_its_ACTIVE_days():
    """A pulse train spans first start to last end, so its [start, end]
    interval covers the ON days between the pulses too.

    Before this was pinned, the trigger asked two separate questions of that
    span -- "was any day off?" and "does the span carry impressions?" -- and
    both are yes for every pulse train that ever ran. It then printed "spend
    is zero but impressions continue in the window" about a channel whose
    off-days carried EXACTLY ZERO impressions and whose every impression fell
    on a day it was spending normally. Measured on the development split that
    was 8 of 15 tracking-loss verdicts, every one of them false, every one of
    them on a correct detection (dev_019 DE/Affiliate: 119-day window, 56 off
    days with 0 impressions, 63 on days carrying all 8.6M of them).

    The claim needs the INTERSECTION -- impressions on the days spend was off
    -- evaluated over `components`, the real windows, not over the span.
    """
    values = [100.0] * 20 + [0.0] * 10 + [100.0] * 20 + [0.0] * 10 + [100.0] * 20
    p = build({"TV": values, "Radio": [50.0] * len(values)})
    e = event(p, "TV", 20, 59, event_type="channel_pulse",
              components=((p.dates[20], p.dates[29]),
                          (p.dates[50], p.dates[59])))
    verdict, reasons = assess(e, p)
    assert verdict == "ok", reasons
    assert reasons == ()


def test_a_grouped_pulse_IS_flagged_when_its_own_off_days_carry_impressions():
    """The other side of the same fix: narrowing the trigger to the pulse's
    real windows must not switch it off. Genuine tracking loss on a pulsing
    channel -- impressions still arriving on the days spend stopped -- is
    exactly what this trigger is for, and it still fires."""
    values = [100.0] * 20 + [0.0] * 10 + [100.0] * 20 + [0.0] * 10 + [100.0] * 20
    p = build({"TV": values, "Radio": [50.0] * len(values)},
              impressions_by_channel={"TV": [100.0] * len(values)})
    e = event(p, "TV", 20, 59, event_type="channel_pulse",
              components=((p.dates[20], p.dates[29]),
                          (p.dates[50], p.dates[59])))
    verdict, reasons = assess(e, p)
    assert verdict == "suspect_tracking_loss", reasons
    assert any("impressions" in r for r in reasons)


def test_a_near_zero_event_is_never_told_its_spend_was_zero():
    """`off_mask` calls a day off at a FRACTION of the active level, so a
    near-zero event -- spend cut to a trickle and held there -- is "off"
    while still buying. The old sentence said "spend is zero" about it.

    On the development split that was dev_012 SE/Radio spending 36,466 across
    its 90 "off" days, dev_034 DE/TV 425 and dev_032 FR/OOH 135, with
    impressions simply proportional to the residual spend. The wording follows
    `OffRun.kind` here for the same reason detection/explain.py's does: a
    sentence that contradicts the numbers printed beside it is worse than no
    sentence.
    """
    p = build({"TV": [100.0] * 60 + [5.0] * 30 + [100.0] * 60,
               "Radio": [50.0] * 150},
              impressions_by_channel={"TV": [100.0] * 150})
    verdict, reasons = assess(event(p, "TV", 60, 89), p)
    assert verdict == "suspect_tracking_loss", reasons
    joined = " ".join(reasons)
    assert "spend is zero" not in joined, joined
    assert "trickle" in joined, joined


def test_an_exact_zero_event_still_says_spend_is_zero():
    """The near-zero wording is chosen by OffRun.kind, so an exact zero must
    still get the stop wording -- otherwise the kind check has simply replaced
    one unconditional sentence with another."""
    p = build({"TV": [100.0] * 60 + [0.0] * 30 + [100.0] * 60,
               "Radio": [50.0] * 150},
              impressions_by_channel={"TV": [100.0] * 150})
    _, reasons = assess(event(p, "TV", 60, 89), p)
    joined = " ".join(reasons)
    assert "spend is zero" in joined, joined
    assert "trickle" not in joined, joined


def test_a_step_change_with_missing_rows_is_still_not_assessed():
    """_STOPPING_TYPES, pinned on a fixture that can actually see it.

    `test_a_step_change_is_not_assessed_for_missing_rows` above states this
    intent but cannot prove it: its fixture has no missing rows at all, so the
    `ok` verdict survives deleting the early return. This one gives the step
    window 21 genuinely absent rows, which the missing-rows trigger would
    report as a data gap were step_change not excluded -- so deleting the
    guard turns this red.
    """
    n = 150
    dates = pd.date_range("2024-01-01", periods=n)
    rows = []
    for i, d in enumerate(dates):
        if 80 <= i < 101:
            continue                      # rows ABSENT inside the new level
        spend = 100.0 if i < 75 else 300.0
        rows.append([d, "P", "TV", "c", 1, spend, 1.0, spend, 0, 0.0, "DE"])
        rows.append([d, "P", "Radio", "c", 1, 50.0, 1.0, 50.0, 0, 0.0, "DE"])
    from detection.io.panel import build_panel
    from tests.detection.test_score import COLS
    p = build_panel(pd.DataFrame(rows, columns=COLS), sid="dev_test")

    verdict, reasons = assess(event(p, "TV", 75, 149, event_type="step_change",
                                    evidence={"z": 6.0}), p)
    assert verdict == "ok", reasons
    assert reasons == ()

    # ... and the fixture really does contain the absence the guard is
    # suppressing, so the assertion above is not passing by coincidence.
    holdout = event(p, "TV", 80, 100, event_type="natural_holdout")
    assert assess(holdout, p)[0] == "suspect_data_gap"


def test_a_pulse_is_only_assessed_on_the_windows_it_actually_reports():
    """The components fix, isolated from the intersection fix.

    A pulse train reports the windows in `components`, not every off day
    inside its span: a short gap between two pulses that never cleared
    MIN_DAYS is not part of the event's claim. Reading the span instead lets
    the trigger indict the train for a day it never reported -- here the two
    reported windows are clean (spend zero, impressions zero) and only the
    unreported three-day gap carries impressions, so a span-scoped trigger
    fires and its sentence, "on the days this event reports as off", is
    false.
    """
    values = ([100.0] * 20 + [0.0] * 10 + [100.0] * 20 + [0.0] * 3
              + [100.0] * 17 + [0.0] * 10 + [100.0] * 20)
    imps = [0.0 if v == 0 else 100.0 for v in values]
    for i in range(50, 53):               # the unreported short gap only
        imps[i] = 500.0
    p = build({"TV": values, "Radio": [50.0] * len(values)},
              impressions_by_channel={"TV": imps})
    e = event(p, "TV", 20, 79, event_type="channel_pulse",
              components=((p.dates[20], p.dates[29]),
                          (p.dates[70], p.dates[79])))
    verdict, reasons = assess(e, p)
    assert verdict == "ok", reasons
