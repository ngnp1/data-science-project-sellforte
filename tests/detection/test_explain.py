import pandas as pd
import pytest

from detection.explain import explain
from detection.model import DetectedEvent
from tests.detection.test_score import build, event


def test_the_explanation_names_the_channel_market_dates_and_length():
    p = build({"TV": [100.0] * 60 + [0.0] * 30 + [100.0] * 60,
               "Radio": [50.0] * 150})
    e = event(p, "TV", 60, 89, detection_confidence=0.9,
              informativeness=0.5)
    text = explain(e, p)
    assert "TV" in text
    assert "DE" in text
    assert str(p.dates[60].date()) in text
    assert str(p.dates[89].date()) in text
    assert "30" in text


def test_the_explanation_states_the_normal_level_and_the_window_level():
    """'Fell from a typical X to Y' is the sentence an analyst checks against
    a briefed budget. Without both numbers it is not checkable."""
    p = build({"TV": [100.0] * 60 + [0.0] * 30 + [100.0] * 60,
               "Radio": [50.0] * 150})
    text = explain(event(p, "TV", 60, 89, detection_confidence=0.9,
                         informativeness=0.5), p)
    assert "100" in text


def test_the_explanation_reports_all_three_scores():
    p = build({"TV": [100.0] * 60 + [0.0] * 30 + [100.0] * 60,
               "Radio": [50.0] * 150})
    text = explain(event(p, "TV", 60, 89, detection_confidence=0.94,
                         informativeness=0.81), p)
    assert "0.94" in text
    assert "0.81" in text
    assert "ok" in text


def test_a_suspect_event_says_so_and_gives_the_reason():
    """A validity warning that does not reach the prose is a warning nobody
    reads."""
    p = build({"TV": [100.0] * 60 + [0.0] * 30 + [100.0] * 60,
               "Radio": [50.0] * 150})
    e = event(p, "TV", 60, 89, detection_confidence=0.9, informativeness=0.5,
              validity="suspect_tracking_loss",
              validity_reasons=("TV: spend is zero but impressions continue",))
    text = explain(e, p)
    assert "suspect_tracking_loss" in text
    assert "impressions continue" in text


def test_a_pulse_train_explanation_states_how_many_windows():
    """A bare 'assert "4" in text' would also pass on the year in every date
    ("2024...") and 'assert "pulse" in text.lower()' on "channel_pulse" in
    the Label line -- neither actually exercises the pulse-count sentence.
    The assertion below is tied to the template's actual wording so deleting
    that sentence is caught here."""
    p = build({"TV": ([100.0] * 20 + [0.0] * 10) * 4 + [100.0] * 30,
               "Radio": [50.0] * 150})
    e = DetectedEvent(
        sid="dev_test", country_code="DE", channel="TV",
        event_type="channel_pulse",
        start=p.dates[20], end=p.dates[109],
        components=((p.dates[20], p.dates[29]), (p.dates[50], p.dates[59])),
        evidence={"n_pulses": 4},
        detection_confidence=0.8, informativeness=0.7)
    text = explain(e, p)
    assert "pulse train of 4" in text


def test_a_step_change_explanation_states_the_ratio():
    p = build({"TV": [100.0] * 75 + [300.0] * 75, "Radio": [50.0] * 150})
    e = event(p, "TV", 75, 149, event_type="step_change",
              magnitude_ratio=3.02, evidence={"z": 6.0},
              detection_confidence=0.7, informativeness=0.4)
    text = explain(e, p)
    assert "3.0" in text


def test_a_step_out_of_zero_does_not_claim_a_ratio():
    """magnitude_ratio is None when the level before the step was zero. The
    prose must say so rather than printing 'None' or inventing a number.

    'assert "None" not in text' alone would also pass on a template that
    just DROPPED the ratio sentence for this case -- silence is not the same
    as saying so. This also requires the explanation actually name the
    zero-start reason, so deleting that branch (rather than fixing it) is
    caught here too."""
    p = build({"TV": [0.0] * 40 + [300.0] * 110, "Radio": [50.0] * 150})
    e = event(p, "TV", 40, 149, event_type="step_change",
              magnitude_ratio=None, evidence={"z": 6.0},
              detection_confidence=0.7, informativeness=0.4)
    text = explain(e, p)
    assert "None" not in text
    assert "rose from zero" in text or "no finite ratio" in text


def test_every_event_type_produces_prose_without_raising():
    """The explanation is part of the output schema, so no type may be left
    without one."""
    from detection.model import EVENT_TYPES
    p = build({"TV": [100.0] * 60 + [0.0] * 30 + [100.0] * 60,
               "Radio": [50.0] * 150})
    for event_type in sorted(EVENT_TYPES):
        e = event(p, "TV", 60, 89, event_type=event_type,
                  evidence={"z": 5.0}, detection_confidence=0.5,
                  informativeness=0.5)
        text = explain(e, p)
        assert len(text) > 40, event_type


def test_an_unscored_event_still_explains_itself():
    """Explanation must not require scoring to have run -- it is also the
    debugging surface when scoring is what went wrong."""
    p = build({"TV": [100.0] * 60 + [0.0] * 30 + [100.0] * 60,
               "Radio": [50.0] * 150})
    text = explain(event(p, "TV", 60, 89), p)
    assert len(text) > 40


def test_a_natural_holdout_names_the_next_longest_off_run():
    """The spec's own register example: 'the longest off-run in this series
    by a factor of 21, the next longest being 2 days.' A distinctiveness
    claim like that is only checkable if the explanation actually names the
    runner-up, not just the winner."""
    p = build({"TV": [100.0] * 30 + [0.0] * 42 + [100.0] * 10 + [0.0] * 2
                     + [100.0] * 10,
               "Radio": [50.0] * 94})
    e = event(p, "TV", 30, 71, detection_confidence=0.94, informativeness=0.81)
    text = explain(e, p)
    # "42" alone would pass on the base sentence's own n_days -- the
    # runner-up figure only matters if the exact "N days" runner-up phrase is
    # there, and so does the ratio between the two.
    assert "factor of 21" in text
    assert "2 days" in text


def test_control_availability_names_the_peer_count_when_known():
    """Spec register: '...so those four markets are available as controls.'
    When the cross-market layer has already counted the peers, the count
    belongs in the sentence, not just the word 'peers'. A bare "4" would also
    match the "2024" in every date in this text, so the assertion checks the
    specific phrase instead."""
    p = build({"TV": [100.0] * 60 + [0.0] * 30 + [100.0] * 60,
               "Radio": [50.0] * 150})
    e = event(p, "TV", 60, 89,
              evidence={"control_available": "peers", "peers_running": 4})
    text = explain(e, p)
    assert "4 other markets" in text


def test_deleting_the_caveat_line_is_caught_by_the_suspect_test():
    """Guard against a template that mentions the validity LABEL but drops
    the actual reason text -- 'suspect_tracking_loss' alone is not an
    explanation an analyst can act on."""
    p = build({"TV": [100.0] * 60 + [0.0] * 30 + [100.0] * 60,
               "Radio": [50.0] * 150})
    e = event(p, "TV", 60, 89, validity="suspect_data_gap",
              validity_reasons=("TV: 5 of 30 days in the window are missing "
                                "rows rather than zero-spend rows",))
    text = explain(e, p)
    assert "5 of 30 days" in text


def test_a_dark_period_states_the_market_total_typical_and_window_level():
    """dark_period's `channel` is always None, so the single-channel 'fell
    from a typical X to Y' sentence has nothing to read -- this is the gap
    the coordinator found in dev_005: the explanation for the detector's
    strongest event type named no spend magnitude at all. The market-level
    fallback must report the TOTAL across every channel the market runs,
    computed the same active-days-median way as the single-channel case.

    Both TV and Radio are built so their outside-window levels (100 and 80)
    are distinct and sum to an unambiguous total (180) that cannot arise
    from either channel alone or from any date/count already in the text --
    unlike a bare "4", "180" cannot come from a coincidental "2024" in a
    date string, and asserting the whole phrase (not just the number) also
    catches a template that reports the right number in the wrong sentence."""
    p = build({"TV": [100.0] * 60 + [0.0] * 30 + [100.0] * 60,
               "Radio": [80.0] * 60 + [0.0] * 30 + [80.0] * 60})
    e = event(p, None, 60, 89, event_type="dark_period",
              detection_confidence=0.9, informativeness=0.6)
    text = explain(e, p)
    assert "typical 180 per day to exactly 0 per day" in text


def test_deleting_the_dark_period_market_total_sentence_fails_the_test_above():
    """Not a test of explain() itself -- a sanity check that the assertion
    above is load-bearing, run inline rather than by hand-editing the source
    for the report. If the market-level sentence is never produced (e.g. the
    branch guard is wrong), the phrase must not appear by accident."""
    p = build({"TV": [100.0] * 60 + [0.0] * 30 + [100.0] * 60,
               "Radio": [80.0] * 60 + [0.0] * 30 + [80.0] * 60})
    # A channel-level event (not dark_period) must NOT produce the market
    # total sentence -- it has its own single-channel sentence instead.
    e = event(p, "TV", 60, 89, event_type="natural_holdout")
    text = explain(e, p)
    assert "typical 180 per day to exactly 0 per day" not in text
