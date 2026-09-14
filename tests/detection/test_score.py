import numpy as np
import pandas as pd
import pytest

from detection import params
from detection.io.panel import build_panel
from detection.model import DetectedEvent
from detection.score import confidence, informativeness, sub_scores

COLS = ["date", "ad_platform", "advertising_channel", "campaign_name",
        "campaign_id", "media_investment", "clicks", "impressions",
        "conversions", "conversion_value", "country_code"]


def build(spend_by_channel, country="DE", start="2024-01-01",
          impressions_by_channel=None):
    """spend_by_channel: {channel: [daily spend]}. Impressions default to
    tracking spend (zero spend -> zero impressions), which is the corroborated
    case; pass impressions_by_channel to break that link."""
    n = len(next(iter(spend_by_channel.values())))
    dates = pd.date_range(start, periods=n)
    rows = []
    for ch, values in spend_by_channel.items():
        imps = (impressions_by_channel or {}).get(ch)
        for i, (d, v) in enumerate(zip(dates, values)):
            impression = imps[i] if imps is not None else (100.0 if v > 0 else 0.0)
            rows.append([d, "P", ch, "c", 1, float(v), 1.0, float(impression),
                         0, 0.0, country])
    return build_panel(pd.DataFrame(rows, columns=COLS), sid="dev_test")


def event(panel, channel, start, end, event_type="natural_holdout",
          country="DE", **kw):
    return DetectedEvent(
        sid="dev_test", country_code=country, channel=channel,
        event_type=event_type,
        start=panel.dates[start], end=panel.dates[end], **kw)


def test_every_sub_score_is_present_and_bounded():
    """All six are reported alongside the result, so all six must exist for
    every event type -- an absent key would silently drop a term from the
    weighted mean."""
    p = build({"TV": [100.0] * 60 + [0.0] * 30 + [100.0] * 60,
               "Radio": [50.0] * 150})
    s = sub_scores(event(p, "TV", 60, 89), p)
    assert set(s) == {"magnitude_evidence", "duration_evidence",
                      "distinctiveness", "edge_sharpness", "corroboration",
                      "consistency"}
    for name, value in s.items():
        assert 0.0 <= value <= 1.0, f"{name} out of range: {value}"


def test_a_deep_exact_zero_run_scores_full_magnitude_evidence():
    p = build({"TV": [100.0] * 60 + [0.0] * 30 + [100.0] * 60,
               "Radio": [50.0] * 150})
    assert sub_scores(event(p, "TV", 60, 89), p)["magnitude_evidence"] == 1.0


def test_a_shallow_near_zero_run_scores_less_than_an_exact_zero():
    """Depth is the discriminator: spend cut to a trickle is weaker evidence of
    a deliberate pause than spend cut to nothing."""
    deep = build({"TV": [100.0] * 60 + [0.0] * 30 + [100.0] * 60,
                  "Radio": [50.0] * 150})
    shallow = build({"TV": [100.0] * 60 + [8.0] * 30 + [100.0] * 60,
                     "Radio": [50.0] * 150})
    assert (sub_scores(event(shallow, "TV", 60, 89), shallow)["magnitude_evidence"]
            < sub_scores(event(deep, "TV", 60, 89), deep)["magnitude_evidence"])


def test_a_step_takes_its_magnitude_evidence_from_z():
    """Steps have no run depth -- the channel never stopped -- so the spec
    routes them through |z| / Z_SATURATION instead.

    z is a literal (4.0), not params.Z_SATURATION / 2: deriving the input from
    the constant under test would make the expected ratio move in lockstep
    with it and the assertion would hold for every possible Z_SATURATION.
    params.Z_SATURATION is pinned to 8.0 by test_params.py, so 4.0 / 8.0 is
    hardcoded as the expected 0.5 rather than recomputed from the live
    parameter -- a Z_SATURATION mutation then changes the actual result
    without moving the expectation, and the test catches it."""
    assert params.Z_SATURATION == 8.0
    p = build({"TV": [100.0] * 75 + [300.0] * 75, "Radio": [50.0] * 150})
    e = event(p, "TV", 75, 149, event_type="step_change", evidence={"z": 4.0})
    assert sub_scores(e, p)["magnitude_evidence"] == pytest.approx(0.5)


def test_step_magnitude_evidence_saturates_rather_than_exceeding_one():
    p = build({"TV": [100.0] * 75 + [300.0] * 75, "Radio": [50.0] * 150})
    e = event(p, "TV", 75, 149, event_type="step_change",
              evidence={"z": params.Z_SATURATION * 10})
    assert sub_scores(e, p)["magnitude_evidence"] == 1.0


def test_duration_evidence_grows_with_length_then_saturates():
    """Sized with literals 10 and 40, with the constant asserted to sit between
    them -- deriving the fixture from MIN_DAYS would make it move with the
    parameter and assert nothing."""
    assert 5 < params.MIN_DAYS <= 20
    short = build({"TV": [100.0] * 60 + [0.0] * 10 + [100.0] * 80,
                   "Radio": [50.0] * 150})
    long = build({"TV": [100.0] * 60 + [0.0] * 40 + [100.0] * 50,
                  "Radio": [50.0] * 150})
    s_short = sub_scores(event(short, "TV", 60, 69), short)["duration_evidence"]
    s_long = sub_scores(event(long, "TV", 60, 99), long)["duration_evidence"]
    assert s_short < s_long
    assert s_long == 1.0


def test_distinctiveness_compares_the_run_against_the_series_own_gaps():
    """A 30-day stop on a channel that never otherwise pauses is far more
    distinctive than the same stop on a channel that goes dark every fortnight.
    This is why the comparison is per series and not global.

    The relative comparison alone (flighting < clean) holds for almost any
    positive DISTINCTIVENESS_SATURATION, including a wildly wrong one -- the
    flighting run's ratio is scaled down either way, so it stays below the
    clean run's unconditional 1.0. The final assertion pins the flighting
    score to the literal 2/3: the run is 30 days against a p90 gap of 15 days,
    a ratio of 2.0, and DISTINCTIVENESS_SATURATION is pinned to 3.0 by
    test_params.py, so 2.0 / 3.0 is hardcoded rather than recomputed from the
    live parameter. A saturation mutation changes the actual result without
    moving this expectation."""
    assert params.DISTINCTIVENESS_SATURATION == 3.0
    clean = build({"TV": [100.0] * 60 + [0.0] * 30 + [100.0] * 60,
                   "Radio": [50.0] * 150})
    # Each cycle is [0]*15 + [100]*15 so every pre-event off-run ends on a
    # non-zero day -- index 59 must be non-zero or it fuses with the event's
    # off-run starting at index 60 into one long run, which would erase the
    # very distinction this test is checking.
    flighting_tv = []
    for _ in range(2):
        flighting_tv += [0.0] * 15 + [100.0] * 15
    flighting_tv = flighting_tv + [0.0] * 30 + [100.0] * 60
    flighting = build({"TV": flighting_tv, "Radio": [50.0] * 150})
    flighting_score = sub_scores(event(flighting, "TV", 60, 89),
                                 flighting)["distinctiveness"]
    clean_score = sub_scores(event(clean, "TV", 60, 89), clean)["distinctiveness"]
    assert flighting_score < clean_score
    assert flighting_score == pytest.approx(2 / 3)


def test_corroboration_is_full_when_impressions_stop_with_spend():
    p = build({"TV": [100.0] * 60 + [0.0] * 30 + [100.0] * 60,
               "Radio": [50.0] * 150})
    assert sub_scores(event(p, "TV", 60, 89), p)["corroboration"] == 1.0


def test_corroboration_collapses_when_impressions_keep_flowing():
    """Spend at zero while impressions continue is tracking loss, not a pause.
    The spec gives this case a specifically low score rather than zero, because
    a late spend feed produces the same shape.

    Asserted against the literal 0.2, not params.CORROBORATION_CONTRADICTED:
    comparing the actual score to the live parameter is a tautology that
    passes for whatever value the parameter currently holds. test_params.py
    pins CORROBORATION_CONTRADICTED to 0.2, so 0.2 is the correct literal
    here, and a mutation to the parameter changes the actual result without
    moving this expectation."""
    assert params.CORROBORATION_CONTRADICTED == 0.2
    imps = [100.0] * 150          # impressions never stop
    p = build({"TV": [100.0] * 60 + [0.0] * 30 + [100.0] * 60,
               "Radio": [50.0] * 150},
              impressions_by_channel={"TV": imps})
    assert sub_scores(event(p, "TV", 60, 89), p)["corroboration"] == 0.2


def test_corroboration_is_unknown_when_impressions_are_never_reported():
    """A channel whose impressions feed is empty everywhere OUTSIDE the event
    window (as opposed to inside it, which is the "collapses" case above)
    gives no signal either way: silence during the window could mean the
    pause is corroborated, or could mean this series just never reports
    impressions. corroboration must report unavailable rather than guessing.

    Asserted against the literal 0.6, not params.CORROBORATION_UNKNOWN: the
    previous suite for this file asserted nothing about CORROBORATION_UNKNOWN
    at all, so mutating 0.6 to any other value passed every test here and was
    only caught by test_params.py's transcription check -- not behavioural
    coverage by this project's own rule. test_params.py pins
    CORROBORATION_UNKNOWN to 0.6, so 0.6 is the correct literal here, and a
    mutation to the parameter changes the actual result without moving this
    expectation."""
    assert params.CORROBORATION_UNKNOWN == 0.6
    p = build({"TV": [100.0] * 60 + [0.0] * 30 + [100.0] * 60,
               "Radio": [50.0] * 150},
              impressions_by_channel={"TV": [0.0] * 150})
    assert sub_scores(event(p, "TV", 60, 89), p)["corroboration"] == 0.6


def test_consistency_is_the_fraction_of_channels_agreeing_for_a_dark_period():
    """A dark period claims the whole market. If one of four channels kept
    running, the claim is three-quarters supported."""
    p = build({"A": [100.0] * 60 + [0.0] * 30 + [100.0] * 60,
               "B": [100.0] * 60 + [0.0] * 30 + [100.0] * 60,
               "C": [100.0] * 60 + [0.0] * 30 + [100.0] * 60,
               "D": [100.0] * 150})
    e = event(p, None, 60, 89, event_type="dark_period")
    assert sub_scores(e, p)["consistency"] == pytest.approx(0.75)


def test_consistency_is_full_for_a_single_channel_event():
    """A single_channel event's `channel` names the channel still RUNNING, so
    its claim is about every OTHER channel in the market having stopped.
    single_channel is in COUNTRY_LEVEL_TYPES for exactly this reason -- it is
    a country-wide claim, just an inverted one. When the other channel really
    did stop for the claimed window, consistency is full.

    This test previously left event_type at its natural_holdout default,
    which bypasses the country-level branch entirely and returns 1.0
    unconditionally -- so despite its name it never exercised single_channel
    at all. That is corrected here by actually passing event_type=
    "single_channel"."""
    p = build({"TV": [100.0] * 150,
               "Radio": [100.0] * 60 + [0.0] * 30 + [100.0] * 60})
    e = event(p, "TV", 60, 89, event_type="single_channel")
    assert sub_scores(e, p)["consistency"] == 1.0


def test_single_channel_run_derived_scores_read_the_channels_that_stopped():
    """CRITICAL regression test. A single_channel event's `channel` names the
    channel still RUNNING -- TV and Radio are dark, Digital keeps running,
    channel="Digital". The run-derived sub-scores (magnitude_evidence,
    distinctiveness, edge_sharpness) must be read from TV/Radio's off-run, not
    from Digital, which has no off-run at all and would silently zero out
    every one of them.

    Reading `event.channel` directly for run selection -- instead of routing
    through subject_channels() for every event type -- passed every other
    test in this file while still returning 0.0 for all three of these on
    every single_channel event: nothing else in the suite constructed a real
    single_channel event and inspected the run-derived scores, so the defect
    had no failing test."""
    p = build({"TV": [100.0] * 60 + [0.0] * 30 + [100.0] * 60,
               "Radio": [100.0] * 60 + [0.0] * 30 + [100.0] * 60,
               "Digital": [100.0] * 150})
    e = event(p, "Digital", 60, 89, event_type="single_channel")
    s = sub_scores(e, p)
    assert s["magnitude_evidence"] > 0.0
    assert s["distinctiveness"] > 0.0
    assert s["edge_sharpness"] > 0.0
    # TV and Radio both go to an exact, clean 30-day zero -- with no other
    # off-run on either series to compare against -- so the fully-determined
    # expectation is that all three saturate at 1.0.
    assert s["magnitude_evidence"] == 1.0
    assert s["distinctiveness"] == 1.0
    assert s["edge_sharpness"] == 1.0


def test_distinctiveness_reads_the_subject_channels_gap_history_not_the_named_channels():
    """IMPORTANT regression test (round 2). _select_run correctly picks the
    run from a SUBJECT channel for a single_channel event, but
    _distinctiveness must build its "other off-runs" population from THAT
    SAME channel too -- never from event.channel, which for a single_channel
    event names the channel still RUNNING and can carry a completely
    unrelated gap history of its own.

    TV (the named, running channel) carries an unrelated 25-day off-run right
    at the start of the series, nowhere near the event window. Radio (the
    real subject -- the channel that actually stopped) has no other off-runs
    at all. The previous test above happens to use a named channel (Digital)
    with NO off-run history whatsoever, so it cannot tell "read from the
    right channel" apart from "read from the wrong channel that happens to
    have nothing else on it either" -- both give 1.0 by coincidence. This
    fixture is built specifically so the two channels' gap histories differ:
    reading TV's history (wrong) gives a 30-day run against a p90 of 25 days
    -- ratio 1.2, score 0.4; reading Radio's history (right) finds no other
    off-runs on Radio at all, so distinctiveness must be 1.0."""
    p = build({"TV": [0.0] * 25 + [100.0] * 125,
               "Radio": [100.0] * 60 + [0.0] * 30 + [100.0] * 60})
    e = event(p, "TV", 60, 89, event_type="single_channel")
    assert sub_scores(e, p)["distinctiveness"] == 1.0


def test_edge_sharpness_is_lower_for_a_gradual_wind_down_than_a_clean_stop():
    """A clean stop (spend at full level right up to the run, full level right
    after) must score higher edge_sharpness than a gradual ramp down and back
    up around the same-length run -- the sub-score exists specifically to tell
    an abrupt pause apart from a tapering wind-down."""
    clean = build({"TV": [100.0] * 60 + [0.0] * 30 + [100.0] * 60,
                   "Radio": [50.0] * 150})
    # 7-day ramp on each side of the run (ROLLING = 7), staying above the
    # off-run threshold (RHO * level = 15) so it is not absorbed into the run
    # itself, but far enough below full level to blunt the transition.
    ramp_down = [90.0, 75.0, 60.0, 45.0, 30.0, 22.0, 20.0]
    ramp_up = [20.0, 22.0, 30.0, 45.0, 60.0, 75.0, 90.0]
    gradual_tv = [100.0] * 53 + ramp_down + [0.0] * 30 + ramp_up + [100.0] * 53
    gradual = build({"TV": gradual_tv, "Radio": [50.0] * 150})
    assert (sub_scores(event(gradual, "TV", 60, 89), gradual)["edge_sharpness"]
            < sub_scores(event(clean, "TV", 60, 89), clean)["edge_sharpness"])


def test_an_event_on_a_channel_with_no_off_run_still_scores():
    """Step changes never stop the channel, so the run-derived sub-scores have
    no run to read. They must degrade to a defined value, not raise."""
    p = build({"TV": [100.0] * 75 + [300.0] * 75, "Radio": [50.0] * 150})
    s = sub_scores(event(p, "TV", 75, 149, event_type="step_change",
                         evidence={"z": 5.0}), p)
    assert all(0.0 <= v <= 1.0 for v in s.values())


def test_subject_channels_is_public_and_shared_across_modules():
    """The ruling on Plan 4 tasks 1-2: this helper must be a PUBLIC name in
    detection/score.py so detection/validity.py and detection/explain.py can
    import it across the module boundary without duplicating the logic. A
    leading underscore would declare the opposite of that intent."""
    from detection.score import subject_channels

    p = build({"TV": [100.0] * 60 + [0.0] * 30 + [100.0] * 60,
               "Radio": [50.0] * 150})
    # single_channel names the channel still RUNNING; its subjects are every
    # OTHER channel in the market -- the inversion the ruling calls out.
    e = event(p, "Radio", 60, 89, event_type="single_channel")
    assert subject_channels(e, p) == ["TV"]


def test_confidence_weights_sum_to_one_for_every_event_type():
    """A row that does not sum to 1 silently rescales that type's confidence
    against every other type, which would corrupt the calibration."""
    from detection.model import EVENT_TYPES
    assert set(params.CONFIDENCE_WEIGHTS) == set(EVENT_TYPES)
    for event_type, weights in params.CONFIDENCE_WEIGHTS.items():
        assert sum(weights.values()) == pytest.approx(1.0), event_type


def test_confidence_weights_name_exactly_the_six_sub_scores():
    p = build({"TV": [100.0] * 60 + [0.0] * 30 + [100.0] * 60,
               "Radio": [50.0] * 150})
    names = set(sub_scores(event(p, "TV", 60, 89), p))
    for event_type, weights in params.CONFIDENCE_WEIGHTS.items():
        assert set(weights) == names, event_type


def test_confidence_is_the_weighted_mean_of_the_reported_sub_scores():
    """The returned sub-scores must be the ones that produced the number.
    Reporting one set and scoring another is how an explanation becomes a lie.

    This recomputes from params.CONFIDENCE_WEIGHTS, the same dict the
    implementation reads -- so it is deliberately NOT a check on the weight
    VALUES (those are covered by test_confidence_weights_sum_to_one_...,
    test_confidence_weights_name_exactly_..., and the ordering tests below).
    What it does catch: confidence() scoring a sub_scores() call that differs
    from the parts it returns, indexing the wrong event type's row, missing a
    term from the sum, or skipping the [0, 1] clip -- all of which would move
    `score` away from `expected` while `weights` and `parts` stay identical on
    both sides."""
    p = build({"TV": [100.0] * 60 + [0.0] * 30 + [100.0] * 60,
               "Radio": [50.0] * 150})
    e = event(p, "TV", 60, 89)
    score, parts = confidence(e, p)
    weights = params.CONFIDENCE_WEIGHTS["natural_holdout"]
    expected = sum(parts[k] * weights[k] for k in weights)
    assert score == pytest.approx(expected)
    assert 0.0 <= score <= 1.0


def test_a_clean_long_exact_zero_outscores_a_short_shallow_one():
    """The ordering is the whole point of the score. If this inverts, ranking
    by confidence is worse than not ranking."""
    strong = build({"TV": [100.0] * 60 + [0.0] * 40 + [100.0] * 50,
                    "Radio": [50.0] * 150})
    weak = build({"TV": [100.0] * 60 + [9.0] * 8 + [100.0] * 82,
                  "Radio": [50.0] * 150})
    assert (confidence(event(strong, "TV", 60, 99), strong)[0]
            > confidence(event(weak, "TV", 60, 67), weak)[0])


def test_tracking_loss_lowers_confidence_against_an_identical_clean_event():
    """Same window, same depth, same duration -- the ONLY difference is that
    impressions kept flowing. Confidence must notice."""
    clean = build({"TV": [100.0] * 60 + [0.0] * 30 + [100.0] * 60,
                   "Radio": [50.0] * 150})
    tracked = build({"TV": [100.0] * 60 + [0.0] * 30 + [100.0] * 60,
                     "Radio": [50.0] * 150},
                    impressions_by_channel={"TV": [100.0] * 150})
    assert (confidence(event(tracked, "TV", 60, 89), tracked)[0]
            < confidence(event(clean, "TV", 60, 89), clean)[0])


def test_informativeness_prefers_a_dark_period_to_a_step_of_equal_length():
    """Spec section 8's type prior: a dark period lets the baseline be read
    directly, a step change does not."""
    p = build({"TV": [100.0] * 60 + [0.0] * 40 + [100.0] * 50,
               "Radio": [50.0] * 150})
    dark = event(p, None, 60, 99, event_type="dark_period")
    step = event(p, "TV", 60, 99, event_type="step_change",
                 evidence={"z": 6.0})
    assert informativeness(dark, p)[0] > informativeness(step, p)[0]


def test_a_window_too_short_to_show_adstock_decay_scores_low_duration():
    """Sized with literals against ADSTOCK_HALF_LIFE asserted, not derived."""
    assert params.ADSTOCK_HALF_LIFE >= 5.0
    # A 40-day window must saturate duration_adequacy regardless of the exact
    # half-life/window-count values, as long as they stay within this bound --
    # pinned here rather than recomputed from the live parameters.
    assert params.ADSTOCK_HALF_LIFE * params.ADSTOCK_WINDOWS_FOR_FULL_CREDIT <= 40.0
    p = build({"TV": [100.0] * 60 + [0.0] * 8 + [100.0] * 82,
               "Radio": [50.0] * 150})
    long_p = build({"TV": [100.0] * 60 + [0.0] * 40 + [100.0] * 50,
                    "Radio": [50.0] * 150})
    _, short_parts = informativeness(event(p, "TV", 60, 67), p)
    _, long_parts = informativeness(event(long_p, "TV", 60, 99), long_p)
    assert short_parts["duration_adequacy"] < long_parts["duration_adequacy"]
    assert long_parts["duration_adequacy"] == 1.0


def test_control_availability_follows_the_cross_market_tag():
    """The cross-market layer already worked out whether peers were running;
    informativeness must use that answer rather than guessing again."""
    p = build({"TV": [100.0] * 60 + [0.0] * 30 + [100.0] * 60,
               "Radio": [50.0] * 150})
    with_peers = event(p, "TV", 60, 89, evidence={"control_available": "peers"})
    without = event(p, "TV", 60, 89, evidence={"control_available": "none"})
    assert informativeness(with_peers, p)[0] > informativeness(without, p)[0]


def test_a_censored_event_is_worth_less_than_the_same_event_fully_observed():
    """An event running to the edge of the series has unknown true extent."""
    p = build({"TV": [0.0] * 40 + [100.0] * 110, "Radio": [50.0] * 150})
    censored = event(p, "TV", 0, 39, evidence={"censored_start": True})
    clean_p = build({"TV": [100.0] * 60 + [0.0] * 40 + [100.0] * 50,
                     "Radio": [50.0] * 150})
    clean = event(clean_p, "TV", 60, 99)
    assert informativeness(censored, p)[0] < informativeness(clean, clean_p)[0]


def test_a_confounded_event_is_worth_less_than_the_same_event_clean():
    """The brief that specified these tests never exercised the cleanliness
    driver at all -- `tags=("confounded",)` never appeared anywhere in this
    file, so CONFOUNDED_PENALTY could be set to 1.0 (a no-op) and nothing
    here would fail. Added to close that hole.

    Asserted against the literal 0.6, not params.CONFOUNDED_PENALTY:
    test_params.py pins CONFOUNDED_PENALTY to 0.6, so 0.6 is the correct
    literal here, and a mutation to the parameter changes the actual result
    without moving this expectation."""
    assert params.CONFOUNDED_PENALTY == 0.6
    p = build({"TV": [100.0] * 60 + [0.0] * 30 + [100.0] * 60,
               "Radio": [50.0] * 150})
    clean = event(p, "TV", 60, 89)
    confounded = event(p, "TV", 60, 89, tags=("confounded",))
    _, confounded_parts = informativeness(confounded, p)
    assert confounded_parts["cleanliness"] == pytest.approx(0.6)
    assert informativeness(confounded, p)[0] < informativeness(clean, p)[0]


def test_contrast_for_a_step_change_reads_the_ratio_not_a_run_depth():
    """A step change never stops the channel, so there is no off-run for
    contrast to read a depth from -- it must fall back to how far |log(ratio)|
    carries the level, on the same saturating scale distinctiveness uses.

    magnitude_ratio is the literal 3 ** 0.5, chosen so that, WITH
    DISTINCTIVENESS_SATURATION pinned to 3.0 by test_params.py,
    |log(ratio)| / log(3.0) works out to the literal 0.5 -- half of the
    saturation input produces half credit. The expected value does not move
    if DISTINCTIVENESS_SATURATION is mutated; the actual result does."""
    assert params.DISTINCTIVENESS_SATURATION == 3.0
    p = build({"TV": [100.0] * 75 + [300.0] * 75, "Radio": [50.0] * 150})
    e = event(p, "TV", 75, 149, event_type="step_change",
              magnitude_ratio=3 ** 0.5, evidence={"z": 6.0})
    _, parts = informativeness(e, p)
    assert parts["contrast"] == pytest.approx(0.5)


def test_informativeness_is_bounded_and_reports_its_drivers():
    p = build({"TV": [100.0] * 60 + [0.0] * 30 + [100.0] * 60,
               "Radio": [50.0] * 150})
    score, parts = informativeness(event(p, "TV", 60, 89), p)
    assert 0.0 <= score <= 1.0
    assert set(parts) == set(params.INFORMATIVENESS_WEIGHTS)
