import pandas as pd

from detection import params
from detection.compose.label import label_market, segment_regimes
from detection.io.panel import build_panel

COLS = ["date", "ad_platform", "advertising_channel", "campaign_name",
        "campaign_id", "media_investment", "clicks", "impressions",
        "conversions", "conversion_value", "country_code"]


def build(spend_by_channel, country="DE", start="2024-01-01"):
    """spend_by_channel: {channel: [daily spend]}"""
    n = len(next(iter(spend_by_channel.values())))
    dates = pd.date_range(start, periods=n)
    rows = []
    for ch, values in spend_by_channel.items():
        for d, v in zip(dates, values):
            rows.append([d, "P", ch, "c", 1, float(v), 1.0, 10.0, 0, 0.0,
                         country])
    return build_panel(pd.DataFrame(rows, columns=COLS), sid="dev_test")


def types(events):
    return sorted({e.event_type for e in events})


def test_all_channels_off_is_a_dark_period():
    p = build({"TV": [100] * 40 + [0] * 20 + [100] * 40,
               "Radio": [50] * 40 + [0] * 20 + [50] * 40})
    events = [e for e in label_market(p, "DE", "dev_test")
              if e.event_type == "dark_period"]
    assert len(events) == 1
    e = events[0]
    assert e.channel is None
    assert e.start == pd.Timestamp("2024-02-10")
    assert e.n_days == 20


def test_one_channel_off_is_a_natural_holdout_naming_the_off_channel():
    p = build({"TV": [100] * 40 + [0] * 20 + [100] * 40,
               "Radio": [50] * 100})
    events = [e for e in label_market(p, "DE", "dev_test")
              if e.event_type == "natural_holdout"]
    assert len(events) == 1
    assert events[0].channel == "TV"


def test_exactly_one_channel_left_running_is_a_single_channel_period():
    """The channel named is the one that stays ON -- the opposite convention to
    natural_holdout, inherited from the generator and asserted by the harness."""
    p = build({"TV": [100] * 40 + [0] * 20 + [100] * 40,
               "Radio": [50] * 40 + [0] * 20 + [50] * 40,
               "Search": [30] * 100})
    events = [e for e in label_market(p, "DE", "dev_test")
              if e.event_type == "single_channel"]
    assert len(events) == 1
    assert events[0].channel == "Search"


def test_a_dark_period_does_not_also_emit_its_component_holdouts():
    """Regime segmentation resolves the nesting structurally: a dark period IS
    the regime where nothing is active, and its constituent per-channel
    holdouts ride along as components rather than competing events."""
    p = build({"TV": [100] * 40 + [0] * 20 + [100] * 40,
               "Radio": [50] * 40 + [0] * 20 + [50] * 40})
    events = label_market(p, "DE", "dev_test")
    assert types(events) == ["dark_period"]
    assert len(events[0].components) == 2


def test_a_regime_shorter_than_min_days_is_ignored():
    p = build({"TV": [100] * 40 + [0] * 3 + [100] * 40,
               "Radio": [50] * 83})
    assert label_market(p, "DE", "dev_test") == []


def test_a_channel_never_run_in_this_market_is_not_a_holdout():
    """A market that simply does not use a channel must not be reported as
    holding it out for the entire series."""
    p = build({"TV": [100] * 80, "Radio": [0] * 80})
    assert label_market(p, "DE", "dev_test") == []


def test_a_channel_off_only_at_the_start_is_not_a_holdout():
    """That is a launch, and P4 plus the cross-market layer own it."""
    p = build({"TV": [0] * 30 + [100] * 60, "Radio": [50] * 90})
    holdouts = [e for e in label_market(p, "DE", "dev_test")
                if e.event_type == "natural_holdout"]
    assert holdouts == []


def test_a_step_change_is_labelled_with_its_ratio():
    """A bounded threefold increase carries its observed budget ratio."""
    import numpy as np
    rng = np.random.default_rng(11)
    tv = (list(100 * (1 + rng.normal(0, 0.05, 120)))
          + list(300 * (1 + rng.normal(0, 0.05, 90)))
          + list(100 * (1 + rng.normal(0, 0.05, 120))))
    p = build({"TV": tv, "Radio": [50] * 330})
    steps = [e for e in label_market(p, "DE", "dev_test")
             if e.event_type == "step_change"]
    assert steps
    assert 2.6 < steps[0].magnitude_ratio < 3.4


def test_a_pulse_train_is_one_event_with_components():
    values = []
    for _ in range(4):
        values += [100] * 21 + [0] * 14
    values += [100] * 21
    p = build({"TV": values, "Radio": [50] * len(values)})
    pulses = [e for e in label_market(p, "DE", "dev_test")
              if e.event_type == "channel_pulse"]
    assert len(pulses) == 1
    assert len(pulses[0].components) == 4


def test_a_pulse_train_suppresses_its_individual_holdouts():
    """Otherwise the same windows are reported twice, once grouped and once
    per-window, and precision collapses."""
    values = []
    for _ in range(4):
        values += [100] * 21 + [0] * 14
    values += [100] * 21
    p = build({"TV": values, "Radio": [50] * len(values)})
    events = label_market(p, "DE", "dev_test")
    assert not [e for e in events
                if e.event_type == "natural_holdout" and e.channel == "TV"]


def test_a_holdout_is_not_also_reported_as_a_step_change():
    """A channel dropping to zero and back is a huge level shift at both edges.
    Without a guard the same window is reported twice -- once as a holdout and
    once as a step -- and precision collapses."""
    p = build({"TV": [100] * 60 + [0] * 30 + [100] * 60,
               "Radio": [50] * 150})
    events = label_market(p, "DE", "dev_test")
    assert [e for e in events
            if e.event_type == "step_change" and e.channel == "TV"] == []
    assert [e for e in events if e.event_type == "natural_holdout"]


def test_a_dark_period_is_not_also_reported_as_step_changes():
    p = build({"TV": [100] * 60 + [0] * 30 + [100] * 60,
               "Radio": [50] * 60 + [0] * 30 + [50] * 60})
    assert types(label_market(p, "DE", "dev_test")) == ["dark_period"]


def test_regimes_partition_the_timeline_without_gaps_or_overlaps():
    p = build({"TV": [100] * 40 + [0] * 20 + [100] * 40,
               "Radio": [50] * 100})
    regimes = segment_regimes(p, "DE")
    assert regimes[0].start == p.dates[0]
    assert regimes[-1].end == p.dates[-1]
    for a, b in zip(regimes, regimes[1:]):
        assert (b.start - a.end).days == 1


def test_events_carry_the_sid():
    p = build({"TV": [100] * 40 + [0] * 20 + [100] * 40,
               "Radio": [50] * 100})
    assert all(e.sid == "dev_test" for e in label_market(p, "DE", "dev_test"))


# --- guards the brief left unpinned -----------------------------------------

def test_two_channels_with_one_off_is_a_holdout_not_a_single_channel_period():
    """The spec table says single_channel needs "at least 2 channels"; that is
    off by one. Two channels with one off leaves exactly one running, but the
    generator calls that a natural_holdout naming the channel that STOPPED.
    single_channel needs MORE THAN ONE channel to have gone dark."""
    p = build({"TV": [100] * 40 + [0] * 20 + [100] * 40,
               "Radio": [50] * 100})
    events = label_market(p, "DE", "dev_test")
    assert types(events) == ["natural_holdout"]
    assert events[0].channel == "TV"


def test_a_market_running_exactly_one_channel_reports_nothing():
    """With no `len(off) > 1` floor, the single regime of a one-channel market
    has |active| == 1 and an empty off set, and `all(...)` over an empty set is
    True -- so the market's entire history reports as a single_channel period."""
    p = build({"TV": [100] * 120})
    assert label_market(p, "DE", "dev_test") == []


def test_a_channel_off_only_at_the_end_is_not_a_holdout():
    """The mirror of the launch case: a discontinuation is censored at the
    series end, and the cross-market layer owns it. This pins the `after` half
    of the was-active-around guard, which the launch test cannot reach."""
    p = build({"TV": [100] * 60 + [0] * 30, "Radio": [50] * 90})
    assert label_market(p, "DE", "dev_test") == []


def test_synchronised_pulse_trains_do_not_also_read_as_single_channel():
    """Two channels pulsing in lockstep leave a third one alone in every gap.
    Without the pulse claim on the single_channel branch, each of those gaps is
    ALSO reported as a single-channel period -- the same windows twice."""
    values = []
    for _ in range(4):
        values += [100] * 21 + [0] * 14
    values += [100] * 21
    p = build({"TV": values, "Radio": [v / 2 for v in values],
               "Search": [30] * len(values)})
    events = label_market(p, "DE", "dev_test")
    assert types(events) == ["channel_pulse"]
    assert len({e.channel for e in events}) == 2


def test_the_regime_length_floor_is_the_min_days_parameter():
    """The floor is pinned with LITERAL regime lengths, so the test cannot move
    with the constant it is testing: 3 days is below MIN_DAYS and 20 is above,
    and the assertion on params states exactly that."""
    assert 3 < params.MIN_DAYS <= 20
    short = build({"TV": [100] * 40 + [0] * 3 + [100] * 40,
                   "Radio": [50] * 83})
    long_ = build({"TV": [100] * 40 + [0] * 20 + [100] * 40,
                   "Radio": [50] * 100})
    assert label_market(short, "DE", "dev_test") == []
    assert len(label_market(long_, "DE", "dev_test")) == 1


def test_a_step_that_never_reverts_is_reported_as_censored():
    """An observed permanent budget change has an unknown closing date."""
    import numpy as np
    from detection.primitives.level_shift import find_step_episodes
    rng = np.random.default_rng(11)
    tv = list(100 * (1 + rng.normal(0, 0.05, 120))) + \
         list(300 * (1 + rng.normal(0, 0.05, 120)))
    p = build({"TV": tv, "Radio": [50] * 240})

    episodes = find_step_episodes(p.series("DE", "TV"))
    assert len(episodes) == 1 and episodes[0].open_ended, (
        "fixture no longer produces an open-ended episode, so this test would "
        "pass for the wrong reason")

    steps = [e for e in label_market(p, "DE", "dev_test") if e.event_type == "step_change"]
    assert len(steps) == 1
    assert steps[0].end == p.dates[-1]
    assert steps[0].evidence["censored_end"] is True


def test_a_step_that_reverts_is_bounded():
    """The shape the generator actually injects: spend rises, holds, reverts.
    The episode must close on the reversal rather than running to series end."""
    import numpy as np
    rng = np.random.default_rng(7)
    tv = (list(100 * (1 + rng.normal(0, 0.05, 120)))
          + list(300 * (1 + rng.normal(0, 0.05, 90)))
          + list(100 * (1 + rng.normal(0, 0.05, 120))))
    p = build({"TV": tv, "Radio": [50] * 330})
    steps = [e for e in label_market(p, "DE", "dev_test")
             if e.event_type == "step_change"]
    assert len(steps) == 1
    assert steps[0].end < p.dates[-1]
    assert 70 < steps[0].n_days < 110


def test_every_emitted_type_is_a_known_event_type():
    from detection.model import EVENT_TYPES
    p = build({"TV": [100] * 40 + [0] * 20 + [100] * 40,
               "Radio": [50] * 40 + [0] * 20 + [50] * 40,
               "Search": [30] * 100})
    for e in label_market(p, "DE", "dev_test"):
        assert e.event_type in EVENT_TYPES
        assert e.country_code == "DE"
        assert e.start <= e.end


def test_no_step_episode_can_be_shorter_than_the_regime_floor():
    """Why _step_events carries no MIN_DAYS floor.

    find_level_shifts keeps one shift per W-wide neighbourhood, so a closed
    episode spans at least W days and an open-ended one runs to the series end
    from at least W days out. While W is at least MIN_DAYS no step episode can
    be too short, and a floor there would be a gate no input could trip. If
    this assertion ever fails, reinstate the floor.
    """
    assert params.W >= params.MIN_DAYS


def test_channels_dormant_from_day_one_are_not_a_single_channel_period():
    """The _was_active_around guard on the SINGLE_CHANNEL branch had no test.
    Its twin on the holdout branch did, which made this copy read as covered --
    bypassing it changed no test while emitting a spurious 120-day event.

    Two of three channels dormant from day one is a market that had not started
    them yet, not a market that switched them off. On a multi-market panel
    find_staggered_launches claims this window first, which is why no
    development scenario reaches this branch; a single-market scenario does, and
    the generator's shape rules allow one.
    """
    p = build({
        "TV": [100.0] * 300,
        "Radio": [0.0] * 120 + [80.0] * 180,
        "Print": [0.0] * 120 + [60.0] * 180,
    })
    events = label_market(p, "DE", "dev_test")
    assert [e for e in events if e.event_type == "single_channel"] == []
