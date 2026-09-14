import pandas as pd

from detection import params
from detection.compose.cross_market import annotate, find_staggered_launches
from detection.compose.label import label_market
from detection.io.panel import build_panel
from detection.model import DetectedEvent

COLS = ["date", "ad_platform", "advertising_channel", "campaign_name",
        "campaign_id", "media_investment", "clicks", "impressions",
        "conversions", "conversion_value", "country_code"]


def build(by_country, start="2024-01-01"):
    """by_country: {country: {channel: [daily spend]}}"""
    rows = []
    for country, channels in by_country.items():
        for ch, values in channels.items():
            dates = pd.date_range(start, periods=len(values))
            for d, v in zip(dates, values):
                rows.append([d, "P", ch, "c", 1, float(v), 1.0, 10.0, 0, 0.0,
                             country])
    return build_panel(pd.DataFrame(rows, columns=COLS), sid="dev_test")


# --- control availability ---------------------------------------------------

def test_a_holdout_with_healthy_peers_is_tagged_cross_market():
    """The most MMM-valuable finding the system can produce: the peers are a
    ready-made control group."""
    p = build({
        "DE": {"TV": [100] * 40 + [0] * 20 + [100] * 40, "Radio": [50] * 100},
        "AT": {"TV": [100] * 100, "Radio": [50] * 100},
    })
    events = annotate(label_market(p, "DE", "dev_test"), p)
    holdout = [e for e in events if e.event_type == "natural_holdout"][0]
    assert "cross_market_holdout" in holdout.tags
    assert holdout.evidence["control_available"] == "peers"


def test_a_holdout_with_every_peer_also_off_is_tagged_global_pause():
    """Real, but with no control group -- and this is also what a pipeline
    outage looks like, so validity suspicion belongs here."""
    p = build({
        "DE": {"TV": [100] * 40 + [0] * 20 + [100] * 40, "Radio": [50] * 100},
        "AT": {"TV": [100] * 40 + [0] * 20 + [100] * 40, "Radio": [50] * 100},
    })
    events = annotate(label_market(p, "DE", "dev_test"), p)
    holdout = [e for e in events if e.event_type == "natural_holdout"][0]
    assert "global_pause" in holdout.tags
    assert holdout.evidence["control_available"] == "none"


def test_a_dark_period_with_healthy_peers_reports_peer_control():
    p = build({
        "DE": {"TV": [100] * 40 + [0] * 20 + [100] * 40,
               "Radio": [50] * 40 + [0] * 20 + [50] * 40},
        "AT": {"TV": [100] * 100, "Radio": [50] * 100},
    })
    dark = [e for e in annotate(label_market(p, "DE", "dev_test"), p)
            if e.event_type == "dark_period"][0]
    assert dark.evidence["control_available"] == "peers"
    assert "global_pause" not in dark.tags
    # The cross-market holdout tag names a single channel's control group and
    # must not be attached to a whole-market blackout.
    assert "cross_market_holdout" not in dark.tags


def test_a_single_market_scenario_reports_no_peer_control():
    p = build({"DE": {"TV": [100] * 40 + [0] * 20 + [100] * 40,
                      "Radio": [50] * 100}})
    holdout = [e for e in annotate(label_market(p, "DE", "dev_test"), p)
               if e.event_type == "natural_holdout"][0]
    assert holdout.evidence["control_available"] == "sibling_channels"


def test_a_peer_that_never_ran_the_channel_is_not_a_control():
    """An all-zero channel means the market never bought it. Counting that
    market as a peer running normally would invent a control group out of a
    channel nobody ever ran."""
    p = build({
        "DE": {"TV": [100] * 40 + [0] * 20 + [100] * 40, "Radio": [50] * 100},
        "AT": {"TV": [0] * 100, "Radio": [50] * 100},
    })
    holdout = [e for e in annotate(label_market(p, "DE", "dev_test"), p)
               if e.event_type == "natural_holdout"][0]
    assert holdout.evidence["control_available"] == "sibling_channels"
    assert holdout.evidence["peers_running"] == 0
    assert holdout.evidence["peers_off"] == 0


def test_a_peer_off_for_only_part_of_the_window_is_still_a_control():
    """A peer has to be off for the WHOLE window to lose its control status:
    a peer that ran for most of the window still carries the contrast."""
    p = build({
        "DE": {"TV": [100] * 40 + [0] * 20 + [100] * 40, "Radio": [50] * 100},
        "AT": {"TV": [100] * 40 + [0] * 10 + [100] * 50, "Radio": [50] * 100},
    })
    holdout = [e for e in annotate(label_market(p, "DE", "dev_test"), p)
               if e.event_type == "natural_holdout"][0]
    assert holdout.evidence["control_available"] == "peers"
    assert "global_pause" not in holdout.tags


def test_a_single_channel_period_asks_about_the_channels_that_WENT_OFF():
    """A single_channel event names the channel still LIVE, so asking the
    peers about that channel asks the opposite question. Here every market
    keeps TV and drops both others at once: there is no control group, even
    though TV is running everywhere."""
    off = [50] * 40 + [0] * 20 + [50] * 40
    p = build({
        "DE": {"TV": [100] * 100, "Radio": list(off), "Print": list(off)},
        "AT": {"TV": [100] * 100, "Radio": list(off), "Print": list(off)},
    })
    single = [e for e in annotate(label_market(p, "DE", "dev_test"), p)
              if e.event_type == "single_channel"][0]
    assert single.channel == "TV"
    assert single.evidence["control_available"] == "none"
    assert "global_pause" in single.tags


def test_annotate_keeps_a_pulse_train_whole():
    """The harness groups a pulsing channel's truth into one interval, so a
    train must survive annotation as ONE event with its windows attached."""
    cycle = [100] * 30 + [0] * 12
    p = build({
        "DE": {"TV": cycle * 4, "Radio": [50] * (42 * 4)},
        "AT": {"TV": [100] * (42 * 4), "Radio": [50] * (42 * 4)},
    })
    pulses = [e for e in annotate(label_market(p, "DE", "dev_test"), p)
              if e.event_type == "channel_pulse"]
    assert len(pulses) == 1
    assert len(pulses[0].components) > 1
    assert pulses[0].evidence["control_available"] == "peers"


def test_annotate_preserves_events_it_cannot_classify():
    e = DetectedEvent(sid="dev_test", country_code="DE", channel="TV",
                      event_type="step_change",
                      start=pd.Timestamp("2024-02-01"),
                      end=pd.Timestamp("2024-03-01"))
    p = build({"DE": {"TV": [100] * 100}, "AT": {"TV": [100] * 100}})
    out = annotate([e], p)
    assert len(out) == 1 and out[0].event_type == "step_change"
    # Untouched, not merely present: peer availability is not established for
    # a step change, so no control claim may be attached to one.
    assert out[0] == e
    assert "control_available" not in out[0].evidence


# --- staggered launches -----------------------------------------------------

def test_staggered_launch_is_emitted_once_per_LATE_market():
    """The harness's truth is PER-MARKET, so a country-less panel event scores
    zero. benchmark/eval/README.md states the fan-out is the detector's job."""
    p = build({
        "DE": {"TV": [100] * 200},
        "AT": {"TV": [0] * 60 + [100] * 140},
        "CH": {"TV": [0] * 120 + [100] * 80},
    })
    launches = find_staggered_launches(p, "dev_test")
    assert {e.country_code for e in launches} == {"AT", "CH"}
    assert all(e.event_type == "staggered_launch" for e in launches)
    assert all(e.country_code is not None for e in launches)


def test_a_launch_window_runs_from_series_start_to_the_day_before_activity():
    p = build({"DE": {"TV": [100] * 200}, "AT": {"TV": [0] * 60 + [100] * 140}})
    at = [e for e in find_staggered_launches(p, "dev_test")
          if e.country_code == "AT"][0]
    assert at.start == p.dates[0]
    assert at.end == p.dates[59]


def test_the_first_market_to_launch_is_the_comparison_group():
    """Nothing was live anywhere during the first market's own dormancy, so
    that dormancy has no peer evidence behind it and is not reported."""
    p = build({
        "DE": {"TV": [0] * 30 + [100] * 170},
        "AT": {"TV": [0] * 120 + [100] * 80},
    })
    assert [e.country_code for e in find_staggered_launches(p, "dev_test")] \
        == ["AT"]


def test_a_market_that_never_ran_the_channel_is_not_a_day_one_witness():
    """CH never bought TV, so it cannot be the market that proves TV was
    runnable from day one. Counting it as one would make DE -- itself a late
    starter, and the real comparison group here -- look like a launch too."""
    p = build({
        "DE": {"TV": [0] * 30 + [100] * 170, "Radio": [50] * 200},
        "AT": {"TV": [0] * 120 + [100] * 80, "Radio": [50] * 200},
        "CH": {"TV": [0] * 200, "Radio": [50] * 200},
    })
    assert [e.country_code for e in find_staggered_launches(p, "dev_test")] \
        == ["AT"]


def test_markets_starting_together_are_not_a_staggered_launch():
    """Coincidental start-up jitter is not a rollout."""
    p = build({
        "DE": {"TV": [0] * 30 + [100] * 170},
        "AT": {"TV": [0] * 32 + [100] * 168},
    })
    # The spread here is two days, and the rollout above spans ninety.
    assert 2 < params.ONSET_SPREAD <= 90
    assert find_staggered_launches(p, "dev_test") == []


def test_a_channel_live_everywhere_from_day_one_is_not_a_launch():
    p = build({"DE": {"TV": [100] * 200}, "AT": {"TV": [100] * 200}})
    assert find_staggered_launches(p, "dev_test") == []


def test_a_channel_no_market_ever_ran_is_not_a_launch():
    """A channel booked at zero everywhere is a channel the advertiser does
    not use. It has no onset anywhere, and must not raise."""
    p = build({
        "DE": {"TV": [100] * 200, "Print": [0] * 200},
        "AT": {"TV": [100] * 200, "Print": [0] * 200},
    })
    assert find_staggered_launches(p, "dev_test") == []
