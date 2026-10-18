import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from detection.compose.staggered_launch.detector import find_staggered_launches
from detection.io.panel import build_panel
from detection.compose.staggered_launch.readiness import verify_staggered_launches
from detection.model import DetectedEvent
from detection.pipeline import run_detection


def frames(target_onset=70, peer_onset=120):
    dates = pd.date_range("2024-01-01", periods=140)
    media, sales = [], []
    profile = [100, 120, 105, 130, 115, 125, 110, 135]
    for country, onset, scale in [("DE", 0, 10), ("AT", target_onset, 1), ("CH", peer_onset, 20)]:
        for i, date in enumerate(dates):
            for channel, spend in [("TV", 100 if i >= onset else 0), ("Radio", 50)]:
                media.append({"date": date, "country_code": country,
                              "advertising_channel": channel, "media_investment": spend * scale})
            sales.append({"date": date, "country_code": country,
                          "turnover": float(profile[(i // 7) % len(profile)] * scale)})
    return pd.DataFrame(media), pd.DataFrame(sales)


def assessment(media, sales):
    panel = build_panel(media, sales)
    events = find_staggered_launches(panel, "example")
    verified = verify_staggered_launches(events, panel)
    event = next(e for e in verified if e.country_code == "AT")
    return event.evidence["launch_verification"]


def test_ready_launch_uses_actual_onset_and_scaled_comparable_sales():
    media, sales = frames()
    result = assessment(media, sales)
    assert result["status"] == "analysis_ready"
    assert result["qualified_peers"] == ["CH"]
    assert result["windows"] == {"pre_start": "2024-01-15", "pre_end": "2024-03-10",
                                  "post_start": "2024-03-11", "post_end": "2024-04-07"}
    assert result["reason_codes"] == []
    assert json.loads(json.dumps(result, allow_nan=False)) == result


@pytest.mark.parametrize("peer_onset, expected", [(97, "not_analysis_ready"), (98, "analysis_ready"),
                                                   (140, "analysis_ready"), (0, "not_analysis_ready")])
def test_peer_must_remain_untreated_through_the_last_post_day(peer_onset, expected):
    media, sales = frames(peer_onset=peer_onset)
    assert assessment(media, sales)["status"] == expected


def test_missing_peer_rows_are_unknown_not_zero():
    media, sales = frames()
    media = media.loc[~((media.country_code == "CH") & (media.advertising_channel == "TV")
                        & (media.date == pd.Timestamp("2024-02-01")))]
    result = assessment(media, sales)
    assert result["status"] == "needs_review"
    assert result["qualified_peers"] == []


@pytest.mark.parametrize("onset", [55, 113])
def test_short_history_does_not_silently_shrink_the_windows(onset):
    media, sales = frames(target_onset=onset, peer_onset=140)
    result = assessment(media, sales)
    assert result["status"] == "not_analysis_ready"
    assert "insufficient_history" in result["reason_codes"]


@pytest.mark.parametrize("country", ["AT", "CH"])
def test_other_channel_budget_change_blocks_readiness(country):
    media, sales = frames()
    mask = (media.country_code == country) & (media.advertising_channel == "Radio")
    media.loc[mask & (media.date >= "2024-03-11"), "media_investment"] *= 2
    assert assessment(media, sales)["status"] == "not_analysis_ready"


@pytest.mark.parametrize("sales_case", ["absent", "missing", "infinite", "negative", "flat"])
def test_missing_invalid_or_inconclusive_sales_require_review(sales_case):
    media, sales = frames()
    if sales_case == "absent":
        sales = None
    elif sales_case == "flat":
        sales.loc[sales.country_code == "CH", "turnover"] = 100
    else:
        value = {"missing": np.nan, "infinite": np.inf, "negative": -1}[sales_case]
        sales.loc[(sales.country_code == "CH") & (sales.date == "2024-03-20"), "turnover"] = value
    assert assessment(media, sales)["status"] == "needs_review"


def test_divergent_prelaunch_sales_fail_and_no_sales_lift_is_required():
    media, sales = frames()
    sales.loc[(sales.country_code == "AT") & (sales.date >= "2024-03-11"), "turnover"] = 0
    assert assessment(media, sales)["status"] == "analysis_ready"
    pre = (sales.country_code == "CH") & sales.date.between("2024-01-15", "2024-03-10")
    sales.loc[pre, "turnover"] = 10000 - sales.loc[pre, "turnover"]
    assert assessment(media, sales)["status"] == "not_analysis_ready"


def test_high_correlation_does_not_hide_a_large_relative_drift():
    media, sales = frames()
    pre = (sales.country_code == "CH") & sales.date.between("2024-01-15", "2024-03-10")
    week = np.arange(8)
    sales.loc[pre, "turnover"] = np.repeat((1 + week / 20) * 2000, 7)
    target_pre = (sales.country_code == "AT") & sales.date.between("2024-01-15", "2024-03-10")
    sales.loc[target_pre, "turnover"] = np.repeat((1 + week / 5) * 100, 7)
    result = assessment(media, sales)
    peer = next(p for p in result["peers"] if p["country_code"] == "CH")
    assert peer["checks"]["sales"]["correlation"] == pytest.approx(1)
    assert peer["checks"]["sales"]["relative_drift"] > 0.1
    assert result["status"] == "not_analysis_ready"


@pytest.mark.parametrize("channel", ["TV", "Radio"])
def test_missing_target_records_require_review(channel):
    media, sales = frames()
    media = media.loc[~((media.country_code == "AT") & (media.advertising_channel == channel)
                        & (media.date == "2024-02-01"))]
    assert assessment(media, sales)["status"] == "needs_review"


def test_passing_peer_wins_over_a_failed_peer_and_output_is_deterministic():
    media, sales = frames()
    extra = media.loc[media.country_code == "CH"].assign(country_code="FI")
    extra.loc[extra.advertising_channel == "TV", "media_investment"] = 100
    media = pd.concat([media, extra], ignore_index=True)
    assert assessment(media, sales)["status"] == "analysis_ready"
    assert assessment(media, sales) == assessment(media.iloc[::-1], sales.iloc[::-1])


def test_verification_preserves_other_events_and_original_evidence():
    media, sales = frames()
    panel = build_panel(media, sales)
    events = find_staggered_launches(panel, "example")
    other = DetectedEvent("example", "AT", "Radio", "step_change", panel.dates[10], panel.dates[12])
    verified = verify_staggered_launches([other, *events], panel)
    assert verified[0] is other
    assert all("launch_verification" not in e.evidence for e in events)
    assert replace(verified[1], evidence=events[0].evidence) == events[0]


def test_pipeline_keeps_detection_independent_of_sales_and_explains_readiness():
    media, sales = frames()
    with_sales = run_detection(media, sales, "example")
    without_sales = run_detection(media, None, "example")
    identity = lambda e: (e.event_type, e.country_code, e.channel, e.start, e.end,
                          e.validity, e.validity_reasons, e.detection_confidence)
    assert list(map(identity, with_sales)) == list(map(identity, without_sales))
    launch = next(e for e in with_sales if e.country_code == "AT" and e.event_type == "staggered_launch")
    assert launch.evidence["launch_verification"]["status"] == "analysis_ready"
    assert "Launch analysis readiness: analysis_ready" in launch.explanation


def test_conflicting_event_and_later_target_stop_block_readiness():
    media, sales = frames()
    panel = build_panel(media, sales)
    events = find_staggered_launches(panel, "example")
    other = DetectedEvent("example", "CH", "Radio", "natural_holdout", panel.dates[60], panel.dates[68])
    result = verify_staggered_launches([*events, other], panel)
    assert next(e for e in result if e.country_code == "AT").evidence["launch_verification"]["status"] == "not_analysis_ready"
    mask = (media.country_code == "AT") & (media.advertising_channel == "TV")
    media.loc[mask & media.date.between("2024-03-26", "2024-04-07"), "media_investment"] = 0
    assert assessment(media, sales)["checks"]["treatment"]["status"] == "fail"
