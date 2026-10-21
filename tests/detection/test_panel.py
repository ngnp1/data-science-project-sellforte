import numpy as np
import pandas as pd

from detection.io.panel import build_panel


def media(rows):
    return pd.DataFrame(rows, columns=[
        "date", "ad_platform", "advertising_channel", "campaign_name",
        "campaign_id", "media_investment", "clicks", "impressions",
        "conversions", "conversion_value", "country_code",
    ]).astype({"date": "datetime64[ns]"})


def row(date, country, channel, spend, clicks=1.0, impr=10.0, campaign="c1"):
    return [pd.Timestamp(date), "P", channel, campaign, 1, spend, clicks,
            impr, 0, 0.0, country]


def test_campaigns_are_aggregated_to_channel_level():
    """Synthetic data has one campaign per channel-day, but the real Sellforte
    export is campaign-grained and spec section 11 calls this the top real-data
    risk: a campaign ending is NOT a channel holdout."""
    m = media([
        row("2024-01-01", "DE", "TV", 100.0, campaign="a"),
        row("2024-01-01", "DE", "TV", 150.0, campaign="b"),
    ])
    p = build_panel(m)
    assert p.series("DE", "TV").loc[pd.Timestamp("2024-01-01")] == 250.0


def test_a_campaign_ending_is_not_a_channel_holdout():
    """Two campaigns; one stops. The channel keeps spending, so the panel must
    show continuous spend rather than a gap."""
    rows = []
    for d in pd.date_range("2024-01-01", periods=4):
        rows.append(row(d, "DE", "TV", 100.0, campaign="a"))
        if d < pd.Timestamp("2024-01-03"):
            rows.append(row(d, "DE", "TV", 50.0, campaign="b"))
    p = build_panel(media(rows))
    assert list(p.series("DE", "TV")) == [150.0, 150.0, 100.0, 100.0]
    assert p.present_mask("DE", "TV").all()


def test_missing_days_are_reindexed_and_flagged_absent():
    """The single biggest real-data failure mode: a missing row and a zero-spend
    row look identical once reindexed, unless presence is tracked separately."""
    m = media([
        row("2024-01-01", "DE", "TV", 100.0),
        row("2024-01-04", "DE", "TV", 120.0),
    ])
    p = build_panel(m)
    s = p.series("DE", "TV")
    assert len(s) == 4
    assert s.loc[pd.Timestamp("2024-01-02")] == 0.0
    present = p.present_mask("DE", "TV")
    assert present.loc[pd.Timestamp("2024-01-01")]
    assert not present.loc[pd.Timestamp("2024-01-02")]


def test_an_explicit_zero_is_present_but_a_missing_row_is_not():
    m = media([
        row("2024-01-01", "DE", "TV", 0.0),
        row("2024-01-03", "DE", "TV", 5.0),
    ])
    p = build_panel(m)
    present = p.present_mask("DE", "TV")
    assert present.loc[pd.Timestamp("2024-01-01")]      # explicit zero
    assert not present.loc[pd.Timestamp("2024-01-02")]  # absent row


def test_the_date_grid_is_complete_and_shared_across_series():
    m = media([
        row("2024-01-01", "DE", "TV", 10.0),
        row("2024-01-05", "AT", "Radio", 20.0),
    ])
    p = build_panel(m)
    assert len(p.dates) == 5
    for c, ch in [("DE", "TV"), ("AT", "Radio")]:
        assert p.series(c, ch).index.equals(p.dates)


def test_channels_in_lists_only_that_markets_channels():
    m = media([
        row("2024-01-01", "DE", "TV", 10.0),
        row("2024-01-01", "AT", "Radio", 20.0),
    ])
    p = build_panel(m)
    assert p.countries == ["AT", "DE"]
    assert p.channels_in("DE") == ["TV"]
    assert p.channels_in("AT") == ["Radio"]


def test_a_channel_absent_from_a_market_reads_as_all_zero_not_missing_key():
    """Cross-market comparison asks about channels a market may not run."""
    m = media([
        row("2024-01-01", "DE", "TV", 10.0),
        row("2024-01-01", "AT", "Radio", 20.0),
    ])
    p = build_panel(m)
    s = p.series("AT", "TV")
    assert (s == 0.0).all()
    assert not p.present_mask("AT", "TV").any()


def test_sales_are_summed_to_one_daily_series_per_country():
    m = media([row("2024-01-01", "DE", "TV", 10.0)])
    s = pd.DataFrame({
        "country": ["Germany"] * 4,
        "turnover": [10.0, 20.0, 30.0, 40.0],
        "customer_type": ["New", "New", "Returning", "Returning"],
        "country_code": ["DE"] * 4,
        "granted_discounts": [0] * 4,
        "sales_channel": ["Ecom", "Stores", "Ecom", "Stores"],
        "date": [pd.Timestamp("2024-01-01")] * 4,
    })
    p = build_panel(m, s)
    assert p.sales.loc[pd.Timestamp("2024-01-01"), "DE"] == 100.0


def test_sales_are_optional():
    p = build_panel(media([row("2024-01-01", "DE", "TV", 10.0)]))
    assert p.sales.empty or p.sales.shape[1] == 0


def test_impressions_and_clicks_are_carried_for_corroboration():
    """Spend zero AND impressions zero is a real pause; spend zero with
    impressions still flowing is a billing or tracking artifact."""
    m = media([row("2024-01-01", "DE", "TV", 0.0, clicks=5.0, impr=500.0)])
    p = build_panel(m)
    assert p.impressions.loc[pd.Timestamp("2024-01-01"), ("DE", "TV")] == 500.0
    assert p.clicks.loc[pd.Timestamp("2024-01-01"), ("DE", "TV")] == 5.0
