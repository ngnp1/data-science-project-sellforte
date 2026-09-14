import numpy as np
import pandas as pd

from detection.io.normalize import (active_level, channel_share,
                                    cross_market_series, market_scale,
                                    scale_free)
from detection.io.panel import build_panel


def s(values, start="2024-01-01"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values)),
                     dtype=float)


def media(rows):
    return pd.DataFrame(rows, columns=[
        "date", "ad_platform", "advertising_channel", "campaign_name",
        "campaign_id", "media_investment", "clicks", "impressions",
        "conversions", "conversion_value", "country_code",
    ]).astype({"date": "datetime64[ns]"})


def row(date, country, channel, spend):
    return [pd.Timestamp(date), "P", channel, "c", 1, spend, 1.0, 10.0, 0,
            0.0, country]


def test_active_level_ignores_off_days():
    """The level must describe what the channel spends when it is RUNNING,
    otherwise a long holdout drags the baseline down and hides itself."""
    assert active_level(s([100.0, 100.0, 0.0, 0.0, 100.0])) == 100.0


def test_active_level_is_nan_when_the_channel_never_ran():
    assert np.isnan(active_level(s([0.0, 0.0, 0.0])))


def test_active_level_is_robust_to_a_spike():
    assert active_level(s([100.0, 100.0, 100.0, 100.0, 9999.0])) == 100.0


def test_scale_free_makes_two_markets_of_different_size_identical():
    """A Finnish holdout and a US holdout must look the same to the detector.
    This is the whole reason detection runs on a log ratio."""
    small = s([100.0, 100.0, 0.0, 0.0, 100.0])
    big = s([13000.0, 13000.0, 0.0, 0.0, 13000.0])
    assert np.allclose(scale_free(small).values, scale_free(big).values)


def test_scale_free_of_zero_is_zero():
    assert scale_free(s([100.0, 0.0])).iloc[1] == 0.0


def test_channel_share_sums_to_one_on_days_with_spend():
    m = media([
        row("2024-01-01", "DE", "TV", 75.0),
        row("2024-01-01", "DE", "Radio", 25.0),
    ])
    share = channel_share(build_panel(m), "DE")
    assert abs(share.loc[pd.Timestamp("2024-01-01")].sum() - 1.0) < 1e-9
    assert abs(share.loc[pd.Timestamp("2024-01-01"), "TV"] - 0.75) < 1e-9


def test_channel_share_is_zero_not_nan_on_a_dark_day():
    """A dark day has no total to divide by. It must not poison the series
    with NaN, because composition logic reads it downstream."""
    m = media([
        row("2024-01-01", "DE", "TV", 0.0),
        row("2024-01-01", "DE", "Radio", 0.0),
    ])
    share = channel_share(build_panel(m), "DE")
    assert (share.loc[pd.Timestamp("2024-01-01")] == 0.0).all()


def test_market_scale_is_the_long_run_median_of_total_spend():
    """The daily totals vary and are deliberately skewed, so median, mean, min
    and max all differ: a constant series cannot tell them apart, and the name
    of this test claims specifically the median.

    Daily totals are 100, 100, 100, 100, 1000 -- median 100, mean 280. A spike
    must not drag the market's scale with it, which is the whole reason this is
    a median."""
    totals = [100.0, 100.0, 100.0, 100.0, 1000.0]
    rows = []
    for d, total in zip(pd.date_range("2024-01-01", periods=5), totals):
        rows.append(row(d, "DE", "TV", total * 0.6))
        rows.append(row(d, "DE", "Radio", total * 0.4))
    scale = market_scale(build_panel(media(rows)), "DE")
    assert scale == 100.0, f"expected the median, got {scale}"


def test_market_scale_ignores_days_the_market_was_dark():
    """The positive-day filter, which a constant fixture leaves unexercised."""
    totals = [200.0, 200.0, 0.0, 0.0, 200.0]
    rows = []
    for d, total in zip(pd.date_range("2024-01-01", periods=5), totals):
        rows.append(row(d, "DE", "TV", total))
    assert market_scale(build_panel(media(rows)), "DE") == 200.0


def test_cross_market_series_divides_out_market_size():
    """Two markets 10x apart in size, running the same relative pattern, must
    produce the same cross-market series -- otherwise the large market dominates
    every comparison, which spec section 5 exists to prevent."""
    shape = [1.0, 0.5, 2.0, 0.25, 1.0]   # varies, so equality is not trivial
    rows = []
    for d, k in zip(pd.date_range("2024-01-01", periods=5), shape):
        rows.append(row(d, "US", "TV", 1000.0 * k))
        rows.append(row(d, "FI", "TV", 100.0 * k))
    p = build_panel(media(rows))
    assert np.allclose(cross_market_series(p, "US", "TV").values,
                       cross_market_series(p, "FI", "TV").values)


def test_market_scale_of_a_silent_market_does_not_divide_by_zero():
    m = media([row("2024-01-01", "DE", "TV", 0.0)])
    p = build_panel(m)
    assert market_scale(p, "DE") == 0.0
    assert not cross_market_series(p, "DE", "TV").isna().any()
