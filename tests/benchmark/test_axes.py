import numpy as np
import pytest

from benchmark.spec import axes


def rng(seed=0):
    return np.random.default_rng(seed)


def test_pools_are_large_enough_for_the_widest_scenarios():
    assert len(axes.COUNTRY_POOL) >= 8
    assert len(axes.CHANNEL_POOL) >= 12


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
def test_pick_countries_returns_n_distinct_markets(n):
    got = axes.pick_countries(rng(), n, "moderate")
    assert len(got) == n
    assert len({c["code"] for c in got}) == n
    assert all(c["market_size"] > 0 for c in got)


def test_extreme_spread_really_is_fifteen_fold():
    got = axes.pick_countries(rng(), 5, "extreme")
    sizes = [c["market_size"] for c in got]
    assert max(sizes) / min(sizes) == pytest.approx(15.0, rel=1e-6)


def test_tight_spread_keeps_markets_comparable():
    got = axes.pick_countries(rng(), 5, "tight")
    sizes = [c["market_size"] for c in got]
    assert max(sizes) / min(sizes) < 1.3


@pytest.mark.parametrize("n", [1, 2, 4, 6, 9, 12])
def test_pick_channels_produces_a_config_the_generator_accepts(n):
    got = axes.pick_channels(rng(), n)
    assert len(got) == n
    assert len({c["name"] for c in got}) == n

    # Every channel except the last needs min/max; the last takes the remainder.
    for ch in got[:-1]:
        assert 0 < ch["spend_share_min"] < ch["spend_share_max"] < 1
    assert "spend_share_min" not in got[-1]

    # The generator's contract: the maxima must leave room for the last channel.
    assert sum(c["spend_share_max"] for c in got[:-1]) < 0.92

    required = {"name", "type", "platform", "true_cvr", "decay",
                "alpha_saturation", "gamma_saturation", "assumed_ctr"}
    for ch in got:
        assert required <= set(ch)
        assert ("true_cpm" in ch) == (ch["type"] == "impression")
        assert ("true_cpc" in ch) == (ch["type"] == "click")


def test_pick_channels_is_deterministic_for_a_given_seed():
    assert axes.pick_channels(rng(5), 6) == axes.pick_channels(rng(5), 6)


@pytest.mark.parametrize("n", [2, 4, 6, 9, 12])
def test_pick_channels_orders_impressions_before_clicks(n):
    """generate_with_simmmulator.R regroups channels into impressions-then-
    clicks before deciding which one is exempt from spend shares -- so the
    list we hand it must already be in that order, and the exemption must
    land on the true last element of that order, not of the random draw."""
    got = axes.pick_channels(rng(3), n)
    types = [c["type"] for c in got]
    assert types == sorted(types, key=lambda t: t != "impression"), \
        "channels are not ordered impressions-then-clicks"

    for ch in got[:-1]:
        assert "spend_share_min" in ch and "spend_share_max" in ch, \
            f"{ch['name']} is missing spend shares but is not the last channel"
    assert "spend_share_min" not in got[-1] and "spend_share_max" not in got[-1], \
        f"the last channel ({got[-1]['name']}) should have no spend shares"


def test_noise_presets_increase_monotonically():
    lo, med, hi = (axes.NOISE_PRESETS[k] for k in ["low", "med", "high"])
    assert lo["error_std"] < med["error_std"] < hi["error_std"]
    assert lo["cvr_mult"] < med["cvr_mult"] < hi["cvr_mult"]


def test_baseline_for_threads_trend_and_seasonality_through():
    got = axes.baseline_for("high", trend_p=1.0, temp_var=5)
    assert got["trend_p"] == 1.0
    assert got["temp_var"] == 5
    assert got["error_std"] == axes.NOISE_PRESETS["high"]["error_std"]
