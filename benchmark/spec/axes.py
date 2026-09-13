"""Pools and presets that benchmark scenarios are drawn from.

Nothing here knows what an event is or how many scenarios exist. This module
answers only: what markets, what channels, how noisy, how spread out.
"""
from __future__ import annotations

import numpy as np

COUNTRY_POOL: list[tuple[str, str]] = [
    ("DE", "Germany"), ("AT", "Austria"), ("CH", "Switzerland"),
    ("US", "United States"), ("FI", "Finland"), ("SE", "Sweden"),
    ("NO", "Norway"), ("NL", "Netherlands"), ("FR", "France"),
    ("PL", "Poland"),
]

# Twelve channel templates spanning both siMMMulator types. Decay rates are
# chosen to span the realistic range: TV lingers, Search does not. Templates
# carry no spend shares -- pick_channels assigns those, because valid shares
# depend on how many channels a scenario ends up with.
CHANNEL_POOL: list[dict] = [
    dict(name="TV", type="impression", platform="TV", true_cvr=0.00003,
         true_cpm=5.0, mean_noisy_cpm_cpc=0.0, std_noisy_cpm_cpc=0.3,
         mean_noisy_cvr=0.0, std_noisy_cvr=0.00001, decay=0.55,
         alpha_saturation=2.0, gamma_saturation=0.4, assumed_ctr=0.001),
    dict(name="Radio", type="impression", platform="Radio", true_cvr=0.00002,
         true_cpm=3.0, mean_noisy_cpm_cpc=0.0, std_noisy_cpm_cpc=0.2,
         mean_noisy_cvr=0.0, std_noisy_cvr=0.000008, decay=0.35,
         alpha_saturation=2.0, gamma_saturation=0.3, assumed_ctr=0.0008),
    dict(name="OOH", type="impression", platform="OOH", true_cvr=0.000015,
         true_cpm=4.0, mean_noisy_cpm_cpc=0.0, std_noisy_cpm_cpc=0.25,
         mean_noisy_cvr=0.0, std_noisy_cvr=0.000006, decay=0.45,
         alpha_saturation=2.0, gamma_saturation=0.35, assumed_ctr=0.0005),
    dict(name="Print", type="impression", platform="Print", true_cvr=0.000012,
         true_cpm=6.0, mean_noisy_cpm_cpc=0.0, std_noisy_cpm_cpc=0.3,
         mean_noisy_cvr=0.0, std_noisy_cvr=0.000005, decay=0.40,
         alpha_saturation=2.0, gamma_saturation=0.35, assumed_ctr=0.0004),
    dict(name="Cinema", type="impression", platform="Cinema", true_cvr=0.00002,
         true_cpm=9.0, mean_noisy_cpm_cpc=0.0, std_noisy_cpm_cpc=0.4,
         mean_noisy_cvr=0.0, std_noisy_cvr=0.000008, decay=0.50,
         alpha_saturation=2.0, gamma_saturation=0.4, assumed_ctr=0.0003),
    dict(name="Google Discovery", type="impression", platform="Google Ads",
         true_cvr=0.00006, true_cpm=8.0, mean_noisy_cpm_cpc=0.0,
         std_noisy_cpm_cpc=0.5, mean_noisy_cvr=0.0, std_noisy_cvr=0.00002,
         decay=0.20, alpha_saturation=2.0, gamma_saturation=0.3,
         assumed_ctr=0.015),
    dict(name="Facebook", type="impression", platform="Meta", true_cvr=0.00005,
         true_cpm=10.0, mean_noisy_cpm_cpc=0.0, std_noisy_cpm_cpc=0.5,
         mean_noisy_cvr=0.0, std_noisy_cvr=0.000015, decay=0.30,
         alpha_saturation=2.0, gamma_saturation=0.3, assumed_ctr=0.012),
    dict(name="Instagram", type="impression", platform="Meta", true_cvr=0.00004,
         true_cpm=12.0, mean_noisy_cpm_cpc=0.0, std_noisy_cpm_cpc=0.6,
         mean_noisy_cvr=0.0, std_noisy_cvr=0.000012, decay=0.25,
         alpha_saturation=2.0, gamma_saturation=0.3, assumed_ctr=0.010),
    dict(name="TikTok", type="impression", platform="TikTok", true_cvr=0.000035,
         true_cpm=7.0, mean_noisy_cpm_cpc=0.0, std_noisy_cpm_cpc=0.7,
         mean_noisy_cvr=0.0, std_noisy_cvr=0.000014, decay=0.18,
         alpha_saturation=2.0, gamma_saturation=0.3, assumed_ctr=0.009),
    dict(name="YouTube", type="impression", platform="Google Ads",
         true_cvr=0.000025, true_cpm=11.0, mean_noisy_cpm_cpc=0.0,
         std_noisy_cpm_cpc=0.5, mean_noisy_cvr=0.0, std_noisy_cvr=0.00001,
         decay=0.35, alpha_saturation=2.0, gamma_saturation=0.35,
         assumed_ctr=0.004),
    dict(name="Google Search", type="click", platform="Google Ads",
         true_cvr=0.02, true_cpc=0.8, mean_noisy_cpm_cpc=0.0,
         std_noisy_cpm_cpc=0.05, mean_noisy_cvr=0.0, std_noisy_cvr=0.005,
         decay=0.10, alpha_saturation=2.0, gamma_saturation=0.2,
         assumed_ctr=0.05),
    dict(name="Affiliate", type="click", platform="Affiliate", true_cvr=0.03,
         true_cpc=0.5, mean_noisy_cpm_cpc=0.0, std_noisy_cpm_cpc=0.04,
         mean_noisy_cvr=0.0, std_noisy_cvr=0.006, decay=0.12,
         alpha_saturation=2.0, gamma_saturation=0.2, assumed_ctr=0.08),
]

# error_std and temp_coef_sd feed step_1_create_baseline; the multipliers scale
# every channel's std_noisy_cpm_cpc and std_noisy_cvr.
NOISE_PRESETS: dict[str, dict] = {
    "low":  dict(error_std=50.0,  temp_coef_sd=250.0,  cpm_cpc_mult=0.5, cvr_mult=0.5),
    "med":  dict(error_std=100.0, temp_coef_sd=500.0,  cpm_cpc_mult=1.0, cvr_mult=1.0),
    "high": dict(error_std=300.0, temp_coef_sd=1500.0, cpm_cpc_mult=2.0, cvr_mult=2.5),
}

# (smallest market_size, largest market_size)
MARKET_SPREAD_PRESETS: dict[str, tuple[float, float]] = {
    "tight":    (0.90, 1.10),
    "moderate": (0.20, 1.30),
    "extreme":  (0.10, 1.50),
}

TREND_LEVELS: tuple[float, ...] = (0.0, 0.5, 1.0)
SEASONALITY_LEVELS: tuple[float, ...] = (0.5, 2.0, 5.0)
NOISE_LEVELS: tuple[str, ...] = ("low", "med", "high")


def pick_countries(rng: np.random.Generator, n: int, spread: str) -> list[dict]:
    """Choose n distinct markets and assign market sizes spanning `spread`.

    The extreme and tight endpoints are pinned to the preset bounds rather than
    sampled, so a scenario labelled "extreme" is guaranteed to exercise the full
    ratio instead of happening to draw two similar sizes.
    """
    lo, hi = MARKET_SPREAD_PRESETS[spread]
    idx = rng.choice(len(COUNTRY_POOL), size=n, replace=False)
    chosen = [COUNTRY_POOL[i] for i in idx]

    if n == 1:
        sizes = [1.0]
    else:
        # Log-spaced between the bounds, then shuffled so market size is not
        # correlated with position in the list.
        sizes = list(np.exp(np.linspace(np.log(lo), np.log(hi), n)))
        rng.shuffle(sizes)

    return [{"code": c, "name": name, "market_size": round(float(s), 4)}
            for (c, name), s in zip(chosen, sizes)]


def pick_channels(rng: np.random.Generator, n: int) -> list[dict]:
    """Choose n distinct channels and give them valid spend shares.

    The generator requires spend_share_min/max on every channel except the last,
    which receives whatever budget remains. The maxima must therefore leave
    headroom, or the last channel's share can go negative.

    Step 5 probe: a one-channel config (no spend_share_min/max at all, since
    the sole channel is also the "last" channel) was submitted to
    generate_with_simmmulator.R. siMMMulator rejected it: step_2_ads_spend
    errors with "You did not enter a number in max_min_proportion_on_each_channel.
    Must enter a numeric." because MAX_MIN_PROPORTION is built from all
    channels except the last, which is empty when n == 1. So n == 1 is
    unsupported by the simulator; this function still returns a well-formed
    single-channel list for callers that need it (e.g. property tests), but no
    scenario generation path should hand it a market with only one channel.
    """
    idx = rng.choice(len(CHANNEL_POOL), size=n, replace=False)
    chosen = [dict(CHANNEL_POOL[i]) for i in idx]

    if n == 1:
        return chosen

    # Dirichlet gives a random but sane budget split. A plain Dirichlet(1,...)
    # draw can hand a channel a near-zero share, which would make
    # spend_share_max < spend_share_min once a floor is applied -- so every
    # component is first floored to at least 0.15/n of the total (mixing in a
    # uniform share while keeping the weights summing to 1), *then* scaled so
    # the first n-1 channels claim at most 80% of the budget. That keeps every
    # min/max pair ordered and the sum of maxima safely under the 0.92
    # headroom limit for every n up to 12 (verified by simulation, worst case
    # ~0.91).
    raw = rng.dirichlet(np.ones(n))
    floor = 0.15 / n
    raw = raw * (1 - n * floor) + floor
    weights = raw * 0.80

    for ch, w in zip(chosen[:-1], weights[:-1]):
        mid = float(w)
        ch["spend_share_min"] = round(mid * 0.85, 4)
        ch["spend_share_max"] = round(mid * 1.15, 4)

    return chosen


def apply_noise(channels: list[dict], noise_level: str) -> list[dict]:
    """Scale per-channel noise to the scenario's noise level."""
    preset = NOISE_PRESETS[noise_level]
    out = []
    for ch in channels:
        ch = dict(ch)
        ch["std_noisy_cpm_cpc"] = round(ch["std_noisy_cpm_cpc"] * preset["cpm_cpc_mult"], 8)
        ch["std_noisy_cvr"] = round(ch["std_noisy_cvr"] * preset["cvr_mult"], 10)
        out.append(ch)
    return out


def baseline_for(noise_level: str, trend_p: float, temp_var: float) -> dict:
    """Build the config.yaml `baseline` block for a scenario."""
    preset = NOISE_PRESETS[noise_level]
    return dict(
        daily_mean=15000.0,
        trend_p=float(trend_p),
        temp_var=float(temp_var),
        temp_coef_mean=100.0,
        temp_coef_sd=preset["temp_coef_sd"],
        error_std=preset["error_std"],
    )
