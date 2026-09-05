"""
Reshape siMMMulator's raw multi-country wide output (raw_daily_wide.csv, produced by
generate_with_simmmulator.R) into the media.csv / sales.csv schema used by the
Sellforte samples, plus a ground_truth.csv answer key and true_roi.csv.

Usage:
    python reformat.py
"""

import hashlib

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

COUNTRY_NAMES = {
    "DE": "Germany", "AT": "Austria", "CH": "Switzerland",
    "US": "United States", "FI": "Finland",
}
CUSTOMER_TYPES = ["New", "Returning"]
SALES_CHANNELS = ["Ecom", "Stores"]

# Purely cosmetic: siMMMulator only simulates impressions OR clicks per channel
# (never both). To match the sample schema (both columns present for every
# channel) we back out an implied secondary metric using an assumed CTR. This
# has no effect on spend, decay, saturation, or conversions -- display only.
ASSUMED_CTR = {
    "TV": 0.001, "Radio": 0.0008, "Google Discovery": 0.015,
    "Facebook": 0.012, "Instagram": 0.010, "Google Search": 0.05,
}


def r_colname(channel):
    """siMMMulator keeps the literal channel name (spaces included) in column names."""
    return channel


def stable_campaign_id(platform, channel, country_code):
    h = hashlib.md5(f"{platform}_{channel}_{country_code}".encode()).hexdigest()
    return int(h[:9], 16) % 900_000_000 + 100_000_000


def build_media_df(raw, channels_meta, revenue_per_conv):
    rows = []
    for _, ch in channels_meta.iterrows():
        channel, platform, ch_type = ch["channel"], ch["platform"], ch["type"]
        col = r_colname(channel)

        spend = raw[f"spend_{col}"].values
        conv = raw[f"conv_{col}"].values.round().astype(int)
        conv = np.clip(conv, 0, None)

        if ch_type == "impression":
            impressions = raw[f"impressions_{col}"].values
            clicks = impressions * ASSUMED_CTR[channel] * RNG.lognormal(0, 0.05, len(raw))
        else:
            clicks = raw[f"clicks_{col}"].values
            impressions = clicks / ASSUMED_CTR[channel] * RNG.lognormal(0, 0.05, len(raw))

        campaign_id = stable_campaign_id(platform, channel, raw["country_code"].iloc[0])
        campaign_name = f"{platform}_{channel}_{raw['country_code'].iloc[0]}"

        rows.append(pd.DataFrame({
            "date": raw["DATE"].values,
            "ad_platform": platform,
            "advertising_channel": channel,
            "campaign_name": campaign_name,
            "campaign_id": campaign_id,
            "media_investment": np.round(spend, 2),
            "clicks": np.round(clicks, 2),
            "impressions": np.round(impressions, 1),
            "conversions": conv,
            "conversion_value": np.round(conv * revenue_per_conv, 2),
            "country_code": raw["country_code"].iloc[0],
        }))
    return pd.concat(rows, ignore_index=True)


def build_sales_df(raw, country_code, country_name):
    n = len(raw)
    combos = [(ct, sc) for ct in CUSTOMER_TYPES for sc in SALES_CHANNELS]
    splits = RNG.dirichlet(alpha=[3, 2, 2, 1], size=n)

    rows = []
    for i in range(n):
        for (ct, sc), frac in zip(combos, splits[i]):
            rows.append({
                "country": country_name,
                "turnover": round(raw["total_revenue"].iloc[i] * frac, 2),
                "customer_type": ct,
                "country_code": country_code,
                "granted_discounts": 0,
                "sales_channel": sc,
                "date": raw["DATE"].iloc[i],
            })
    return pd.DataFrame(rows)


def build_ground_truth(events, dates_by_country):
    rows = []
    for _, ev in events.iterrows():
        dates = dates_by_country[ev["country"]]
        end_idx = min(ev["end_day"], len(dates) - 1)
        rows.append({
            "pattern_id": ev["pattern_id"],
            "pattern_type": ev["pattern_type"],
            "country_code": ev["country"],
            "channel": ev["channel"],
            "start_date": str(dates[ev["start_day"]].date()),
            "end_date": str(dates[end_idx].date()),
            "multiplier": ev["multiplier"] if ev["pattern_type"] == "step_change" else "",
            "description": ev["description"],
        })
    return pd.DataFrame(rows)


def main():
    raw = pd.read_csv("raw_daily_wide.csv", parse_dates=["DATE"])
    channels_meta = pd.read_csv("channels_meta.csv")
    run_meta = pd.read_csv("run_meta.csv")
    events = pd.read_csv("events_config.csv")
    revenue_per_conv = run_meta["revenue_per_conv"].iloc[0]

    media_parts, sales_parts = [], []
    dates_by_country = {}
    for country_code, group in raw.groupby("country_code"):
        group = group.sort_values("DATE").reset_index(drop=True)
        dates_by_country[country_code] = group["DATE"]
        media_parts.append(build_media_df(group, channels_meta, revenue_per_conv))
        sales_parts.append(build_sales_df(group, country_code, COUNTRY_NAMES[country_code]))

    media_df = pd.concat(media_parts, ignore_index=True)
    sales_df = pd.concat(sales_parts, ignore_index=True)
    ground_truth_df = build_ground_truth(events, dates_by_country)

    media_df.to_csv("media.csv", index=False)
    sales_df.to_csv("sales.csv", index=False)
    ground_truth_df.to_csv("ground_truth.csv", index=False)

    roi = media_df.groupby("advertising_channel").agg(
        true_spend=("media_investment", "sum"),
        true_revenue=("conversion_value", "sum"),
    )
    roi["true_roi"] = roi["true_revenue"] / roi["true_spend"]
    roi.reset_index().to_csv("true_roi.csv", index=False)

    print(f"Wrote media.csv ({len(media_df)} rows), sales.csv ({len(sales_df)} rows)")
    print(f"Wrote ground_truth.csv ({len(ground_truth_df)} events) and true_roi.csv")


if __name__ == "__main__":
    main()
