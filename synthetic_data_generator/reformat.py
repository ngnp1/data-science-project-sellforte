"""
Reshape siMMMulator's raw multi-country wide output (raw_daily_wide.csv, produced by
generate_with_simmmulator.R) into the media.csv / sales.csv schema used by the
Sellforte samples, plus a ground_truth.csv answer key and true_roi.csv.

All configuration (countries, channels, customer types, sales channels) comes
from config.yaml, the same file generate_with_simmmulator.R reads -- one
source of truth for both languages.

Usage:
    python reformat.py
"""

import hashlib

import numpy as np
import pandas as pd
import yaml

RNG = np.random.default_rng(42)


def r_colname(channel):
    """siMMMulator keeps the literal channel name (spaces included) in column names."""
    return channel


def stable_campaign_id(platform, channel, country_code):
    h = hashlib.md5(f"{platform}_{channel}_{country_code}".encode()).hexdigest()
    return int(h[:9], 16) % 900_000_000 + 100_000_000


def build_media_df(raw, channels, revenue_per_conv):
    rows = []
    for ch in channels:
        channel, platform, ch_type = ch["name"], ch["platform"], ch["type"]
        assumed_ctr = ch["assumed_ctr"]
        col = r_colname(channel)

        spend = raw[f"spend_{col}"].values
        conv = raw[f"conv_{col}"].values.round().astype(int)
        conv = np.clip(conv, 0, None)

        if ch_type == "impression":
            impressions = raw[f"impressions_{col}"].values
            clicks = impressions * assumed_ctr * RNG.lognormal(0, 0.05, len(raw))
        else:
            clicks = raw[f"clicks_{col}"].values
            impressions = clicks / assumed_ctr * RNG.lognormal(0, 0.05, len(raw))

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


def build_sales_df(raw, country_code, country_name, customer_types, sales_channels):
    n = len(raw)
    combos = [(ct, sc) for ct in customer_types for sc in sales_channels]
    splits = RNG.dirichlet(alpha=[3, 2, 2, 1][: len(combos)], size=n)

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
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    with open("events_config.yaml") as f:
        events = pd.DataFrame(yaml.safe_load(f))

    raw = pd.read_csv("raw_daily_wide.csv", parse_dates=["DATE"])

    country_names = {c["code"]: c["name"] for c in config["countries"]}
    revenue_per_conv = config["revenue_per_conv"]

    media_parts, sales_parts = [], []
    dates_by_country = {}
    for country_code, group in raw.groupby("country_code"):
        group = group.sort_values("DATE").reset_index(drop=True)
        dates_by_country[country_code] = group["DATE"]
        media_parts.append(build_media_df(group, config["channels"], revenue_per_conv))
        sales_parts.append(build_sales_df(
            group, country_code, country_names[country_code],
            config["customer_types"], config["sales_channels"],
        ))

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
