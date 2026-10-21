
from benchmark.eval.model import Event
import numpy as np
import pandas as pd


RHO = 0.15
MIN_DAYS = 7


def detect(media_df, sales_df, sid: str) -> list[Event]:
    if media_df is None or media_df.empty:
        return []

    df = media_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    all_dates = pd.date_range(
        df["date"].min(),
        df["date"].max(),
        freq="D"
    )

    events = []

    for country in sorted(df["country_code"].dropna().unique()):
        country_data = df[df["country_code"] == country]

        daily_spend = (
            country_data.groupby(["date", "advertising_channel"])[
                "media_investment"
            ]
            .sum(min_count=1)
            .unstack("advertising_channel")
            .reindex(all_dates)
            .fillna(0)
        )

        # Ignores channels that the market never used
        channels = [
            channel for channel in daily_spend.columns
            if (daily_spend[channel] > 0).any()
        ]

        if not channels:
            continue

        daily_spend = daily_spend[channels]

        # A channel is off when spend is at most 15% of its usual level
        normal_spend = daily_spend.where(daily_spend > 0).median()
        off = daily_spend.le(RHO * normal_spend, axis="columns")

        # A dark day has no active advertising channels
        dark = off.all(axis=1).to_numpy()

        changes = np.diff(
            np.concatenate(([False], dark, [False])).astype(int)
        )

        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0] - 1

        for start, end in zip(starts, ends):
            days = end - start + 1

            if days >= MIN_DAYS:
                events.append(
                    Event(
                        sid=sid,
                        country_code=country,
                        channel=None,
                        event_type="dark_period",
                        start=all_dates[start],
                        end=all_dates[end]
                    )
                )

    return sorted(
        events,
        key=lambda event: (event.country_code or "", event.start)
    )
