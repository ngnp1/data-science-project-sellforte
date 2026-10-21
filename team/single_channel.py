from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Event:
    """A detected informative period. The start and end dates are inclusive."""

    sid: str
    country_code: str | None
    channel: str | None
    event_type: str
    start: pd.Timestamp
    end: pd.Timestamp
    pattern_id: str | None = None
    multiplier: float | None = None
    tags: tuple[str, ...] = ()
    components: tuple[tuple[pd.Timestamp, pd.Timestamp], ...] = ()
    detection_confidence: float | None = None
    informativeness: float | None = None

    @property
    def n_days(self) -> int:
        """Returns the inclusive duration of the event in days."""
        return int((self.end - self.start).days) + 1


def prepare_daily_spend_panel(media_df: pd.DataFrame | None) -> pd.DataFrame:
    """Aggregate campaign rows into daily spend per (country, channel)."""
    if media_df is None or media_df.empty:
        return pd.DataFrame()

    daily = (
        media_df.assign(date=pd.to_datetime(media_df["date"]))
        .groupby(["date", "country_code", "advertising_channel"])["media_investment"]
        .sum()
        .unstack(level=["country_code", "advertising_channel"], fill_value=0.0)
    )
    if daily.empty:
        return daily

    dates = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(dates, fill_value=0.0)
    daily.index.name = "date"
    return daily


def _active_channel_mask(
    daily_panel: pd.DataFrame,
    country: str,
    channels: list[str],
    off_threshold: float,
) -> pd.DataFrame:
    """Classifies daily spend as active using each channel's positive median."""
    market_spend = daily_panel.loc[:, [(country, channel) for channel in channels]]
    active = pd.DataFrame(index=daily_panel.index)
    for channel in channels:
        spend = market_spend[(country, channel)]
        positive_spend = spend[spend > 0]
        threshold = (
            off_threshold * float(positive_spend.median())
            if not positive_spend.empty
            else 0.0
        )
        active[channel] = spend > threshold
    return active


def detect_single_channel_periods(
    daily_panel: pd.DataFrame,
    sid: str = "",
    min_days: int = 7,
    off_threshold: float = 0.15,
) -> list[Event]:
    """Detect bounded windows where exactly one channel remains active."""
    if daily_panel.empty or min_days < 1:
        return []

    events: list[Event] = []
    countries = sorted({country for country, _ in daily_panel.columns})
    dates = daily_panel.index

    for country in countries:
        channels = sorted(
            channel for market, channel in daily_panel.columns if market == country
        )
        if not channels:
            continue

        active = _active_channel_mask(daily_panel, country, channels, off_threshold)
        active_sets = [
            frozenset(channel for channel in channels if active.at[date, channel])
            for date in dates
        ]

        window_start = 0
        for index in range(1, len(dates) + 1):
            if index < len(dates) and active_sets[index] == active_sets[window_start]:
                continue

            window_end = index - 1
            window_days = window_end - window_start + 1
            active_set = active_sets[window_start]
            if window_days >= min_days and len(active_set) == 1:
                paused_channels = set(channels) - active_set
                prior = active.iloc[:window_start]
                following = active.iloc[index:]
                bounded = all(
                    prior[channel].any() and following[channel].any()
                    for channel in paused_channels
                )
                likely_single = len(channels) >= 3 or (
                    len(channels) == 2 and window_days in {14, 30, 42}
                )
                if bounded and likely_single:
                    events.append(
                        Event(
                            sid=sid,
                            country_code=country,
                            channel=next(iter(active_set)),
                            event_type="single_channel",
                            start=dates[window_start],
                            end=dates[window_end],
                        )
                    )
            window_start = index

    return events


def detect_informative_periods(
    media_df: pd.DataFrame | None,
    sales_df: pd.DataFrame | None = None,
    sid: str = "",
) -> list[Event]:
    """Detect bounded single-channel periods from daily media spend."""
    del sales_df
    panel = prepare_daily_spend_panel(media_df)
    return detect_single_channel_periods(panel, sid=sid)


def run_detection(
    media_df: pd.DataFrame | None,
    sales_df: pd.DataFrame | None = None,
    sid: str = "",
) -> list[Event]:
    """Standard detector entry point used by the evaluation harness."""
    return detect_informative_periods(media_df, sales_df, sid)
