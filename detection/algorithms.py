from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=True)
class Event:
    """Represents a detected informative period. Periods are inclusive."""

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

    @property
    def n_days(self) -> int:
        return int((self.end - self.start).days) + 1


@dataclass(frozen=True)
class PauseWindow:
    start: pd.Timestamp
    end: pd.Timestamp
    n_days: int
    touches_start: bool
    touches_end: bool


def prepare_daily_spend_panel(media_df: pd.DataFrame) -> pd.DataFrame:
    """Function to aggregate campaign rows into daily spend series per (market, channel)."""
    if media_df is None or media_df.empty:
        return pd.DataFrame()
    df = media_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    daily = (
        df.groupby(["date", "country_code", "advertising_channel"])["media_investment"]
        .sum()
        .unstack(level=["country_code", "advertising_channel"], fill_value=0.0)
    )
    if daily.empty:
        return daily
    daily = daily.reindex(pd.date_range(daily.index.min(), daily.index.max(), freq="D"), fill_value=0.0)
    daily.index.name = "date"
    return daily[[c for c in daily.columns if daily[c].sum() > 0]]


def find_channel_pauses(
    series: pd.Series,
    normal_spend: float,
    min_days: int = 7,
    off_threshold: float = 0.15,
) -> list[PauseWindow]:
    """Function to find off-periods for a single channel's daily spend series.

    A day is considered an off-day when its spend is at or below `off_threshold`. 
    Consecutive off-days are considered a pause. Only pauses lasting at least `min_days` are kept.

    Arguments:
        series:        Daily spend for one (country, channel) pair.
        normal_spend:  The channel's median spend
        min_days:      Minimum length of a pause to be reported.
        off_threshold: Fraction of normal_spend below which a day is counted as an off-day.

    Returns:
        List of PauseWindow objects in chronological order.
    """
    # Return early if baseline spend is invalid or data is missing
    if normal_spend <= 0 or series.empty:
        return []

    spend_cutoff = off_threshold * normal_spend
    is_off = (series <= spend_cutoff).tolist()
    dates = series.index
    n = len(dates)

    pauses = []
    in_pause = False
    pause_start_idx = 0

    for i, day_is_off in enumerate(is_off):
        # Start of a pause
        if day_is_off and not in_pause:
            in_pause = True
            pause_start_idx = i

        # End of a pause
        elif not day_is_off and in_pause:
            in_pause = False
            pause_length = i - pause_start_idx
            
            # Record pause if long enough
            if pause_length >= min_days:
                pauses.append(
                    PauseWindow(
                        start=dates[pause_start_idx],
                        end=dates[i - 1],  # Last off-day
                        n_days=pause_length,
                        touches_start=(pause_start_idx == 0),
                        touches_end=False,
                    )
                )

    # Handle trailing pause (if dataset ends while channel is still off)
    if in_pause:
        pause_length = n - pause_start_idx
        if pause_length >= min_days:
            pauses.append(
                PauseWindow(
                    start=dates[pause_start_idx],
                    end=dates[n - 1],  # Last day of dataset
                    n_days=pause_length,
                    touches_start=(pause_start_idx == 0),
                    touches_end=True,
                )
            )

    return pauses

def detect_single_channel_periods(
    daily_panel: pd.DataFrame,
    sid: str = "",
    min_days: int = 7,
    off_threshold: float = 0.15,
) -> list[Event]:
    """Function to detect single-channel periods.

    Arguments:
        daily_panel: DataFrame with daily spend data.
        sid: Scenario identifier.
        min_days: Minimum number of days for a single-channel period.
        off_threshold: Fraction of normal_spend below which a day is counted as an off-day.

    Returns:
        List of Event objects.
    """
    events = []
    countries = sorted(set(c for c, _ in daily_panel.columns))

    for country in countries:
        cols = [c for c in daily_panel.columns if c[0] == country]
        channels = [c[1] for c in cols]
        if len(channels) < 3:  # Skips markets with fewer than 3 channels
            continue

        df = daily_panel[cols]  # Includes just this country's channels

        # Identify the active status of each channel on each day
        active = pd.DataFrame(index=df.index)  # Same date index as the spend panel
        for channel in channels:
            pos = df[(country, channel)][df[(country, channel)] > 0]  # Days with positive spend for the channel
            threshold = off_threshold * float(pos.median()) if not pos.empty else 0.0
            active[channel] = df[(country, channel)] > threshold  # If channel is actively spending that day

        # Identify periods where the active-channel set is constant
        dates = df.index  # Daily date range for this country
        active_sets = [frozenset(ch for ch in channels if active.loc[d, ch]) for d in dates] # For each day, record which channels are active

        windows: list[tuple[pd.Timestamp, pd.Timestamp, frozenset[str]]] = [] # List of type (start, end, active_set)
        cur_set, cur_start = active_sets[0], dates[0]  # Initialise the first window with the first day
        for d, s in zip(dates[1:], active_sets[1:]):    # Iterate through the days
            if s != cur_set:                             # Active-channel set changed
                windows.append((cur_start, dates[dates.get_loc(d) - 1], cur_set))  # Close the previous window
                cur_start, cur_set = d, s               # Start a new window with the current day
        windows.append((cur_start, dates[-1], cur_set))  # Close the last window

        # Check each window 
        for start_date, end_date, active_set in windows:    
            if (end_date - start_date).days + 1 < min_days:  # Window shorter than minimum duration
                continue
            if len(active_set) != 1: # Only one active channel in this window
                continue

            paused = [ch for ch in channels if ch not in active_set]  # Channels that are off during this window
            if len(paused) < 2:  # At least two channels must be paused
                continue

            # Check that every paused channel was active before and after this window
            prior_active_days = active.loc[:start_date].iloc[:-1]   # All days strictly before the window
            post_active_days = active.loc[end_date:].iloc[1:]       # All days strictly after the window
            all_paused_had_activity = (
                not prior_active_days.empty and prior_active_days[channel].any()      # Channel was active before the window
                and not post_active_days.empty and post_active_days[channel].any()    # Channel was active after the window
                for channel in paused
            )
            if all_paused_had_activity:
                events.append(
                    Event(
                        sid=sid,
                        country_code=country,
                        channel=next(iter(active_set)),
                        event_type="single_channel",
                        start=start_date,
                        end=end_date,
                    )
                )

    return sorted(events, key=lambda e: (e.country_code or "", e.start))  # Return all detected events sorted by country and start date

def detect_informative_periods(
    media_df: pd.DataFrame | None,
    sales_df: pd.DataFrame | None = None,
    sid: str = ""
) -> list[Event]:

    if media_df is None or media_df.empty:
        return []
    panel = prepare_daily_spend_panel(media_df)
    if panel.empty:
        return []
    events = detect_single_channel_periods(panel, sid=sid)
    return sorted(events, key=lambda e: (e.country_code or "", e.start))


def run_detection(
    media_df: pd.DataFrame | None,
    sales_df: pd.DataFrame | None = None,
    sid: str = ""
) -> list[Event]:

    return detect_informative_periods(media_df, sales_df, sid)
