from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

# Pulse-train (flighting) grouping. A train is identified by REGULARITY --
# pauses of similar length at similar intervals -- never by length alone.
PULSE_MIN_PAUSES = 2
PULSE_LENGTH_TOLERANCE = 0.5   # fractional deviation from the train's first pause
PULSE_GAP_TOLERANCE = 0.6      # fractional deviation from the train's first gap

# Up to this many days of spend inside a pause is a stray booking, not a restart.
BLIP_DAYS = 2

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

def build_active_matrix(
    daily_panel: pd.DataFrame,
    country: str,
    channels: list[str],
    off_threshold: float = 0.15,
) -> pd.DataFrame:
    """Function to decide, for every day, which of a market's channels were running.

    "Normal spend" is the median over the days a channel ACTUALLY RAN, not the
    median over every day. A channel that is off for most of the period would
    otherwise have a median of zero, the cutoff would collapse to zero, and
    every day would read as active - so the longer a pause ran, the better it
    would hide itself.

    Arguments:
        daily_panel:   DataFrame with daily spend data.
        country:       Market to build the matrix for.
        channels:      Channels that market ran.
        off_threshold: Fraction of normal spend below which a day counts as off.

    Returns:
        Bool DataFrame, same date index as the panel, one column per channel.
    """
    active = pd.DataFrame(index=daily_panel.index)
    for channel in channels:
        series = daily_panel[(country, channel)]
        normal_spend = _normal_spend(series)
        active[channel] = series > off_threshold * normal_spend
    return active


def _normal_spend(series: pd.Series) -> float:
    """Median spend over the days the channel actually ran (0.0 if it never did)."""
    positive = series[series > 0]
    return float(positive.median()) if not positive.empty else 0.0


def _find_pauses_ignoring_blips(
    series: pd.Series,
    normal_spend: float,
    min_days: int = 7,
    off_threshold: float = 0.15,
    blip_days: int = BLIP_DAYS,
) -> list[PauseWindow]:
    """Function to find a channel's pauses, where a stray day or two of spend does not split one.

    Same pauses as `find_channel_pauses`, except that an on-stretch of at most
    `blip_days`, with off-days on both sides, counts as part of the pause.
    Without that, one stray booking in the middle of a holdout cuts it into two
    halves of similar length one day apart - exactly what a two-beat flighting
    schedule looks like - and the holdout is suppressed as a pulse train. The
    stray day is filled in BEFORE the min_days cut, so 5 off + 1 on + 5 off is
    one 11-day pause rather than two pauses too short to keep.

    Returns:
        List of PauseWindow objects in chronological order.
    """
    if normal_spend <= 0 or series.empty:
        return []
    off = (series <= off_threshold * normal_spend).to_numpy(bool).copy()

    if blip_days > 0 and off.any() and not off.all():
        edges = np.diff(np.concatenate([[1], off.astype(int), [1]]))
        for a, b in zip(np.flatnonzero(edges == -1), np.flatnonzero(edges == 1)):
            if 0 < a and b < len(off) and b - a <= blip_days:
                off[a:b] = True  # short on-stretch inside a pause

    edges = np.diff(np.concatenate([[0], off.astype(int), [0]]))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1
    dates, n = series.index, len(series)
    return [
        PauseWindow(
            start=dates[s],
            end=dates[e],
            n_days=int(e - s + 1),
            touches_start=bool(s == 0),
            touches_end=bool(e == n - 1),
        )
        for s, e in zip(starts, ends)
        if e - s + 1 >= min_days
    ]


def _market_continuity(
    daily_panel: pd.DataFrame,
    country: str,
    others: list[str],
    pause: PauseWindow,
) -> float:
    """Function to measure what the OTHER channels spent during a pause.

    Returns their mean daily spend inside the window as a fraction of their
    median daily spend outside it. Near 1.0 means the rest of the market
    carried on as usual.

    This is a second, independent signal to the channel COUNT used in
    `detect_natural_holdouts`, and the two fail differently. A market that
    halved every other budget has a count of zero - no other channel is off -
    but a ratio near 0.5, so only the ratio sees that the window is confounded.

    Returns:
        The ratio, or NaN when there are no other channels to measure.
    """
    if not others:
        return float("nan")

    total = daily_panel[[(country, ch) for ch in others]].sum(axis=1)
    inside = (total.index >= pause.start) & (total.index <= pause.end)
    outside_median = float(total[~inside].median()) if (~inside).any() else 0.0
    if outside_median <= 0:
        return float("nan")
    return float(total[inside].mean()) / outside_median


def _find_pulse_trains(pauses: list[PauseWindow]) -> list[list[int]]:
    """Function to group a channel's pauses into regular on/off trains (flighting).

    A train is identified by REGULARITY - pauses of similar length at similar
    intervals - and deliberately not by length alone. Television and radio are
    bought in flights, so a channel with routine gaps produces many pauses that
    are normal scheduling rather than events.

    Pauses touching either end of the data are excluded: a channel dormant at
    the start has not stopped, it has not begun.

    Returns:
        List of trains, each a list of indices into `pauses`.
    """
    interior = [i for i, p in enumerate(pauses)
                if not p.touches_start and not p.touches_end]
    if len(interior) < PULSE_MIN_PAUSES:
        return []

    trains: list[list[int]] = []
    current = [interior[0]]
    reference_length = pauses[interior[0]].n_days
    reference_gap: int | None = None

    for previous, index in zip(interior, interior[1:]):
        gap = (pauses[index].start - pauses[previous].end).days - 1
        similar_length = (abs(pauses[index].n_days - reference_length)
                          <= PULSE_LENGTH_TOLERANCE * reference_length)
        similar_gap = (reference_gap is None
                       or abs(gap - reference_gap) <= PULSE_GAP_TOLERANCE * reference_gap)

        if similar_length and similar_gap and index == previous + 1:
            current.append(index)
            reference_gap = gap if reference_gap is None else reference_gap
        else:
            if len(current) >= PULSE_MIN_PAUSES:
                trains.append(current)
            current = [index]
            reference_length = pauses[index].n_days
            reference_gap = None

    if len(current) >= PULSE_MIN_PAUSES:
        trains.append(current)
    return trains


def detect_natural_holdouts(
    daily_panel: pd.DataFrame,
    sid: str = "",
    min_days: int = 7,
    off_threshold: float = 0.15,
    other_off_share: float = 0.80,
    market_continuity_min: float = 0.60,
    emit_pulse_trains: bool = False,
) -> list[Event]:
    """Function to detect natural holdouts: one channel pauses, the market keeps running.

    A natural holdout behaves like an unintended holdout experiment, because the
    channels still spending tell you what sales would have done anyway.

    Finding the pause is the easy half. On the committed dataset there are 16
    windows where a channel sits at zero and only ONE is a holdout - the rest
    belong to the dark period, the single-channel period, the radio flighting
    and the US launch. So most of this function is ruling those out:

        0 other channels running  -> dark period        (not ours)
        1 other channel running   -> single-channel     (not ours)
        2+ others running         -> natural holdout    (ours)

    plus a spend check for a market that cut every other budget at the same
    time, and a before/after check so a channel that had not launched yet is
    not mistaken for one that stopped.

    Arguments:
        daily_panel:           DataFrame with daily spend data.
        sid:                   Scenario identifier.
        min_days:              Minimum length of a pause to be reported.
        off_threshold:         Fraction of normal spend below which a day is off.
        other_off_share:       Share of the window another channel must be off
                               across before it counts as also paused.
        market_continuity_min: Share of its normal spend the rest of the market
                               must keep for the window to be a clean holdout.
        emit_pulse_trains:     Whether to also emit regular flighting schedules
                               as one grouped `channel_pulse` event. Off by
                               default so the two detectors cannot both claim
                               the same windows; the windows are suppressed
                               either way.

    Returns:
        List of Event objects.
    """
    events: list[Event] = []
    countries = sorted(set(c for c, _ in daily_panel.columns))

    for country in countries:
        channels = sorted(c[1] for c in daily_panel.columns if c[0] == country)
        if len(channels) < 2:  # One channel cannot hold out while others run
            continue

        active = build_active_matrix(daily_panel, country, channels, off_threshold)

        for channel in channels:
            series = daily_panel[(country, channel)]
            normal_spend = _normal_spend(series)
            if normal_spend <= 0:  # Market never bought this channel
                continue

            pauses = _find_pauses_ignoring_blips(series, normal_spend, min_days, off_threshold)
            trains = _find_pulse_trains(pauses)
            in_a_train = {i for train in trains for i in train}
            others = [ch for ch in channels if ch != channel]

            for index, pause in enumerate(pauses):
                window = (active.index >= pause.start) & (active.index <= pause.end)

                # How many OTHER channels were also off across this window?
                n_others_off = sum(
                    1 for ch in others
                    if (~active[ch][window]).mean() >= other_off_share
                )
                if n_others_off == len(others):
                    continue  # Dark period
                if n_others_off == len(others) - 1 and len(others) >= 2:
                    continue  # Single-channel period

                # Did the rest of the market keep spending, or cut back too?
                continuity = _market_continuity(daily_panel, country, others, pause)
                if pd.isna(continuity) or continuity < market_continuity_min:
                    continue  # Broad pullback, not a clean holdout

                # The channel must have run BOTH before and after the pause.
                # Without this, a channel that had not launched yet looks like
                # a holdout - but it never stopped, it had not started.
                # "Ran" needs min_days of activity, symmetric with the pause
                # itself: one stray day of spend (a test booking) before a
                # launch must not turn the launch into a holdout.
                ran_before = active[channel][active.index < pause.start].sum() >= min_days
                ran_after = active[channel][active.index > pause.end].sum() >= min_days
                if not ran_before or not ran_after:
                    continue  # Launch (see detect_staggered_launches) or discontinuation

                if index in in_a_train:
                    continue  # One window of a flighting schedule

                events.append(
                    Event(
                        sid=sid,
                        country_code=country,
                        channel=channel,
                        event_type="natural_holdout",
                        start=pause.start,
                        end=pause.end,
                    )
                )

            if emit_pulse_trains:
                events.extend(
                    _build_pulse_events(sid, country, channel, pauses, trains)
                )

    return sorted(events, key=lambda e: (e.country_code or "", e.start))


def _build_pulse_events(
    sid: str,
    country: str,
    channel: str,
    pauses: list[PauseWindow],
    trains: list[list[int]],
) -> list[Event]:
    """Function to emit one grouped Event per pulse train.

    ONE event spanning first start to last end, with the individual windows
    attached as `components`. The grouping is not cosmetic: the evaluation
    harness groups pulse truth the same way, so a detector emitting one event
    per off-window matches no truth row at all.
    """
    events = []
    for train in trains:
        windows = [pauses[i] for i in train]
        events.append(
            Event(
                sid=sid,
                country_code=country,
                channel=channel,
                event_type="channel_pulse",
                start=windows[0].start,
                end=windows[-1].end,
                components=tuple((w.start, w.end) for w in windows),
            )
        )
    return events


def detect_staggered_launches(
    daily_panel: pd.DataFrame,
    sid: str = "",
    min_days: int = 7,
    off_threshold: float = 0.15,
    other_off_share: float = 0.80,
    comparison_on_share: float = 0.80,
) -> list[Event]:
    """Function to detect staggered launches: a channel starts later in some markets.

    While a late market is still waiting, the markets that already run the
    channel are its comparison group - the regions take turns being each
    other's control.

    A launch is the channel's LEADING pause: off from the first day of the data
    until it switches on. That is exactly the window `detect_natural_holdouts`
    rejects as "not yet launched", so the two detectors split those windows
    between them instead of both claiming one.

    One event per LATE market, covering the days it was still off while another
    market was already running the channel. The per-market split is required:
    the harness has one truth row per late market, and a single panel-wide
    event matches none of them.

    Not reported:
        - a channel that switches on everywhere on the same day (no comparison group)
        - the first market to launch (nobody ran it earlier, so it is not late)
        - a window where the whole market is dark (a market that had not opened
          yet, not a channel that had not launched)

    Arguments:
        daily_panel:         DataFrame with daily spend data.
        sid:                 Scenario identifier.
        min_days:            Minimum length of the late window to be reported.
        off_threshold:       Fraction of normal spend below which a day is off.
        other_off_share:     Share of the window another channel IN THIS MARKET
                             must be off across to count as off (dark-market check).
        comparison_on_share: Share of the window another MARKET must be running
                             the channel across to count as a comparison.

    Returns:
        List of Event objects, one per late market. The markets that served as
        the comparison are recorded in `tags` as "compare_with:AT,CH,...".
    """
    countries = sorted(set(c for c, _ in daily_panel.columns))
    channels_by_country = {
        country: sorted(ch for c, ch in daily_panel.columns if c == country)
        for country in countries
    }
    active_by_country = {
        country: build_active_matrix(daily_panel, country, chans, off_threshold)
        for country, chans in channels_by_country.items()
    }
    all_channels = sorted({ch for chans in channels_by_country.values() for ch in chans})

    events: list[Event] = []
    for channel in all_channels:
        markets = [
            country for country in countries
            if channel in channels_by_country[country]
            and _normal_spend(daily_panel[(country, channel)]) > 0
        ]
        if len(markets) < 2:
            continue  # Nothing to stagger against

        # First day the channel ran in each market
        first_on = {}
        for country in markets:
            on = active_by_country[country][channel]
            first_on[country] = on.idxmax() if on.any() else None

        for country in markets:
            series = daily_panel[(country, channel)]
            pauses = _find_pauses_ignoring_blips(series, _normal_spend(series), min_days, off_threshold)
            if not pauses or not pauses[0].touches_start or pauses[0].touches_end:
                continue  # Running from day one, or never switched on here
            lead = pauses[0]

            # The window starts once some other market is already running it
            earlier = [
                first_on[o] for o in markets
                if o != country and first_on[o] is not None and first_on[o] <= lead.end
            ]
            if not earlier:
                continue  # Nobody ran it first: simultaneous (or first) launch
            start = max(lead.start, min(earlier))
            window = (daily_panel.index >= start) & (daily_panel.index <= lead.end)
            if window.sum() < min_days:
                continue

            # Markets genuinely running it across the window, not just one day
            comparison = [
                o for o in markets
                if o != country
                and active_by_country[o][channel][window].mean() >= comparison_on_share
            ]
            if not comparison:
                continue

            # A whole market dark at the start had not opened; that is not a
            # channel launch
            here = active_by_country[country]
            others_here = [ch for ch in channels_by_country[country] if ch != channel]
            if others_here and all(
                (~here[ch][window]).mean() >= other_off_share for ch in others_here
            ):
                continue

            events.append(
                Event(
                    sid=sid,
                    country_code=country,
                    channel=channel,
                    event_type="staggered_launch",
                    start=start,
                    end=lead.end,
                    tags=("compare_with:" + ",".join(comparison),),
                )
            )

    return sorted(events, key=lambda e: (e.country_code or "", e.start))


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
    events += detect_natural_holdouts(panel, sid=sid)
    events += detect_staggered_launches(panel, sid=sid)
    return sorted(events, key=lambda e: (e.country_code or "", e.start))


def run_detection(
    media_df: pd.DataFrame | None,
    sales_df: pd.DataFrame | None = None,
    sid: str = ""
) -> list[Event]:

    return detect_informative_periods(media_df, sales_df, sid)
