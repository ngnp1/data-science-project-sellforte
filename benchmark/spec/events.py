"""Builders for events_config.yaml entries, one function per event family.

Every builder returns a list of dicts in the generator's own schema, so the
output can be dumped straight to YAML. Nothing here reads config.yaml -- the
caller is responsible for passing country codes and channel names that exist.

Two pattern types are *negative controls*: they change the data but are not
events a detector should report. They are listed in NON_EVENT_TYPES, and the
evaluation truth loader drops them from the matchable set, so anything detected
inside their windows counts as a false positive -- which is exactly what should
happen for a gradual ramp or a naturally intermittent channel.
"""
from __future__ import annotations

NON_EVENT_TYPES: frozenset[str] = frozenset({"ramp_block", "intermittent_baseline"})

DEFAULT_N_DAYS = 730


def _entry(pattern_id, pattern_type, country, channel, start_day, end_day,
           multiplier, description, n_days):
    if end_day > n_days:
        raise ValueError(
            f"{pattern_id}: window ends on day {end_day}, which exceeds the "
            f"{n_days}-day series")
    if start_day < 0 or start_day >= end_day:
        raise ValueError(f"{pattern_id}: invalid window [{start_day}, {end_day})")
    return dict(pattern_id=pattern_id, pattern_type=pattern_type,
                country=country, channel=channel, start_day=int(start_day),
                end_day=int(end_day), multiplier=multiplier,
                description=description)


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text).upper().strip("_")


def dark(country, start, length, n_days=DEFAULT_N_DAYS):
    """Every channel in one country goes to zero."""
    return [_entry(f"{country}_DARK_{start}", "dark_period", country, "ALL",
                   start, start + length, 0,
                   f"All {country} advertising is off for {length} days.",
                   n_days)]


def single_channel(country, keep, start, length, n_days=DEFAULT_N_DAYS):
    """Only `keep` stays on; every other channel in the country goes to zero."""
    return [_entry(f"{country}_SINGLE_{_slug(keep)}_{start}", "single_channel",
                   country, keep, start, start + length, 0,
                   f"Only {keep} remains active in {country} for {length} days.",
                   n_days)]


def holdout(country, channel, start, length, multiplier=0, n_days=DEFAULT_N_DAYS):
    """One channel drops out; every other channel continues untouched."""
    depth = "to zero" if multiplier == 0 else f"to {multiplier:g}x"
    return [_entry(f"{country}_HOLD_{_slug(channel)}_{start}", "natural_holdout",
                   country, channel, start, start + length, multiplier,
                   f"{channel} drops {depth} in {country} for {length} days "
                   f"while other channels continue.", n_days)]


def step(country, channel, start, length, multiplier, n_days=DEFAULT_N_DAYS):
    """One channel's budget jumps by `multiplier` and holds."""
    direction = "rises" if multiplier > 1 else "falls"
    return [_entry(f"{country}_STEP_{_slug(channel)}_{start}", "step_change",
                   country, channel, start, start + length, multiplier,
                   f"{channel} spend {direction} to {multiplier:g}x in "
                   f"{country} for {length} days.", n_days)]


def pulse(country, channel, starts, length, n_days=DEFAULT_N_DAYS):
    """On/off/on. The only place adstock decay is observable."""
    return [
        _entry(f"{country}_PULSE_{_slug(channel)}_{i + 1}", "channel_pulse",
               country, channel, s, s + length, 0,
               f"{channel} off for {length} days in {country} "
               f"(pulse {i + 1} of {len(starts)}).", n_days)
        for i, s in enumerate(starts)
    ]


def launch(country, channel, length, n_days=DEFAULT_N_DAYS):
    """A channel is dormant at the start of the series, then launches."""
    return [_entry(f"{country}_LAUNCH_{_slug(channel)}", "staggered_launch",
                   country, channel, 0, length, 0,
                   f"{channel} launches in {country} {length} days into the "
                   f"series.", n_days)]


def global_pause(countries, start, length, n_days=DEFAULT_N_DAYS):
    """Every channel in every country stops at once -- real, but with no
    control group, and indistinguishable from a data outage by spend alone."""
    return [
        _entry(f"GLOBAL_PAUSE_{c}_{start}", "global_pause", c, "ALL",
               start, start + length, 0,
               f"All advertising stops in {c} during a {length}-day global "
               f"pause.", n_days)
        for c in countries
    ]


def ramp(country, channel, start, block, multipliers, n_days=DEFAULT_N_DAYS):
    """NEGATIVE CONTROL. Spend drifts upward across contiguous blocks rather
    than stepping. A step detector that fires here is wrong."""
    out = []
    for i, m in enumerate(multipliers):
        s = start + i * block
        out.append(_entry(f"{country}_RAMP_{_slug(channel)}_{i + 1}",
                          "ramp_block", country, channel, s, s + block, m,
                          f"Ramp block {i + 1} of {len(multipliers)} at "
                          f"{m:g}x -- gradual drift, not a step.", n_days))
    return out


def intermittent(country, channel, n, off_len, period, start=0,
                 n_days=DEFAULT_N_DAYS):
    """NEGATIVE CONTROL. A flighting channel whose short gaps are normal. Any
    single gap detected as a holdout is a false positive."""
    return [
        _entry(f"{country}_INTERMITTENT_{_slug(channel)}_{i + 1}",
               "intermittent_baseline", country, channel,
               start + i * period, start + i * period + off_len, 0,
               f"Routine {off_len}-day flighting gap {i + 1} of {n}.", n_days)
        for i in range(n)
    ]
