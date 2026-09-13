"""Deterministic assembly of the 100-scenario benchmark.

Every scenario is a (config.yaml, events_config.yaml, seed) triple. Given the
same code, build_all() always returns the same 100 scenarios -- that is what
makes the sealed test split provable rather than merely asserted.

Development scenarios draw seeds 1000-1999, test scenarios 5000-5999.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

import numpy as np

from benchmark.spec import axes, events as ev

# family -> (n_dev, n_test)
FAMILY_COUNTS: dict[str, tuple[int, int]] = {
    "null": (4, 5),
    "dark": (3, 4),
    "single_channel": (3, 4),
    "holdout": (3, 4),
    "step": (5, 6),
    "pulse": (3, 4),
    "launch": (3, 4),
    "cross_market": (3, 4),
    "mixed": (8, 10),
    "edge": (10, 10),
}

EDGE_CASES = (
    "censored_start", "censored_end", "back_to_back", "overlapping",
    "too_short", "very_long", "single_channel_market",
    "intermittent_channel", "gradual_ramp", "global_pause",
)

SEED_BASE = {"dev": 1000, "test": 5000}

CAMPAIGN_SPEND = dict(daily_total_mean=18500.0, daily_total_std=4000.0)


@dataclass(frozen=True)
class Scenario:
    sid: str
    split: str
    seed: int
    family: str
    years: int
    countries: tuple[dict, ...]
    channels: tuple[dict, ...]
    baseline: dict
    campaign_spend: dict
    events: tuple[dict, ...]
    meta: dict


def _shape(rng, family, index, gindex):
    """Pick the structural parameters for one scenario.

    Stratified rather than independently sampled, so each family reliably spans
    all three noise levels instead of clustering by chance.

    `index` counts within the family, `gindex` across the whole split. Noise
    stays keyed to the family-local counter (`index % 3`), so every family --
    even ones with only 3-6 scenarios -- still cycles through all three noise
    levels. Seasonality, market spread, and trend are each keyed to an
    independent base-3 digit of the global counter `gindex` (`% 3`, `// 3 % 3`,
    `// 9 % 3`) rather than to `index`: driving trend off the family-local
    counter meant families smaller than 9 could never reach `index >= 6`, so
    `trend_p == 1.0` only ever showed up on the two largest families (`mixed`,
    `edge`) -- confounding "high trend" with "hardest family" in every
    downstream breakdown. Keeping the three global axes on independent digits
    of `gindex` (rather than reusing one another's stride, which would make two
    of them identical) keeps them mutually decorrelated too.
    """
    noise_level = axes.NOISE_LEVELS[index % 3]
    temp_var = axes.SEASONALITY_LEVELS[gindex % 3]
    spread = ("tight", "moderate", "extreme")[(gindex // 3) % 3]
    trend_p = axes.TREND_LEVELS[(gindex // 9) % 3]

    if family == "cross_market":
        n_countries = int(rng.choice([3, 5, 8]))
    elif family == "edge":
        n_countries = int(rng.choice([2, 3]))
    elif family == "launch":
        # A staggered launch needs a comparison market, so at least 2.
        n_countries = int(rng.choice([2, 3, 5, 8]))
    elif family == "mixed":
        # A country-level event needs a country to itself, so at least 2.
        n_countries = int(rng.choice([2, 3, 5, 8]))
    else:
        n_countries = int(rng.choice([1, 2, 3, 5, 8]))

    if family == "mixed":
        # Channel-level events sharing a country need distinct channels --
        # up to 3 at once (holdout, step, pulse) -- so keep enough headroom.
        n_channels = int(rng.choice([4, 6, 9, 12]))
    else:
        n_channels = int(rng.choice([2, 4, 6, 9, 12]))

    years = 1 if n_countries >= 8 else 2
    return noise_level, trend_p, temp_var, spread, n_countries, n_channels, years


def _mk(sid, split, seed, family, rng, *, force_countries=None,
        force_channels=None, force_years=None, index=0, gindex=0,
        extra_meta=None, event_fn=None):
    (noise_level, trend_p, temp_var, spread,
     n_countries, n_channels, years) = _shape(rng, family, index, gindex)

    n_countries = force_countries or n_countries
    n_channels = force_channels or n_channels
    years = force_years or (1 if n_countries >= 8 else years)

    countries = axes.pick_countries(rng, n_countries, spread)
    channels = axes.apply_noise(axes.pick_channels(rng, n_channels), noise_level)
    n_days = 365 * years

    entries = tuple(event_fn(rng, countries, channels, n_days)) if event_fn else ()

    sizes = [c["market_size"] for c in countries]
    meta = dict(
        family=family, noise_level=noise_level, trend_p=trend_p,
        temp_var=temp_var, market_spread=spread, n_countries=n_countries,
        n_channels=n_channels, years=years, n_days=n_days,
        market_size_ratio=round(max(sizes) / min(sizes), 3),
        n_events=len(entries),
        event_types=sorted({e["pattern_type"] for e in entries}),
    )
    meta.update(extra_meta or {})

    return Scenario(
        sid=sid, split=split, seed=seed, family=family, years=years,
        countries=tuple(countries), channels=tuple(channels),
        baseline=axes.baseline_for(noise_level, trend_p, temp_var),
        campaign_spend=dict(CAMPAIGN_SPEND), events=entries, meta=meta,
    )


# --- per-family event builders -------------------------------------------
# Each takes (rng, countries, channels, n_days) and returns a list of entries.

def _pick_window(rng, n_days, length):
    """A start offset that leaves the window inside the series with margin."""
    latest = n_days - length - 30
    return int(rng.integers(30, max(31, latest)))


def _events_dark(rng, countries, channels, n_days):
    c = countries[int(rng.integers(len(countries)))]["code"]
    length = int(rng.choice([14, 42, 90]))
    return ev.dark(c, _pick_window(rng, n_days, length), length, n_days)


def _events_single(rng, countries, channels, n_days):
    c = countries[int(rng.integers(len(countries)))]["code"]
    keep = channels[int(rng.integers(len(channels)))]["name"]
    length = int(rng.choice([14, 30, 42]))
    return ev.single_channel(c, keep, _pick_window(rng, n_days, length), length, n_days)


def _events_holdout(rng, countries, channels, n_days):
    c = countries[int(rng.integers(len(countries)))]["code"]
    ch = channels[int(rng.integers(len(channels)))]["name"]
    length = int(rng.choice([14, 42, 90]))
    mult = float(rng.choice([0, 0, 0.02, 0.05, 0.08]))
    return ev.holdout(c, ch, _pick_window(rng, n_days, length), length, mult, n_days)


def _events_step(rng, countries, channels, n_days):
    c = countries[int(rng.integers(len(countries)))]["code"]
    ch = channels[int(rng.integers(len(channels)))]["name"]
    length = int(rng.choice([42, 56, 90]))
    mult = float(rng.choice([0.33, 1.5, 3.0, 5.0]))
    return ev.step(c, ch, _pick_window(rng, n_days, length), length, mult, n_days)


def _events_pulse(rng, countries, channels, n_days):
    c = countries[int(rng.integers(len(countries)))]["code"]
    ch = channels[int(rng.integers(len(channels)))]["name"]
    length = int(rng.choice([10, 14, 21]))
    gap = length + int(rng.choice([14, 21, 28]))
    n_pulses = int(rng.choice([2, 3, 4]))
    first = _pick_window(rng, n_days, gap * n_pulses + length)
    return ev.pulse(c, ch, [first + i * gap for i in range(n_pulses)], length, n_days)


_LAUNCH_OFFSET_POOL = tuple(range(30, 30 + 15 * 12, 15))  # 30, 45, ..., 195


def _events_launch(rng, countries, channels, n_days):
    ch = channels[int(rng.integers(len(channels)))]["name"]
    out = []
    # Staggered: each market lights up on its own distinct offset (sampled
    # without replacement, so two markets never launch on the same day); the
    # first market keeps offset 0 and carries the channel from day one, to
    # act as the comparison group. _shape excludes n_countries == 1 from this
    # family, so there is always at least one other market to stagger.
    offsets = [0] + [int(o) for o in
                      rng.choice(_LAUNCH_OFFSET_POOL, size=len(countries) - 1,
                                 replace=False)]
    for country, off in zip(countries, offsets):
        if off > 0:
            out += ev.launch(country["code"], ch, off, n_days)
    return out


def _events_cross_market(rng, countries, channels, n_days):
    """One market holds a channel out while every peer keeps running it."""
    ch = channels[int(rng.integers(len(channels)))]["name"]
    c = countries[int(rng.integers(len(countries)))]["code"]
    length = int(rng.choice([42, 56, 90]))
    return ev.holdout(c, ch, _pick_window(rng, n_days, length), length, 0, n_days)


_MIXED_BUILDERS = [_events_dark, _events_single, _events_holdout,
                   _events_step, _events_pulse]
# dark_period and single_channel each claim their whole country (channel ==
# "ALL", or every other channel goes to zero) -- nothing else may target that
# country at all. holdout/step/pulse only claim one channel, so several of
# them can share a country as long as they use different channels.
_MIXED_COUNTRY_LEVEL = {_events_dark, _events_single}


def _events_mixed(rng, countries, channels, n_days):
    """Three to five simultaneous events of different kinds across markets.

    Each event gets an exclusive slice of (country, channel) reserved before
    any builder runs, so events can never contradict one another the way R
    applies them in sequence -- e.g. a dark_period zeroing a country and a
    step_change later multiplying that same (now-zero) channel, which would
    leave ground truth listing an event with no visible trace in the data.
    """
    n_countries = len(countries)

    # Country-level events need a country to themselves; any channel-level
    # events need at least one country left over to share. Retry the
    # count/kind draw until it packs into what's available -- with
    # n_countries >= 2 (guaranteed by _shape for this family) a satisfiable
    # combination always exists (e.g. k=3 with no country-level events).
    while True:
        k = int(rng.integers(3, 6))
        idx = rng.choice(len(_MIXED_BUILDERS), size=k, replace=False)
        chosen = [_MIXED_BUILDERS[i] for i in idx]
        n_country_level = sum(1 for b in chosen if b in _MIXED_COUNTRY_LEVEL)
        n_channel_level = k - n_country_level
        slots_needed = n_country_level + (1 if n_channel_level else 0)
        if slots_needed <= n_countries:
            break

    order = [int(i) for i in rng.permutation(n_countries)]
    out = []
    slot = 0

    # Country-level builders each claim one exclusive, never-shared country.
    for b in chosen:
        if b in _MIXED_COUNTRY_LEVEL:
            out += b(rng, [countries[order[slot]]], channels, n_days)
            slot += 1

    # Channel-level builders round-robin across whatever countries are left,
    # sharing only when there are more of them than remaining countries --
    # and when they do share, each is handed only the channels its
    # country-mates have not already claimed.
    remaining = order[slot:]
    used_channels_by_country = {c: set() for c in remaining}
    channel_builders = [b for b in chosen if b not in _MIXED_COUNTRY_LEVEL]
    for j, b in enumerate(channel_builders):
        c_idx = remaining[j % len(remaining)]
        avail = [ch for ch in channels
                 if ch["name"] not in used_channels_by_country[c_idx]]
        entries = b(rng, [countries[c_idx]], avail, n_days)
        for e in entries:
            if e["channel"] != "ALL":
                used_channels_by_country[c_idx].add(e["channel"])
        out += entries

    return out


def _events_edge(rng, countries, channels, n_days, case):
    c0 = countries[0]["code"]
    ch0 = channels[0]["name"]
    ch1 = channels[min(1, len(channels) - 1)]["name"]

    if case == "censored_start":
        return ev.holdout(c0, ch0, 0, 60, 0, n_days)
    if case == "censored_end":
        return ev.holdout(c0, ch0, n_days - 60, 60, 0, n_days)
    if case == "back_to_back":
        return (ev.holdout(c0, ch0, 200, 42, 0, n_days)
                + ev.step(c0, ch0, 242, 56, 3.0, n_days))
    if case == "overlapping":
        return (ev.holdout(c0, ch0, 200, 60, 0, n_days)
                + ev.step(c0, ch1, 230, 60, 2.5, n_days))
    if case == "too_short":
        return ev.holdout(c0, ch0, 300, 5, 0, n_days)
    if case == "very_long":
        return ev.holdout(c0, ch0, 100, 180, 0, n_days)
    if case == "single_channel_market":
        # Every channel but one is held out for the whole series, so the
        # detector sees a market with exactly one active channel.
        out = []
        for ch in channels[1:]:
            out += ev.holdout(c0, ch["name"], 0, n_days, 0, n_days)
        out += ev.dark(c0, 300, 42, n_days)
        return out
    if case == "intermittent_channel":
        return (ev.intermittent(c0, ch0, n=20, off_len=3, period=14, start=30,
                                n_days=n_days)
                + ev.holdout(c0, ch1, 400, 42, 0, n_days))
    if case == "gradual_ramp":
        return ev.ramp(c0, ch0, 200, 10, [1.2, 1.4, 1.6, 1.8, 2.0], n_days)
    if case == "global_pause":
        return ev.global_pause([c["code"] for c in countries], 300, 21, n_days)
    raise ValueError(f"unknown edge case: {case}")


EVENT_FNS = {
    "null": None,
    "dark": _events_dark,
    "single_channel": _events_single,
    "holdout": _events_holdout,
    "step": _events_step,
    "pulse": _events_pulse,
    "launch": _events_launch,
    "cross_market": _events_cross_market,
    "mixed": _events_mixed,
}


def build_split(split: str) -> list[Scenario]:
    base = SEED_BASE[split]
    out: list[Scenario] = []
    counter = 0

    for family, (n_dev, n_test) in FAMILY_COUNTS.items():
        n = n_dev if split == "dev" else n_test
        for i in range(n):
            seed = base + counter
            counter += 1
            # No family suffix: the sid becomes a dataset directory name on the
            # detector side, and spec section 4 requires those to hide generation
            # parameters. `test_001_null` would announce "this scenario contains
            # no events" in the one place a detector is allowed to look.
            sid = f"{split}_{counter:03d}"
            rng = np.random.default_rng(seed)

            if family == "edge":
                case = EDGE_CASES[i % len(EDGE_CASES)]
                forced_channels = 2 if case == "single_channel_market" else None
                out.append(_mk(
                    sid, split, seed, family, rng, index=i, gindex=counter - 1,
                    force_channels=forced_channels,
                    extra_meta={"edge_case": case},
                    event_fn=lambda r, co, ch, nd, _c=case:
                        _events_edge(r, co, ch, nd, _c),
                ))
            else:
                out.append(_mk(sid, split, seed, family, rng, index=i,
                               gindex=counter - 1,
                               event_fn=EVENT_FNS[family]))
    return out


def build_all() -> list[Scenario]:
    return build_split("dev") + build_split("test")


def spec_hash(scenarios: list[Scenario]) -> str:
    """SHA-256 over the canonical JSON of every scenario definition.

    Recorded in the SEALED marker, so a regenerated test split with even one
    changed parameter is detectable.
    """
    blob = json.dumps([asdict(s) for s in scenarios], sort_keys=True,
                      separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()
