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

    `index` counts within the family, `gindex` across the whole split. Noise and
    trend are driven by the family-local counter while seasonality and market
    spread are driven by the global one, so the axes stay decorrelated -- drive
    them all off `index` and every high-noise scenario would also be an
    extreme-spread scenario, which would make the breakdowns in spec section 9
    uninterpretable.
    """
    noise_level = axes.NOISE_LEVELS[index % 3]
    trend_p = axes.TREND_LEVELS[(index // 3) % 3]
    temp_var = axes.SEASONALITY_LEVELS[gindex % 3]
    spread = ("tight", "moderate", "extreme")[(gindex // 3) % 3]

    if family == "cross_market":
        n_countries = int(rng.choice([3, 5, 8]))
    elif family == "edge":
        n_countries = int(rng.choice([2, 3]))
    else:
        n_countries = int(rng.choice([1, 2, 3, 5, 8]))

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


def _events_launch(rng, countries, channels, n_days):
    ch = channels[int(rng.integers(len(channels)))]["name"]
    out = []
    # Staggered: each market lights up at a different offset; at least one
    # market carries the channel from day one, to act as the comparison group.
    offsets = [0] + [int(rng.choice([45, 90, 150])) for _ in countries[1:]]
    for country, off in zip(countries, offsets):
        if off > 0:
            out += ev.launch(country["code"], ch, off, n_days)
    if not out:  # single-market scenario, force one launch
        out = ev.launch(countries[0]["code"], ch, 90, n_days)
    return out


def _events_cross_market(rng, countries, channels, n_days):
    """One market holds a channel out while every peer keeps running it."""
    ch = channels[int(rng.integers(len(channels)))]["name"]
    c = countries[int(rng.integers(len(countries)))]["code"]
    length = int(rng.choice([42, 56, 90]))
    return ev.holdout(c, ch, _pick_window(rng, n_days, length), length, 0, n_days)


def _events_mixed(rng, countries, channels, n_days):
    """Three to five simultaneous events of different kinds across markets."""
    builders = [_events_dark, _events_single, _events_holdout,
                _events_step, _events_pulse]
    k = int(rng.integers(3, 6))
    idx = rng.choice(len(builders), size=k, replace=False)
    out = []
    for i in idx:
        out += builders[i](rng, countries, channels, n_days)
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
            sid = f"{split}_{counter:03d}_{family}"
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
