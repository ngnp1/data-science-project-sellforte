"""Spec section 8 -- the templated natural-language explanation.

Every number in the prose comes from the event or the panel. Nothing is
softened and nothing is invented: the register is a colleague pointing at the
data, not a model expressing an opinion. An analyst must be able to check
every claim in the sentence against the export.

Must work on an UNSCORED event (detection_confidence and informativeness both
None): this is also the debugging surface reached for when scoring itself is
what went wrong, so it cannot depend on scoring having already run.

Uses `subject_channels`, the single public helper detection/score.py and
detection/validity.py both already share, rather than reading `event.channel`
directly wherever the claim is about "the channel(s) that stopped" -- for a
single_channel event `channel` names the channel still RUNNING, and reading it
directly here would repeat the inversion that has already produced real
defects elsewhere in this codebase.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from detection.io.panel import Panel
from detection.model import DetectedEvent
from detection.primitives.zero_runs import OffRun, find_off_runs, off_mask
from detection.score import subject_channels

_TYPE_PHRASE = {
    "dark_period": "every channel in the market stopped together",
    "single_channel": "all channels but one stopped",
    "natural_holdout": "one channel stopped while the rest kept running",
    "step_change": "spend moved to a new level and held there",
    "channel_pulse": "the channel switched on and off repeatedly",
    "staggered_launch": "the channel started later here than elsewhere",
}

# Event types for which _rank_off_run's single-series "longest run in this
# series" comparison is both well-defined (a single channel, a single
# contiguous off-run) and not already covered by a more specific sentence
# elsewhere in explain(): channel_pulse gets its own pulse-count sentence, and
# step_change and staggered_launch have no off-run to rank at all.
_RANKABLE_TYPES = frozenset({"natural_holdout", "single_channel", "dark_period"})


def _typical_level(panel: Panel, country: str | None, channel: str | None,
                   start, end) -> float:
    """Median spend on active days OUTSIDE the event window."""
    if channel is None or country is None:
        return float("nan")
    key = (country, channel)
    if key not in panel.spend.columns:
        return float("nan")
    series = panel.spend[key]
    outside = series.drop(series.loc[start:end].index)
    active = outside[outside > 0]
    return float(active.median()) if not active.empty else float("nan")


def _window_level(panel: Panel, country: str | None, channel: str | None,
                  start, end) -> float:
    """Mean spend INSIDE the event window, on the same series."""
    if channel is None or country is None:
        return float("nan")
    key = (country, channel)
    if key not in panel.spend.columns:
        return float("nan")
    window = panel.spend.loc[start:end, key]
    return float(window.mean()) if not window.empty else float("nan")


def _market_total(panel: Panel, country: str | None) -> pd.Series | None:
    """Summed daily spend across every channel this market runs, or None
    when there is no market to sum (no country, or the country runs no
    channel at all)."""
    if country is None:
        return None
    channels = panel.channels_in(country)
    if not channels:
        return None
    cols = [(country, ch) for ch in channels]
    return panel.spend[cols].sum(axis=1)


def _typical_market_level(panel: Panel, country: str | None, start, end) -> float:
    """Median MARKET-TOTAL spend on active days OUTSIDE the window -- the
    same "median over active days" rule _typical_level uses for one channel,
    applied to the sum across every channel the market runs.

    A country-level event (dark_period: `event.channel` is always None)
    has no single channel for _typical_level to read, so without this the
    explanation's most checkable sentence -- "fell from a typical X to
    exactly Y" -- was silently dropped for precisely the event type this
    detector is strongest at. Reading the TOTAL rather than one channel is
    deliberate: a dark period is a claim about every channel at once, so the
    number that claim is checked against is the market's whole daily spend,
    not any single channel's.
    """
    total = _market_total(panel, country)
    if total is None:
        return float("nan")
    outside = total.drop(total.loc[start:end].index)
    active = outside[outside > 0]
    return float(active.median()) if not active.empty else float("nan")


def _window_market_level(panel: Panel, country: str | None, start, end) -> float:
    """Mean MARKET-TOTAL spend INSIDE the window -- the market-wide
    counterpart to _window_level."""
    total = _market_total(panel, country)
    if total is None:
        return float("nan")
    window = total.loc[start:end]
    return float(window.mean()) if not window.empty else float("nan")


def _best_covering_run(panel: Panel, country: str | None, channel: str | None,
                       start, end) -> OffRun | None:
    """The off-run on `channel`'s series with the largest overlap with
    [start, end], or None when the channel/country is missing or the series
    has no off-run at all. Shared by `_rank_off_run` (the distinctiveness
    sentence) and `explain()`'s "stopped" vs "cut to a trickle" wording,
    which both need the SAME run, not two independently-selected ones that
    could silently disagree.
    """
    if not channel or not country:
        return None
    key = (country, channel)
    if key not in panel.spend.columns:
        return None
    series = panel.spend[key]
    present = panel.present_mask(country, channel)
    runs = find_off_runs(series, present)
    if not runs:
        return None
    best, best_overlap = None, 0
    for run in runs:
        overlap = (min(run.end, end) - max(run.start, start)).days + 1
        if overlap > best_overlap:
            best, best_overlap = run, overlap
    return best


def _rank_off_run(panel: Panel, country: str | None, channel: str | None,
                  start, end) -> str | None:
    """"This is the longest off-run in this series by a factor of Q, the
    next longest being M days" -- the spec's own worked register, computed
    against every off-run this channel's series has, not just the notable
    ones, since the spec's own example ranks against a 2-day runner-up that
    is itself well under the reportable floor.

    Returns None when there is no single well-defined run to rank (no
    channel, no off-run at all, or nothing else to compare it against),
    rather than force a sentence with nothing behind it.
    """
    if not channel or not country:
        return None
    key = (country, channel)
    if key not in panel.spend.columns:
        return None
    series = panel.spend[key]
    present = panel.present_mask(country, channel)
    runs = find_off_runs(series, present)
    if not runs:
        return None

    best = _best_covering_run(panel, country, channel, start, end)
    if best is None:
        return None

    others = [r.n_days for r in runs
             if not (r.start == best.start and r.end == best.end)]
    if not others:
        return f"This is the only off-run on {channel} in this series."

    # Every OffRun spans at least one day (n_days = hi - lo + 1 with hi >=
    # lo), so `longest_other` here is always >= 1 and division below never
    # sees a zero denominator.
    longest_other = max(others)
    if best.n_days > longest_other:
        factor = best.n_days / longest_other
        return (f"This is the longest off-run on {channel} in this series "
                f"by a factor of {factor:.1f}, the next longest being "
                f"{longest_other} days.")
    if best.n_days == longest_other:
        return (f"This ties for the longest off-run on {channel} in this "
                f"series, both {best.n_days} days.")
    return (f"{channel}'s longest off-run in this series runs "
            f"{longest_other} days; this one is shorter.")


def _control_sentence(event: DetectedEvent, subjects: list[str]) -> str | None:
    """Describes the control group honestly for however many channels this
    event actually claims -- a dark_period's subjects are every channel in
    the market, and saying "this channel" (singular) about it would be a
    claim the export cannot back up. The cross-market layer's own rule (see
    detection/compose/cross_market.py's _peer_status) counts a peer as
    "running" if it kept even ONE subject channel alive, not necessarily
    all of them, so the multi-channel wording says "at least one of" rather
    than implying every peer ran every one of this market's channels.
    """
    control = event.evidence.get("control_available")
    if control is None:
        return None
    single = subjects[0] if len(subjects) == 1 else None

    if control == "peers":
        n = event.evidence.get("peers_running")
        counted = isinstance(n, (int, float)) and n
        if single is not None:
            if counted:
                n = int(n)
                noun = "market" if n == 1 else "markets"
                pronoun = "it is" if n == 1 else "they are"
                return (f"{n} other {noun} ran {single} while it was off "
                        f"here, so {pronoun} available as a control.")
            return (f"Other markets ran {single} throughout, so they are "
                    f"available as controls.")
        if counted:
            n = int(n)
            noun = "market" if n == 1 else "markets"
            return (f"{n} other {noun} kept at least one of these "
                    f"{len(subjects)} channels running during the window, "
                    f"so {'it is' if n == 1 else 'they are'} available as "
                    f"a control.")
        return ("Other markets kept at least one of these channels running "
                "throughout, so they are available as a control.")
    if control == "sibling_channels":
        what = single or "these channels"
        return (f"No peer market ran {what}; the market's other channels "
                f"are the only available control.")
    if control == "none":
        return "No control group is available for this window."
    return None


def _base_sentence(event: DetectedEvent, where: str,
                   off_kind: str | None = None) -> str:
    """The opening claim. Phrased per type because `event.channel` means a
    different thing for each one: for a dark_period it is always None (the
    claim is about every channel, so naming "every channel" twice -- once as
    subject, once inside the generic phrase -- read as filler); for a
    single_channel event it names the channel that stayed RUNNING, and
    leading the sentence with that name reads as if that channel were the
    one that stopped, exactly the inversion this module's tests exist to
    catch. Every other type's `channel` already names the channel the claim
    is actually about, so the generic form applies unchanged.

    `off_kind` is the OffRun.kind ("exact_zero" | "near_zero" | "missing")
    of the run the claim is actually about, for the two event types
    (natural_holdout, single_channel) whose default phrasing says "stopped".
    A near-zero run -- spend cut to a small fraction of normal, not to
    nothing -- is NOT a stop: saying "stopped" and then, one sentence later,
    naming a nonzero window level contradicts itself in the reader's face.
    Only "near_zero" changes the wording; "missing" (a data gap, not a
    confirmed pause) keeps the default phrasing and relies on the separate
    Caveat sentence (validity.py's suspect_data_gap) to flag the gap instead
    -- collapsing "missing" into this near-zero check would trade one
    unchecked claim for a different one.
    """
    span = (f"for {event.n_days} consecutive days "
           f"({event.start.date()} to {event.end.date()})")
    if event.event_type == "dark_period":
        return f"{where}: every channel stopped together {span}."
    if event.event_type == "single_channel":
        survivor = event.channel or "one channel"
        verb = "were cut to a trickle" if off_kind == "near_zero" else "stopped"
        return f"{where}: every channel except {survivor} {verb} {span}."
    what = event.channel or "every channel"
    if event.event_type == "natural_holdout" and off_kind == "near_zero":
        phrase = "one channel was cut to a trickle while the rest kept running"
    else:
        phrase = _TYPE_PHRASE.get(event.event_type, event.event_type)
    return f"{what} in {where}: {phrase} {span}."


def _isolation_sentence(panel: Panel, event: DetectedEvent,
                        subjects: list[str], where: str) -> str | None:
    """Spec section 8's "All five other AT channels ran at normal levels
    throughout" -- a claim about the market's OTHER channels, distinct from
    cross-market control availability, and directly checkable against
    off_mask the same way validity.py and score.py already read it.

    Only for natural_holdout and channel_pulse: both name exactly one
    subject channel and make no claim at all about the rest of the market
    except that they are unaffected, so "N other channels ran normally" adds
    real information. dark_period has no single subject channel to contrast
    against (already excluded, per the accepted round-1 finding).
    single_channel's claim is the opposite one -- every OTHER channel
    stopped, which its own base sentence already states -- so a "ran
    normally" sentence would contradict it rather than support it.
    staggered_launch and step_change make no claim about neighbouring
    channels at all (see detection/score.py's own note on step_change).
    """
    if event.event_type not in {"natural_holdout", "channel_pulse"}:
        return None
    country = event.country_code
    if not country:
        return None
    others = [ch for ch in panel.channels_in(country) if ch not in subjects]
    if not others:
        return None
    normal = 0
    for ch in others:
        off = off_mask(panel.series(country, ch), panel.present_mask(country, ch))
        if not bool(off.loc[event.start:event.end].any()):
            normal += 1
    if normal == len(others):
        if len(others) == 1:
            # "All 1 other channel" is not a sentence -- read the whole
            # module out loud, not just the numbers in it, and this is
            # exactly the kind of thing that check catches.
            return f"The other channel in {where} ran at normal levels throughout."
        return (f"All {len(others)} other channels in {where} ran at normal "
                f"levels throughout.")
    noun = "channel" if len(others) == 1 else "channels"
    return (f"{normal} of {len(others)} other {noun} in {where} ran at "
           f"normal levels throughout the window.")


def explain(event: DetectedEvent, panel: Panel) -> str:
    where = event.country_code or "the panel"

    subjects = subject_channels(event, panel)
    primary = subjects[0] if len(subjects) == 1 else None

    # The base sentence's "stopped" wording is only correct for a genuine
    # exact zero; a near-zero run needs different wording (see
    # _base_sentence's docstring), so the run has to be looked up before the
    # base sentence is built, not after.
    off_kind = None
    if primary is not None and event.event_type in {"natural_holdout",
                                                     "single_channel"}:
        covering = _best_covering_run(panel, event.country_code, primary,
                                      event.start, event.end)
        if covering is not None:
            off_kind = covering.kind

    parts = [_base_sentence(event, where, off_kind)]

    typical = _typical_level(panel, event.country_code, primary,
                             event.start, event.end)
    window = _window_level(panel, event.country_code, primary,
                           event.start, event.end)
    if np.isfinite(typical):
        if np.isfinite(window):
            # Direction has to be read off the actual numbers, not assumed:
            # a step_change window can hold a HIGHER level than the typical
            # outside it (a budget increase), and "fell" would be a false
            # claim there. Every stopping type (dark_period, single_channel,
            # natural_holdout, channel_pulse, staggered_launch's dormancy
            # window) always lands in the window <= 0 case, so it always
            # reads "fell ... to exactly 0" without needing a type check.
            #
            # The nonzero cases say "an average of" rather than hedging with
            # "about": the number is an exact, recomputable mean of the
            # window, not an estimate, and the brief's own register --
            # spec section 8's "to exactly EUR0" -- states both endpoints
            # flatly. "An average of" is precise about what kind of number
            # this is (a window mean, not a single day's spend) without
            # understating how sure the detector is of it. It also has to
            # read as a NOUN PHRASE, not a verb -- "level_word" fills the
            # slot "... to {level_word} per day...", and "to averaged 405
            # per day" is not a sentence a verb belongs in.
            if window <= 0:
                verb, level_word = "fell", "exactly 0"
            elif window > typical:
                verb, level_word = "rose", f"an average of {window:,.0f}"
            elif window < typical:
                verb, level_word = "fell", f"an average of {window:,.0f}"
            else:
                verb, level_word = "moved", f"an average of {window:,.0f}"
            parts.append(
                f"{primary} {verb} from a typical {typical:,.0f} per day to "
                f"{level_word} per day over this window.")
        else:
            parts.append(
                f"Normal spend on {primary} is typically {typical:,.0f} per "
                f"day outside the window.")
    elif event.channel is None:
        # A country-level event (dark_period) has no single channel for the
        # block above to read -- event.channel is always None for one -- so
        # without this fallback the most checkable sentence in the whole
        # explanation ("fell from a typical X to exactly Y") was silently
        # missing for exactly the event type this detector is strongest at.
        # Reports the MARKET's total daily spend across every channel it
        # runs, computed the same active-days-median way as the
        # single-channel sentence above.
        market_typical = _typical_market_level(panel, event.country_code,
                                               event.start, event.end)
        market_window = _window_market_level(panel, event.country_code,
                                             event.start, event.end)
        if np.isfinite(market_typical) and np.isfinite(market_window):
            if market_window <= 0:
                verb, level_word = "fell", "exactly 0"
            elif market_window > market_typical:
                verb, level_word = "rose", f"an average of {market_window:,.0f}"
            elif market_window < market_typical:
                verb, level_word = "fell", f"an average of {market_window:,.0f}"
            else:
                verb, level_word = "moved", f"an average of {market_window:,.0f}"
            n_channels = (len(panel.channels_in(event.country_code))
                         if event.country_code else len(subjects))
            parts.append(
                f"Total spend across {n_channels} channels in {where} "
                f"{verb} from a typical {market_typical:,.0f} per day to "
                f"{level_word} per day over this window.")

    if event.event_type == "step_change":
        if event.magnitude_ratio is None:
            parts.append(
                "Spend rose from zero, so there is no finite ratio to "
                "report.")
        else:
            parts.append(
                f"Spend moved to {event.magnitude_ratio:.2f}x its previous "
                f"level.")
        z = event.evidence.get("z")
        if z is not None:
            parts.append(f"The shift measures {abs(float(z)):.1f} robust "
                         f"standard deviations.")

    if event.event_type == "channel_pulse":
        n = event.evidence.get("n_pulses", len(event.components))
        parts.append(
            f"This is one pulse train of {n} separate off-windows, reported "
            f"as a single event because the windows belong to one flighting "
            f"pattern.")

    if event.event_type in _RANKABLE_TYPES and primary is not None:
        rank = _rank_off_run(panel, event.country_code, primary,
                             event.start, event.end)
        if rank:
            parts.append(rank)

    isolation = _isolation_sentence(panel, event, subjects, where)
    if isolation:
        parts.append(isolation)

    if event.event_type == "staggered_launch":
        onset = event.evidence.get("onset")
        earliest = event.evidence.get("earliest_market_onset")
        n_markets = event.evidence.get("n_markets")
        if onset and earliest and n_markets:
            channel_name = event.channel or "the channel"
            parts.append(
                f"{channel_name} first ran here on {onset}, versus "
                f"{earliest} in the earliest of the {n_markets} markets "
                f"compared.")

    control_sentence = _control_sentence(event, subjects)
    if control_sentence:
        parts.append(control_sentence)

    label = event.event_type
    if event.tags:
        label += " (" + ", ".join(event.tags) + ")"
    scored = []
    if event.detection_confidence is not None:
        scored.append(f"confidence {event.detection_confidence:.2f}")
    if event.informativeness is not None:
        scored.append(f"informativeness {event.informativeness:.2f}")
    scored.append(f"validity {event.validity}")
    # Capitalise only the first character -- str.capitalize() would also
    # force-lowercase everything after it, which is harmless today (every
    # value here is already lowercase) but silent enough to mangle a
    # validity string with real casing later without any test noticing.
    scored_sentence = ", ".join(scored)
    scored_sentence = scored_sentence[0].upper() + scored_sentence[1:]
    parts.append(f"Label: {label}. " + scored_sentence + ".")

    if event.validity_reasons:
        parts.append("Caveat: " + "; ".join(event.validity_reasons) + ".")

    return " ".join(parts)
