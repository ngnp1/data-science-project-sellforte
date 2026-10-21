"""Plain-language names, copy and colours for detected events.

The viewer's reader may not be a statistician, so every detector key that
reaches the screen goes through here first. Colours are the dataviz reference
palette's categorical slots 1-6 in their fixed order, stepped separately for a
light and a dark surface; a type keeps its colour whatever the filters show.
"""
from __future__ import annotations

from dataclasses import dataclass

from detection.model import DetectedEvent


@dataclass(frozen=True)
class TypeInfo:
    name: str
    measures: str
    color: str
    color_dark: str


TYPES: dict[str, TypeInfo] = {
    "dark_period": TypeInfo(
        "All advertising off",
        "Shows baseline sales with no advertising.",
        "#2a78d6", "#3987e5"),
    "single_channel": TypeInfo(
        "Only one channel on",
        "Shows that channel's effect on its own.",
        "#eb6834", "#d95926"),
    "natural_holdout": TypeInfo(
        "One channel paused",
        "Shows what the paused channel adds.",
        "#1baf7a", "#199e70"),
    "step_change": TypeInfo(
        "Budget step",
        "Shows how sales react to a budget change.",
        "#eda100", "#c98500"),
    "channel_pulse": TypeInfo(
        "On–off pulsing",
        "Shows how long ads keep working after they stop.",
        "#e87ba4", "#d55181"),
    "staggered_launch": TypeInfo(
        "Staggered launch",
        "Shows the channel's effect, using other markets as comparison.",
        "#008300", "#008300"),
}

# (plain name, one-line meaning) for the confidence sub-scores and the
# informativeness drivers the pipeline writes into each event's evidence.
SCORE_NAMES: dict[str, tuple[str, str]] = {
    "magnitude_evidence": ("How completely spend changed",
                           "1 means spend fell to exactly zero, or the budget moved far outside its usual range."),
    "duration_evidence": ("Long enough",
                          "Longer periods are less likely to be noise."),
    "distinctiveness": ("Unusual for this channel",
                        "Compared with this channel's other pauses; a routine gap scores low."),
    "edge_sharpness": ("Clean start and stop",
                       "Spend switched over within a day or two instead of drifting."),
    "corroboration": ("Impressions agree",
                      "Impressions stopped too, so this is not just missing spend data."),
    "consistency": ("All channels agree",
                    "For market-wide periods, the share of channels that actually stopped."),
    "duration_adequacy": ("Long enough to see after-effects",
                          "Advertising keeps working for a while after it stops; short periods hide that."),
    "contrast": ("Size of the change",
                 "Bigger changes in spend give a clearer reading."),
    "cleanliness": ("No overlapping findings",
                    "Other changes in the same market at the same time blur the reading."),
    "control_availability": ("Comparison available",
                             "Other markets or channels kept running and can serve as a comparison."),
    "sales_snr": ("Sales moved",
                  "How far sales moved from the 8 weeks before, against their usual ups and downs."),
    "type_prior": ("Usefulness of this pattern",
                   "Some patterns, such as all advertising off, are more informative by nature."),
}


def short_label(e: DetectedEvent) -> str:
    """What and how long, short enough to sit beside a bar on the period map."""
    ch = e.channel
    what = {
        "dark_period": "All channels off",
        "single_channel": f"Only {ch}",
        "natural_holdout": f"{ch} paused",
        "step_change": f"{ch} ×{e.magnitude_ratio:.1f}" if e.magnitude_ratio else f"{ch} step",
        "channel_pulse": f"{ch} ×{len(e.components)} pauses",
        "staggered_launch": f"{ch} late",
    }.get(e.event_type, e.event_type)
    return f"{what} · {e.n_days}d"


def score_rows(parts: dict[str, float]) -> list[tuple[str, str, float]]:
    """(plain name, meaning, value) per sub-score or driver, in input order."""
    return [(*SCORE_NAMES.get(key, (key, "")), value) for key, value in parts.items()]


def priority_band(score: float | None) -> str:
    if score is None:
        return "Unscored"
    return "High" if score >= 0.75 else "Medium" if score >= 0.5 else "Low"


def evidence_band(score: float | None) -> str:
    if score is None:
        return "Unscored"
    return "Strong" if score >= 0.8 else "Moderate" if score >= 0.6 else "Weak"


def event_id(e: DetectedEvent) -> str:
    return f"{e.country_code}|{e.channel or 'ALL'}|{e.event_type}|{e.start:%Y-%m-%d}"


def title(e: DetectedEvent) -> str:
    ch = e.channel
    what = {
        "dark_period": "All advertising off",
        "single_channel": f"Only {ch} on",
        "natural_holdout": f"{ch} paused",
        "step_change": f"{ch} budget step",
        "channel_pulse": f"{ch} on–off pulsing",
        "staggered_launch": f"{ch} launched late",
    }.get(e.event_type, e.event_type)
    return f"{e.country_code} · {what}"

