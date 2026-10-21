"""Score the natural-holdout and staggered-launch detectors against the committed answer key.

A quick local check while developing, for when the full benchmark has not been
generated yet. The real evaluation is the harness:

    python -m benchmark.eval.run_dev --detector detection.pipeline:run_detection

Run this from the repository root:

    python verify_holdouts.py
    python verify_holdouts.py --explain     # every window, its verdict, and why

Only the two patterns these detectors own are scored. Everything else in the
answer key -- the dark period, the single-channel period, the pulse train, the
step change -- belongs to another detector and is listed, not scored. Ruling a
dark period out is the right answer here; reporting it would be a false positive.

Two conventions have to be right or every window looks a day wrong:

- ground_truth.csv writes `end_date` as the first UNAFFECTED day, because
  events_config.yaml uses Python-slice offsets. The last affected day is one
  earlier, so this subtracts a day before comparing.
- Staggered-launch truth is one row per late market, so each must be matched
  by an event carrying that market's country code.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from detection.algorithms import (
    _find_pauses_ignoring_blips,
    _find_pulse_trains,
    _market_continuity,
    _normal_spend,
    build_active_matrix,
    detect_natural_holdouts,
    detect_staggered_launches,
    prepare_daily_spend_panel,
)

DATA = Path(__file__).resolve().parent / "synthetic_data_generator" / "data"

OWNED = {"natural_holdout", "staggered_launch"}

# Must match the defaults of detect_natural_holdouts
MIN_DAYS, OFF_THRESHOLD, OTHER_OFF_SHARE, CONTINUITY_MIN = 7, 0.15, 0.80, 0.60


def load_truth(path: Path) -> pd.DataFrame:
    truth = pd.read_csv(path)
    truth["start_date"] = pd.to_datetime(truth["start_date"])
    truth["end_date"] = pd.to_datetime(truth["end_date"]) - pd.Timedelta(days=1)
    return truth


def iou(a_start, a_end, b_start, b_end) -> float:
    """Temporal intersection over union, on inclusive day intervals."""
    inter = max(0, (min(a_end, b_end) - max(a_start, b_start)).days + 1)
    if not inter:
        return 0.0
    return inter / ((max(a_end, b_end) - min(a_start, b_start)).days + 1)


def score(events, truth, iou_threshold=0.5):
    """One-to-one matching on market, channel, type and IoU >= threshold."""
    targets = truth[truth["pattern_type"].isin(OWNED)]
    matches, claimed = [], set()

    for _, t in targets.iterrows():
        best, best_iou = None, 0.0
        for i, e in enumerate(events):
            if (i in claimed or e.country_code != t["country_code"]
                    or e.channel != t["channel"] or e.event_type != t["pattern_type"]):
                continue
            overlap = iou(e.start, e.end, t["start_date"], t["end_date"])
            if overlap > best_iou:
                best, best_iou = i, overlap

        hit = best is not None and best_iou >= iou_threshold
        if hit:
            claimed.add(best)
        matches.append({
            "pattern_id": t["pattern_id"],
            "type": t["pattern_type"],
            "market": t["country_code"],
            "channel": t["channel"],
            "truth_start": t["start_date"].date(),
            "truth_end": t["end_date"].date(),
            "result": "found" if hit else "MISSED",
            "iou": round(best_iou, 3),
            "start_err": (events[best].start - t["start_date"]).days if hit else None,
            "end_err": (events[best].end - t["end_date"]).days if hit else None,
        })

    false_positives = [e for i, e in enumerate(events) if i not in claimed]
    ignored = sorted(truth.loc[~truth["pattern_type"].isin(OWNED), "pattern_id"])
    return pd.DataFrame(matches), false_positives, ignored


def explain(panel: pd.DataFrame, launches=()) -> pd.DataFrame:
    """Every candidate window and why it was kept or dropped.

    Mirrors the checks in detect_natural_holdouts, in the same order, and marks
    the windows detect_staggered_launches claimed. The right answer for the
    wrong reason is still a bug, so this prints the evidence behind each verdict
    rather than just the surviving events.
    """
    claimed_launch = {(e.country_code, e.channel, e.end) for e in launches}
    rows = []
    for country in sorted({c for c, _ in panel.columns}):
        channels = sorted(ch for c, ch in panel.columns if c == country)
        active = build_active_matrix(panel, country, channels, OFF_THRESHOLD)

        for channel in channels:
            series = panel[(country, channel)]
            normal = _normal_spend(series)
            if normal <= 0:
                continue

            pauses = _find_pauses_ignoring_blips(series, normal, MIN_DAYS, OFF_THRESHOLD)
            in_train = {i for t in _find_pulse_trains(pauses) for i in t}
            others = [ch for ch in channels if ch != channel]

            for i, pause in enumerate(pauses):
                window = (active.index >= pause.start) & (active.index <= pause.end)
                n_off = sum(1 for ch in others
                            if (~active[ch][window]).mean() >= OTHER_OFF_SHARE)
                continuity = _market_continuity(panel, country, others, pause)
                before = int(active[channel][active.index < pause.start].sum())
                after = int(active[channel][active.index > pause.end].sum())

                if n_off == len(others):
                    verdict = "dark_period"
                elif n_off == len(others) - 1 and len(others) >= 2:
                    verdict = "single_channel_period"
                elif pd.isna(continuity) or continuity < CONTINUITY_MIN:
                    verdict = "market_pulled_back"
                elif before < MIN_DAYS:
                    verdict = ("STAGGERED LAUNCH"
                               if (country, channel, pause.end) in claimed_launch
                               else "not_yet_launched")
                elif after < MIN_DAYS:
                    verdict = "discontinued"
                elif i in in_train:
                    verdict = "belongs_to_pulse_train"
                else:
                    verdict = "NATURAL HOLDOUT"

                rows.append({
                    "verdict": verdict, "market": country, "channel": channel,
                    "start": pause.start.date(), "end": pause.end.date(),
                    "days": pause.n_days, "others_off": f"{n_off}/{len(others)}",
                    "continuity": None if pd.isna(continuity) else round(continuity, 3),
                    "on_before": before, "on_after": after,
                })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--media", default=str(DATA / "media.csv"))
    parser.add_argument("--truth", default=str(DATA / "ground_truth.csv"))
    parser.add_argument("--explain", action="store_true",
                        help="show every candidate window and its verdict")
    args = parser.parse_args()

    panel = prepare_daily_spend_panel(pd.read_csv(args.media, parse_dates=["date"]))
    holdouts = detect_natural_holdouts(panel, sid="local")
    launches = detect_staggered_launches(panel, sid="local")
    events = sorted(holdouts + launches, key=lambda e: (e.country_code or "", e.start))

    print(f"{len(events)} events\n")
    for e in events:
        print(f"  {e.event_type:17} {e.country_code}  {e.channel:17} "
              f"{e.start.date()} -> {e.end.date()}  {e.n_days:>3}d"
              + (f"  [{', '.join(e.tags)}]" if e.tags else ""))

    if args.explain:
        pd.set_option("display.width", 200)
        print("\nevery candidate window:\n")
        print(explain(panel, launches).sort_values(["verdict", "market", "channel"])
              .to_string(index=False))

    matches, false_positives, ignored = score(events, load_truth(Path(args.truth)))
    tp = int((matches["result"] == "found").sum())
    fp, fn = len(false_positives), len(matches) - tp
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    boundary = pd.concat([matches["start_err"].dropna().abs(),
                          matches["end_err"].dropna().abs()])

    print(f"\nrecall {recall:.0%}  precision {precision:.0%}  "
          f"false positives {fp}  "
          f"boundary error {boundary.mean() if len(boundary) else float('nan'):.1f} days\n")
    print(matches.to_string(index=False))

    for e in false_positives:
        print(f"false positive: {e.event_type} {e.country_code} {e.channel} "
              f"{e.start.date()} -> {e.end.date()}")
    if ignored:
        print("\nowned by other detectors: " + ", ".join(ignored))


if __name__ == "__main__":
    main()
