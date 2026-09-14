"""Run a detector over a split and aggregate the results."""
from __future__ import annotations

from pathlib import Path

from benchmark.eval import truth as T
from benchmark.eval.breakdowns import breakdown_by, event_breakdowns
from benchmark.eval.matching import match_events
from benchmark.eval.metrics import evaluate_scenario, false_positive_rate, prf
from benchmark.eval.model import Event

BREAKDOWN_AXES = ("noise_level", "family", "n_countries", "n_channels",
                  "years", "trend_p", "market_spread")


def _load_frames(split: str, sid: str, root: Path | None):
    """A detector reads ONLY these two files. Loaded lazily so the oracles,
    which ignore them, do not pay for 157 MB of CSV parsing."""
    import pandas as pd
    from benchmark.harness.runner import DATASETS_DIR, dataset_dir
    d = dataset_dir(split, sid, root or DATASETS_DIR)
    media = pd.read_csv(d / "media.csv", parse_dates=["date"])
    sales = pd.read_csv(d / "sales.csv", parse_dates=["date"])
    return media, sales


def evaluate_split(detector, split: str, root: Path | None = None,
                   sids: list[str] | None = None,
                   load_data: bool = False) -> dict:
    sids = list(sids or T.list_scenarios(split, root))

    per_scenario: dict = {}
    meta_by_sid: dict = {}
    all_truth: list[Event] = []
    all_pred: list[Event] = []
    null_pred: list[Event] = []
    null_country_years = 0.0

    for sid in sids:
        meta = T.load_meta(split, sid, root)
        meta_by_sid[sid] = meta
        truth = T.load_truth(split, sid, root)

        media, sales = (_load_frames(split, sid, root)
                        if load_data else (None, None))
        pred = list(detector(media, sales, sid))

        per_scenario[sid] = evaluate_scenario(truth, pred)
        all_truth.extend(truth)
        all_pred.extend(pred)

        if meta["family"] == "null":
            # Kept as a flat list, not a running count, so the published rate
            # comes out of metrics.false_positive_rate rather than a second,
            # untested copy of the same arithmetic living here.
            null_pred.extend(pred)
            null_country_years += meta["n_countries"] * meta["years"]

    overall_match = match_events(all_truth, all_pred)
    overall = prf(overall_match.n_tp, overall_match.n_fp, overall_match.n_fn)
    overall.update(n_tp=overall_match.n_tp, n_fp=overall_match.n_fp,
                   n_fn=overall_match.n_fn)

    from benchmark.eval.metrics import (boundary_error, channel_and_market_accuracy,
                                        day_level, iou_stats, operating_curve,
                                        per_type, reliability_curve,
                                        type_confusion)

    return {
        "split": split,
        "n_scenarios": len(sids),
        "overall": overall,
        "per_type": per_type(all_truth, all_pred),
        "iou": iou_stats(overall_match),
        "boundary": boundary_error(overall_match),
        "accuracy": channel_and_market_accuracy(all_truth, all_pred),
        "day_level": day_level(all_truth, all_pred),
        "confusion": type_confusion(all_truth, all_pred),
        "reliability": reliability_curve(all_truth, all_pred),
        "operating": operating_curve(all_truth, all_pred),
        "null_fp_rate": false_positive_rate(null_pred, null_country_years),
        "null_country_years": null_country_years,
        "per_scenario": per_scenario,
        "breakdowns": {ax: breakdown_by(per_scenario, meta_by_sid, ax)
                       for ax in BREAKDOWN_AXES},
        # Spec section 9 item 10's duration, magnitude and near-zero-vs-exact
        # -zero axes, which meta.json cannot express -- they live on the
        # individual truth events, so they are bucketed per event rather than
        # per scenario.
        "event_breakdowns": event_breakdowns(overall_match),
    }
