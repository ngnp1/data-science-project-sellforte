"""Render an evaluation result as readable Markdown."""
from __future__ import annotations

from benchmark.eval.breakdowns import CONFOUNDED_AXES


def _pct(x: float) -> str:
    return f"{x:.3f}"


def _is_degenerate(m: dict) -> bool:
    """True when a breakdown row has no TP, FP, or FN at all -- e.g. a `null`
    family bucket, which has no truth events and (correctly) no predictions.
    `metrics.prf` reports 0.0/0.0/0.0 for that case by its own frozen
    convention, but rendering it as `0.000` reads as total failure on exactly
    the bucket where the detector is behaving perfectly. Presentational only:
    the underlying dict still carries the real zeros for anything downstream
    that aggregates n_tp/n_fp/n_fn."""
    return m["n_tp"] + m["n_fp"] + m["n_fn"] == 0


def _rate_cell(m: dict, key: str) -> str:
    return "n/a" if _is_degenerate(m) else _pct(m[key])


def render_markdown(results: dict) -> str:
    o = results["overall"]
    lines = [
        f"# Evaluation — {results['split']} split",
        "",
        f"{results['n_scenarios']} scenarios.",
        "",
        "## Headline",
        "",
        "| metric | value |",
        "|---|---|",
        f"| Precision | {_pct(o['precision'])} |",
        f"| Recall | {_pct(o['recall'])} |",
        f"| F1 | {_pct(o['f1'])} |",
        f"| TP / FP / FN | {o['n_tp']} / {o['n_fp']} / {o['n_fn']} |",
        f"| Mean IoU | {_pct(results['iou']['mean_iou'])} |",
        f"| Median IoU | {_pct(results['iou']['median_iou'])} |",
        f"| False positives per country-year (null scenarios) "
        f"| {results['null_fp_rate']:.3f} |",
        f"| Channel accuracy (relaxed match) "
        f"| {_pct(results['accuracy']['channel_accuracy'])} |",
        f"| Market accuracy (relaxed match) "
        f"| {_pct(results['accuracy']['market_accuracy'])} |",
        f"| Day-level F1 | {_pct(results['day_level']['f1'])} |",
        "",
        "## Boundary error (days)",
        "",
        "| | median | p90 |",
        "|---|---|---|",
        f"| start | {results['boundary']['start_median']:.1f} "
        f"| {results['boundary']['start_p90']:.1f} |",
        f"| end | {results['boundary']['end_median']:.1f} "
        f"| {results['boundary']['end_p90']:.1f} |",
        "",
        "## Per event type",
        "",
        "| type | precision | recall | F1 | TP | FP | FN |",
        "|---|---|---|---|---|---|---|",
    ]
    for t, m in sorted(results["per_type"].items()):
        lines.append(
            f"| {t} | {_pct(m['precision'])} | {_pct(m['recall'])} "
            f"| {_pct(m['f1'])} | {m['n_tp']} | {m['n_fp']} | {m['n_fn']} |")

    if results["confusion"]:
        lines += ["", "## Type confusion (truth → predicted)", "",
                  "| truth | predicted | n |", "|---|---|---|"]
        for (a, b), n in sorted(results["confusion"].items()):
            flag = "" if a == b else "  ← substitution"
            lines.append(f"| {a} | {b}{flag} | {n} |")

    if results.get("reliability"):
        lines += ["", "## Reliability — is the confidence score honest?", "",
                  "| confidence bin | n | mean confidence | empirical precision |",
                  "|---|---|---|---|"]
        for b in results["reliability"]:
            lines.append(
                f"| {b['bin_lo']:.1f}–{b['bin_hi']:.1f} | {b['n']} "
                f"| {_pct(b['mean_confidence'])} "
                f"| {_pct(b['empirical_precision'])} |")
    else:
        lines += ["", "## Reliability", "",
                  "_No detection carried a confidence score, so calibration "
                  "cannot be assessed._"]

    if results.get("operating"):
        lines += ["", "## Operating curve — the precision/recall trade-off", "",
                  "| confidence cut | detections kept | precision | recall | F1 |",
                  "|---|---|---|---|---|"]
        for c in results["operating"]:
            lines.append(
                f"| {c['cut']:.2f} | {c['n_pred']} | {_pct(c['precision'])} "
                f"| {_pct(c['recall'])} | {_pct(c['f1'])} |")

    lines += ["", "## Breakdowns", ""]
    for axis, data in results["breakdowns"].items():
        warning = data.get("_warning", "")
        lines.append(f"### {axis}" + (" ⚠️" if axis in CONFOUNDED_AXES else ""))
        lines.append("")
        if warning:
            lines += [f"> **{warning}**", ""]
        lines += ["| value | scenarios | precision | recall | F1 |",
                  "|---|---|---|---|---|"]
        for value, m in sorted((k, v) for k, v in data.items()
                               if k != "_warning"):
            lines.append(
                f"| {value} | {m['n_scenarios']} | {_rate_cell(m, 'precision')} "
                f"| {_rate_cell(m, 'recall')} | {_rate_cell(m, 'f1')} |")
        lines.append("")

    return "\n".join(lines)
