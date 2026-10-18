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


def _boundary_cell(b: dict, key: str) -> str:
    """Boundary error is defined only over MATCHED pairs. With no matches at
    all `metrics.boundary_error` returns zeros -- and rendered bare, a detector
    that matched nothing (`never_detect`) reads as flawless boundary
    localisation, which is the exact inverse of the truth. Same degenerate-row
    class as the breakdown buckets above: render it as `n/a` and publish `n`
    beside it so the reader can see how many matches the figure rests on."""
    return "n/a" if not b.get("n") else f"{b[key]:.1f}"


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
        f"| Day coverage F1 (type ignored; pulse envelopes) | {_pct(results['day_level']['f1'])} |",
        "",
        "Read F1 alongside recall and false alarms. IoU and boundary errors "
        "describe matched events only. Day coverage ignores event type and "
        "uses the outer pulse window; it does not verify individual pauses.",
        "",
        "## Boundary error (days)",
        "",
        "| | median | p90 | n (matched pairs) |",
        "|---|---|---|---|",
        f"| start | {_boundary_cell(results['boundary'], 'start_median')} "
        f"| {_boundary_cell(results['boundary'], 'start_p90')} "
        f"| {results['boundary'].get('n', 0)} |",
        f"| end | {_boundary_cell(results['boundary'], 'end_median')} "
        f"| {_boundary_cell(results['boundary'], 'end_p90')} "
        f"| {results['boundary'].get('n', 0)} |",
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

    pulse = results.get("pulse_components")
    if pulse and (pulse["n_truth_pulses"] or pulse["n_pred_pulses"]):
        m = pulse["event_level"]
        lines += ["", "## Pulse off-window accuracy", "",
                  "Individual pauses are matched at IoU >= 0.5; day coverage "
                  "uses only the off-windows. Read alongside grouped-event F1.", "",
                  "| metric | value |", "|---|---|",
                  f"| Component precision | {_pct(m['precision'])} |",
                  f"| Component recall | {_pct(m['recall'])} |",
                  f"| Component F1 | {_pct(m['f1'])} |",
                  f"| Component TP / FP / FN | {m['n_tp']} / {m['n_fp']} / {m['n_fn']} |",
                  f"| Off-day coverage F1 | {_pct(pulse['day_level']['f1'])} |",
                  f"| Predictions missing components | {pulse['missing_pred_components']} |",
                  f"| Truth pulses missing components | {pulse['missing_truth_components']} |"]
        if pulse["missing_truth_components"]:
            lines += ["", "Warning: incomplete truth components prevent a full pulse assessment."]

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

    else:
        lines += ["", "## Operating curve", "",
                  "_Unavailable: no predictions, or some predictions lack confidence "
                  "scores. Headline metrics still include every prediction._"]

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

    event_breakdowns = results.get("event_breakdowns") or {}
    if event_breakdowns:
        lines += [
            "## Event-level breakdowns",
            "",
            "_Spec §9 item 10's duration, magnitude and near-zero-vs-exact-zero"
            " axes. `meta.json` carries none of the three, so these bucket "
            "individual TRUTH events rather than whole scenarios — which makes "
            "them **recall-only**: a false positive belongs to no truth "
            "bucket, so precision has no denominator here and is not "
            "reported._",
            "",
        ]
        for axis, rows in event_breakdowns.items():
            lines += [f"### {axis}", ""]
            if not rows:
                lines += ["_No truth event carries this property._", ""]
                continue
            lines += ["| value | truth events | matched | recall |",
                      "|---|---|---|---|"]
            for r in rows:
                lines.append(
                    f"| {r['value']} | {r['n_events']} | {r['n_matched']} "
                    f"| {_pct(r['recall'])} |")
            lines.append("")

    return "\n".join(lines)
