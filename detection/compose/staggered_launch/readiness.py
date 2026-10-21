"""Screen launch opportunities; this does not estimate advertising effects."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from detection import params
from detection.io.normalize import active_level
from detection.io.panel import Panel
from detection.model import DetectedEvent, subject_channels
from detection.primitives.zero_runs import off_mask


def _onset(event: DetectedEvent) -> pd.Timestamp:
    """First active day of a launch. Recorded by our detector; for a launch
    from another detector it is the day after the inactive window ends."""
    onset = event.evidence.get("onset")
    return pd.Timestamp(onset) if onset is not None else event.end + pd.Timedelta(days=1)


def _check(status: str, code: str, **evidence) -> dict:
    return {"status": status, "code": code, **evidence}


def _status(checks: dict) -> str:
    statuses = [c["status"] for c in checks.values()]
    if "fail" in statuses:
        return "not_analysis_ready"
    if "unknown" in statuses:
        return "needs_review"
    return "analysis_ready"


def _coverage(panel: Panel, country: str, channel: str,
              dates: pd.DatetimeIndex) -> dict:
    present = panel.present_mask(country, channel).reindex(dates, fill_value=False)
    missing = int((~present).sum())
    return _check("unknown" if missing else "pass",
                  "missing_media_rows" if missing else "observed_media",
                  missing_days=missing, expected_days=len(dates))


def _inactive(panel: Panel, country: str, channel: str,
              dates: pd.DatetimeIndex) -> pd.Series:
    s = panel.series(country, channel)
    # All-zero observed markets can be untreated controls even if never buyers.
    if not np.isfinite(active_level(s)):
        return (s <= params.EPS_ABS).reindex(dates, fill_value=False)
    return off_mask(s).reindex(dates, fill_value=False)


def _competing_changes(events: list[DetectedEvent], panel: Panel, country: str,
                       channel: str, pre: pd.DatetimeIndex,
                       post: pd.DatetimeIndex) -> dict:
    dates = pre.append(post)
    changes, missing = [], []
    channels = sorted(ch for c, ch in panel.spend.columns
                      if c == country and ch != channel)
    for ch in channels:
        coverage = _coverage(panel, country, ch, dates)
        if coverage["status"] != "pass":
            missing.append(ch)
            continue
        s = panel.series(country, ch)
        before = s.reindex(pre).to_numpy().reshape(-1, params.LAUNCH_WEEK_DAYS).sum(axis=1)
        after = s.reindex(post).to_numpy().reshape(-1, params.LAUNCH_WEEK_DAYS).sum(axis=1)
        baseline, following = float(np.median(before)), float(np.median(after))
        change = abs(following - baseline) / max(baseline, params.EPS_ABS)
        if change >= params.LAUNCH_COMPETING_CHANGE:
            changes.append({"channel": ch, "weekly_spend_change": float(change)})
    for other in events:
        if other.country_code != country or not (set(subject_channels(other, panel)) - {channel}):
            continue
        start, end = other.start, other.end
        if other.event_type == "staggered_launch":
            start = end = _onset(other)
        if start <= dates[-1] and dates[0] <= end:
            changes.append({"event_type": other.event_type, "channel": other.channel,
                            "start": str(start.date()), "end": str(end.date())})
    status = "fail" if changes else "unknown" if missing else "pass"
    code = "competing_changes" if changes else "missing_other_channel_rows" if missing else "stable_other_channels"
    return _check(status, code, changes=changes, missing_channels=missing)


def _sales(panel: Panel, country: str, dates: pd.DatetimeIndex) -> pd.Series:
    if country not in panel.sales:
        return pd.Series(float("nan"), index=dates)
    return pd.to_numeric(panel.sales[country].reindex(dates), errors="coerce")


def _sales_comparison(panel: Panel, target: str, peer: str,
                      pre: pd.DatetimeIndex, post: pd.DatetimeIndex) -> dict:
    values = {c: _sales(panel, c, pre.append(post)) for c in (target, peer)}
    invalid = {c: int((~np.isfinite(s) | (s < 0)).sum()) for c, s in values.items()}
    if any(invalid.values()):
        return _check("unknown", "missing_or_invalid_sales", invalid_days=invalid)
    weekly = [values[c].reindex(pre).to_numpy().reshape(-1, params.LAUNCH_WEEK_DAYS).sum(axis=1)
              for c in (target, peer)]
    if any(w.mean() <= 0 or np.ptp(w) <= params.EPS_ABS * max(1, abs(w.mean()))
           for w in weekly):
        return _check("unknown", "flat_prelaunch_sales")
    target_norm, peer_norm = [w / w.mean() for w in weekly]
    correlation = float(np.corrcoef(target_norm, peer_norm)[0, 1])
    drift = float(abs(np.polyfit(np.arange(len(target_norm)), target_norm - peer_norm, 1)[0])
                  * (len(target_norm) - 1))
    passed = correlation >= params.LAUNCH_SALES_CORRELATION and drift <= params.LAUNCH_SALES_DRIFT
    return _check("pass" if passed else "fail",
                  "comparable_prelaunch_sales" if passed else "divergent_prelaunch_sales",
                  correlation=correlation, relative_drift=drift,
                  prelaunch_weekly_variability={c: float(w.std() / w.mean())
                                               for c, w in zip((target, peer), weekly)})


def _verify(event: DetectedEvent, events: list[DetectedEvent], panel: Panel) -> dict:
    onset = _onset(event)
    pre = pd.date_range(onset - pd.Timedelta(days=params.LAUNCH_PRE_DAYS),
                        periods=params.LAUNCH_PRE_DAYS)
    post = pd.date_range(onset, periods=params.LAUNCH_POST_DAYS)
    dates = pre.append(post)
    target, channel = event.country_code, event.channel
    fits = dates[0] >= panel.dates[0] and dates[-1] <= panel.dates[-1]
    checks = {
        "history": _check("pass" if fits else "fail",
                          "adequate_history" if fits else "insufficient_history"),
        "media_coverage": _coverage(panel, target, channel, dates),
        "data_validity": _check("pass" if event.validity == "ok" else "unknown",
                                "validity_ok" if event.validity == "ok" else "suspect_data",
                                reasons=list(event.validity_reasons)),
        "other_channels": _competing_changes(events, panel, target, channel, pre, post),
    }
    if checks["media_coverage"]["status"] == "pass":
        stable = bool(_inactive(panel, target, channel, pre).all()
                      and (~_inactive(panel, target, channel, post)).all())
        checks["treatment"] = _check("pass" if stable else "fail",
                                     "stable_launch" if stable else "unstable_launch")
    else:
        checks["treatment"] = _check("unknown", "unknown_launch_activity")

    peers = []
    for peer in panel.countries:
        if peer == target:
            continue
        coverage = _coverage(panel, peer, channel, dates)
        # Known activity disqualifies a peer even when other days are missing.
        observed = panel.present_mask(peer, channel).reindex(dates, fill_value=False)
        active = observed & ~_inactive(panel, peer, channel, dates)
        untreated = (_check("fail", "peer_already_active") if active.any()
                     else _check("unknown", "unknown_peer_activity") if coverage["status"] != "pass"
                     else _check("pass", "not_yet_launched"))
        peer_checks = {
            "media_coverage": coverage,
            "untreated": untreated,
            "other_channels": _competing_changes(events, panel, peer, channel, pre, post),
            "sales": _sales_comparison(panel, target, peer, pre, post),
        }
        peers.append({"country_code": peer, "status": _status(peer_checks), "checks": peer_checks})

    if any(p["status"] == "analysis_ready" for p in peers):
        comparison = _check("pass", "qualified_comparison")
    elif any(p["status"] == "needs_review" for p in peers):
        comparison = _check("unknown", "unresolved_comparison")
    else:
        comparison = _check("fail", "no_qualified_comparison")
    checks["comparison"] = comparison
    return {
        "version": 1,
        "status": _status(checks),
        "reason_codes": [c["code"] for c in checks.values() if c["status"] != "pass"],
        "windows": {"pre_start": str(pre[0].date()), "pre_end": str(pre[-1].date()),
                    "post_start": str(post[0].date()), "post_end": str(post[-1].date())},
        "thresholds": {"pre_days": params.LAUNCH_PRE_DAYS, "post_days": params.LAUNCH_POST_DAYS,
                       "active_confirmation_days": params.LAUNCH_ACTIVE_DAYS,
                       "competing_change": params.LAUNCH_COMPETING_CHANGE,
                       "sales_correlation": params.LAUNCH_SALES_CORRELATION,
                       "sales_drift": params.LAUNCH_SALES_DRIFT},
        "checks": checks,
        "peers": peers,
        "qualified_peers": [p["country_code"] for p in peers if p["status"] == "analysis_ready"],
        "interpretation": "Screening for further analysis only; causal comparability and sales effects are not established.",
    }


def verify_staggered_launches(events: list[DetectedEvent], panel: Panel) -> list[DetectedEvent]:
    """Attach auditable readiness evidence without changing detections or scores."""
    return [replace(e, evidence={**e.evidence, "launch_verification": _verify(e, events, panel)})
            if e.event_type == "staggered_launch" else e for e in events]
