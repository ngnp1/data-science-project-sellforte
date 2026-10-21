# Staggered launches: detection, analysis readiness, and priority

The goal is to find a channel starting at different times across markets, then
check whether each late start offers a useful opportunity for Sellforte to study
marketing performance. Readiness is a screening decision, not proof of a fair
causal comparison and not an estimate of sales lift or ROI.

## 1. Detect the spending pattern

`detection/primitives/onset.py` finds initial inactive periods; this folder's
`detector.py` compares markets using the shared peer-comparison helper in
`../cross_market.py`. The public entry point remains `pipeline.run_detection()`.

- Campaign rows are summed into daily market/channel spend, with missing rows
  tracked separately from explicit zeros.
- The existing inactivity rule is spend at most the larger of `1e-6` and 15% of
  the channel's median positive daily spend.
- A late-start candidate needs an initial inactive period containing at least
  seven observed inactive days, followed immediately by 14 consecutive observed
  active days. Missing confirmation rows and brief trial campaigns do not pass.
- A market active for the first 14 observed calendar days can establish the
  earliest start. An unconfirmed spike cannot establish it. Markets never buying
  the channel do not establish an earlier start.
- Each reported market must start at least 14 days after the earliest qualifying
  start. The earliest market itself is not reported as a staggered launch.
- One event is returned per late-starting market. Its inclusive `start`/`end`
  describe the initial inactive period, ending the day before launch. The actual
  activity start is `evidence["onset"]`.

Evidence also records `observed_inactive_days`, `active_confirmation_days`,
`onset_separation_days`, `earliest_market_onset`, and `n_markets`. Incomplete
initial history remains a warning: this can be a late start within the observed
data, but earlier advertising or a restart cannot be ruled out.

## 2. Verify the opportunity for analysis

`readiness.verify_staggered_launches(events, panel)` returns annotated
events after the existing validity and scoring stages. It leaves event identity,
confidence, validity, and the generic informativeness score unchanged. Sales may
change readiness but never which launches are detected.

All windows are anchored on the actual launch, not the event's initial start:

| Check | Default requirement |
|---|---|
| History | 56 days before launch and 28 days from launch onward, including launch day. Never shorten these windows automatically. |
| Target activity | Observed inactivity throughout the pre-period, then continuous observed activity throughout the 28-day post-period. This is stricter than the 14-day detection confirmation. |
| Coverage | Complete daily media records for the target channel and assessed other channels in both markets. Missing rows cannot prove inactivity. |
| Data validity | Existing target validity is `ok`; suspect data requires review. |
| Comparison | A different market remains inactive in this channel throughout both windows. Already-active markets do not qualify. |
| Other channels | No detected event affecting another channel overlaps the analysis window; no other channel's median weekly spend changes by 25% or more between pre and post. |
| Sales coverage | Complete finite, non-negative daily sales in both markets throughout both windows. |
| Pre-launch sales | Eight complete seven-day sales totals per market, normalized by each market's own mean. Correlation must be at least 0.8 and relative drift at most 0.1. |

Weeks are consecutive seven-day blocks anchored to the analysis window, not
calendar-week resampling. Relative drift is the absolute least-squares slope of
the difference between the two normalized weekly sales series, multiplied by
seven weekly intervals. Flat or zero-mean pre-sales make this check inconclusive.
The assessment also records pre-period weekly variability (`std / mean`). It
does not test whether post-launch sales increased: well-observed unchanged or
lower sales can still be informative.

Every peer is assessed separately. Explicit zero records can establish an
untreated comparison even when the peer never buys the channel. A peer launching
after the entire post-period can qualify; launching on its last day cannot.
The earlier active market used to establish a staggered pattern is **not** the
untreated comparison required by this screen. Consequently, the last adopter in
a rollout may have no eligible comparison and fail readiness despite being a
correct detection.

The checks for unrecorded channel rows cover channels represented in the input;
the code cannot discover an advertising channel missing from the export entirely.

## 3. Read the assessment

The versioned, JSON-compatible result is attached to each launch as
`event.evidence["launch_verification"]`. No fields were added to `DetectedEvent`.

| Outcome | Meaning |
|---|---|
| `analysis_ready` | All target checks pass and at least one peer passes all comparison checks. |
| `needs_review` | No known target failure, but required evidence is missing or inconclusive and at least one peer could still qualify. |
| `not_analysis_ready` | A target check fails, or all peers demonstrably fail (including no peers at all). |

A known failure takes precedence over unknown evidence within one assessment.
Across peers, a passing peer wins; otherwise an unresolved peer takes precedence
over peers known to fail. All peer checks are retained, including failed peers.

The assessment contains:

- `version`, `status`, and top-level `reason_codes`;
- exact pre/post `windows` and the `thresholds` applied;
- target `checks` and individual `peers` with their own checks and reason codes;
- `qualified_peers` and an explicit interpretation limit.

Check statuses are `pass`, `fail`, or `unknown`. Examples of reason codes are
`insufficient_history`, `missing_media_rows`, `unstable_launch`,
`peer_already_active`, `competing_changes`, `missing_or_invalid_sales`,
`flat_prelaunch_sales`, and `divergent_prelaunch_sales`. An overall
`no_qualified_comparison` or `unresolved_comparison` is explained by the detailed
peer checks. Existing `control_available` evidence describes active peers during
initial dormancy; it is separate from the untreated peers in this assessment.

For example, for an onset of 2024-03-11, pre-launch dates are 2024-01-15 through
2024-03-10 and post-launch dates are 2024-03-11 through 2024-04-07. A matching
peer that first starts on 2024-04-08 can qualify. With full clean data and similar
pre-sales, the result is `analysis_ready`, with that peer in `qualified_peers`.

```python
from pprint import pprint
from detection.pipeline import run_detection

events = run_detection(media_df, sales_df, sid="sample")
for event in events:
    if event.event_type == "staggered_launch":
        pprint(event.evidence["launch_verification"])
```

The existing viewer's evidence display exposes this structure. Launch explanations
include the readiness outcome; the existing generic informativeness score remains
a separate heuristic and should not be interpreted as readiness or causal impact.

## 4. Rank and export launches

The backend provides an explicit launch-only shortlist for downstream consumers:

```python
import json
from detection.compose.staggered_launch import (
    rank_staggered_launches,
    summarize_staggered_launches,
)

# events is the existing output of run_detection(media, sales, sid).
launches = rank_staggered_launches(events)
rows = summarize_staggered_launches(events)
payload = json.dumps(rows, allow_nan=False)
```

Both functions accept mixed event types and select every staggered launch,
including those rejected by readiness. Priority is `analysis_ready`, then
`needs_review`, then `not_analysis_ready`, then `unassessed`. Missing or
unrecognized readiness evidence is unassessed, never assumed to pass.

Within each group, existing informativeness and then confidence sort descending.
Missing or non-finite scores sort last at their respective sort position.
Scenario, market, channel, and event dates break ties deterministically. The
ordering is an analysis triage policy, not a probability or a validated measure
of usefulness. It does not correct limitations of the generic informativeness
score or add new weights. In particular, that score's sales driver requires
history before the initial inactive interval and can default even when there
is enough history before the actual launch.

`rank_staggered_launches` returns the original event objects without modifying
them. `summarize_staggered_launches` returns independent JSON-compatible records:

- `rank` (one-based), `sid`, `country_code`, and `channel`;
- `launch_date`, `inactive_start`, and `inactive_end` as ISO date strings;
- `readiness`, `reason_codes`, and `qualified_peers`;
- `windows` containing `pre_start`, `pre_end`, `post_start`, and `post_end`;
- existing `detection_confidence` and `informativeness` scores.

Missing launch dates/windows and non-finite scores export as `null`. An onset
is never inferred from the initial inactive interval. These helpers preserve
classification, readiness evidence, original scores, and the public pipeline's
existing output order. The shared viewer continues using its existing ranking.

## 5. Development results and reproducibility

The [tracked development report](../../../benchmark/reports/staggered-launch-dev-2026-10-18.md)
records the completed 43-scenario baseline, explicit exclusions, detection
metrics, readiness outcomes, and the command to reproduce it. Detailed outputs
and append-only run history stay under the ignored local cache. The report
command requires complete data for every selected development scenario; it
never generates missing data or silently reduces coverage.

## 6. Validation and limitations

All screening thresholds live in `params.py` and are provisional defaults, not
values validated by Sellforte. Tests cover sustained activity, onset separation,
window boundaries, missing records, comparison eligibility, competing changes,
market-size differences, sales comparability, and detection independence from sales.

```bash
python -m pytest tests/detection/staggered_launch tests/detection/test_onset.py -q
python -m pytest -q -m "not slow"
```

Use generated development data when available; never tune against the sealed
final benchmark. Detection precision/recall does not measure readiness accuracy.
Readiness needs independent review on real company data.

Before using a ready launch, an analyst should confirm channel availability and
history, review the selected comparison and exact windows, and investigate
promotions, prices, holidays, spillovers, and external market changes. These
factors are not supplied by the current media/sales schema. Correlation and a
small drift do not establish parallel counterfactual trends. No causal effect,
confidence interval, statistical power, adstock model, or ROI is estimated.
