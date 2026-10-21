# Staggered-launch backend baseline: development evaluation

## Scope and provenance

The evaluation completed on 2026-10-18 at 11:28:55 UTC, using detector commit
`a48943c` and detection source SHA-256
`5f623ca6c93e35a3ea33cd982193fffc1343a831e9dc3eda1b05e794c77ea491`.

All 43 selected development scenarios completed evaluation: `dev_001` through
`dev_033`, and `dev_036` through `dev_045`. The configured development split
contains 45 scenarios. Generation of `dev_034` and `dev_035` was stopped at the
user's request; these two scenarios are explicitly outside this evaluation.
Generation retries and cancellations are operational history, separate from
detector false positives and false negatives.

Inputs were generated with the existing configured seeds. No detection
parameters were tuned for this run. These are development results on synthetic
data, not a held-out test evaluation or validation on company data. No held-out
test evaluation was performed in this work, and historical test records/seals
were preserved.

## Staggered-launch detection results

| Metric | Result |
|---|---|
| Correct detections (TP) | 10 |
| False alarms (FP) | 1 |
| Missed launches (FN) | 0 |
| Precision | 90.91% |
| Recall | 100.00% |
| F1 | 95.24% |
| Matched interval start/end median absolute error | 0 / 0 days |
| Matched interval start/end 90th-percentile absolute error | 0 / 0 days |

Timing errors cover the 10 matched launches only. Event intervals represent the
initial inactive period; the activity onset is the following day.

The false alarm is `dev_036`, US / Google Discovery, with detected inactive
interval 2024-01-01 through 2024-02-29 and activity onset 2024-03-01. The scenario
labels this as a natural holdout censored at the observation start, while the
detector identifies the observed spending pattern as a staggered launch.
Classification remains unchanged. Ranking retains this candidate as well.

## Readiness screening, evaluated separately

| Outcome | Detected launches |
|---|---|
| Analysis-ready | 2 |
| Needs review | 0 |
| Not analysis-ready | 9 |

| Top-level rejection or unresolved-evidence reason | Count |
|---|---|
| No qualified comparison | 6 |
| Insufficient history | 3 |
| Missing media rows | 3 |
| Missing other-channel rows | 3 |
| Unknown launch activity | 3 |
| Unresolved comparison | 3 |

Reasons overlap and should not be summed as numbers of launches. The two ready
launches are Germany and Sweden in `dev_023`. Launch labels establish detection
correctness; they do not establish readiness accuracy. The readiness thresholds
are provisional screens, not proof of causal comparability or estimates of lift.

## Backend additions and handoff

The launch module exports `rank_staggered_launches(events)` and
`summarize_staggered_launches(events)`. The first retains all launches and places
ready launches before review cases, known rejections, and unassessed launches.
Existing informativeness and confidence break ties, followed by stable event
identifiers. The second supplies JSON-compatible, ranked records with actual
onsets, inactive intervals, readiness reasons, qualified comparisons, windows,
and existing scores for a future UI or other consumer.

This is an explicit analysis priority policy. It changes neither detection
correctness nor the numeric scores. The policy is not validated ranking
accuracy or a calibrated probability of usefulness. Classification, shared
scoring, readiness thresholds, pipeline ordering, and the current UI are
unchanged. No new model, weights, or change-point algorithm are introduced.

### Verification on 2026-10-21

The new report command completed the same 43 scenarios without regenerating
inputs. All 66 detector event outputs were identical to the saved baseline,
including classifications, dates, numeric scores, explanations, and readiness
evidence. Both launch-specific and overall detection metrics were unchanged.

All 11 detected launches remain in the ranked export. The two analysis-ready
launches, `dev_023` / Germany and Sweden, occupy positions 1 and 2. This confirms
the ordering policy was applied; it is not an independent usefulness assessment.

The rerun began at 17:32:15 UTC using Python 3.13.3, commit `a48943c`, and a dirty
working tree containing these additions. Detection source SHA-256 was
`013138b2de063c136211e2a54ba6a987023e2e0113f93d863510d5f177d0276f`;
evaluation source SHA-256 was
`124227a944df993c09f07ed2b8d71c042e777c30db25da6ee2200d808c51c5c3`.
The source hash changes because of the new backend helpers, not detection-rule
changes. The saved comparison is local at
`.cache/staggered-launch-dev/baseline-comparison.json`.

Focused and regression checks: 120 passed, 15 skipped because required generated
fixtures were unavailable, and one existing Windows path-separator assertion
was explicitly deselected. Detector isolation and source-only checks passed.

See the [launch module documentation](../../detection/compose/staggered_launch/README.md)
for backend usage and field meanings.

## Reproduce on the existing development data

From the repository root, with the existing Python dependencies installed:

```powershell
python -m benchmark.eval.run_staggered_dev --exclude dev_034 dev_035 --out .cache/staggered-launch-dev
```

The command checks every selected scenario against the configured development
split before evaluation. Missing media, sales, or required answer files cause an
explicit error; nothing is silently skipped or generated. Without `--exclude`,
all configured development scenarios are required. No test split option exists.

The output directory contains:

- `metrics.json`: launch detection, readiness counts/reasons, and detailed
  existing evaluator results for all patterns;
- `launch-summary.json`: ranked launch records;
- `run-metadata.json`: selected/excluded IDs, timestamp, commit, working-tree
  state, detector/evaluator source hashes, Python version, and elapsed time;
- `predictions.jsonl`: raw event outputs with readiness evidence;
- `report.md`: readable launch metrics, misses/false alarms, and ordered results;
- `run-history.jsonl`: append-only history local to that output directory.

The command does not append to tracked development/final histories or modify
scenario definitions, generated inputs, or test seals. Commit plus source hashes
and working-tree state distinguish the original baseline from later runs that
include the new backend helpers. Regeneration with different dependencies may
produce different inputs; reusing the existing generated data is required for
the direct baseline comparison.
