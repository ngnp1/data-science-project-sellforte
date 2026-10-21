# Scoring and ranking

A detector only says *what* it found: market, channel, pattern and dates. This package adds the rest: scores, data checks and an explanation. It also decides the order findings are shown in. Every detector goes through it, so all findings are judged by the same rules.

```python
from detection.scoring import score_findings, ranked

findings = score_findings(findings, build_panel(media, sales))
order = ranked(findings)   # most useful first
```

## Two scores

Each finding gets two scores from 0 to 1. They answer different questions.

| Score | Question | Shown as |
|---|---|---|
| **Informativeness** | How useful is this period to learn from? | Priority: High (0.75 or more), Medium (0.5 or more), Low |
| **Confidence** | How sure are we the pattern is really there? | Evidence: Strong (0.8 or more), Moderate (0.6 or more), Weak |

A finding can be certain but not useful, such as a real 3-day pause that overlaps other changes. That is why the two are kept apart.

**Informativeness** is a weighted sum of six parts:

| Part | Weight | High when |
|---|---|---|
| Duration | 0.25 | The period is long enough to see ads' effect fade after they stop. |
| Comparison | 0.20 | Other markets (1.0) or other channels (0.7) can serve as a comparison. With neither, it is 0.3. |
| Size of change | 0.15 | Spend changed a lot. |
| No overlap | 0.15 | Nothing else changed in that market at the same time. An overlap gives 0.6. |
| Steady sales | 0.15 | Day-to-day sales noise is small, so a change would show. |
| Pattern type | 0.10 | The pattern is usually informative. All ads off scores 1.0 and a budget step 0.45. |

If a period touches the first or last day of the data, the total is multiplied by 0.7, because part of it may be cut off.

**Confidence** is a weighted sum of six checks:
- how completely spend changed;
- whether the period is long enough;
- how unusual it is for that channel;
- how cleanly it starts and stops;
- whether impressions agree with spend;
- for market-wide periods, how many channels agree.

The weights depend on the pattern type.

## Ranking

`ranked()` sorts by informativeness, highest first. Ties go to the higher confidence, and unscored findings go last. A data warning does not move a finding. The app only hides it when **Hide periods with data warnings** is ticked.

## What `score_findings` does, in order

1. Notes how other markets behaved at the same time ([compose/cross_market.py](../compose/cross_market.py)).
2. Flags periods at the edge of the data and periods that overlap another finding.
3. Computes confidence and informativeness ([score.py](score.py)).
4. Checks data quality: `ok` or a warning with reasons ([validity.py](validity.py)).
5. For late launches, checks whether other markets can serve as a comparison ([compose/staggered_launch/readiness.py](../compose/staggered_launch/readiness.py)).
6. Writes a plain-language explanation ([explain.py](explain.py)).

## Files

| File | Job |
|---|---|
| [findings.py](findings.py) | `score_findings`: runs the steps above. |
| [score.py](score.py) | Confidence and informativeness. |
| [validity.py](validity.py) | Data-quality checks. |
| [explain.py](explain.py) | Explanations. |
| [rank.py](rank.py) | `ranked`: the display order. |

Weights and limits are in [../params.py](../params.py), under `CONFIDENCE_WEIGHTS`, `INFORMATIVENESS_WEIGHTS`, `TYPE_PRIOR` and `CONTROL_SCORE`. The Priority and Evidence labels are in [ui/labels.py](../../ui/labels.py).

Scores are rules of thumb, not probabilities. They do not measure what the advertising caused.
