# Scoring

The detector finds changes in advertising spend. It gives each finding two
scores: **Priority** and **Evidence**.

## Priority

Priority ranks findings by how useful they are to investigate.

| Criterion | Weight |
|---|---:|
| Long enough to see after-effects | 25% |
| Comparison available | 20% |
| Size of the spend change | 15% |
| No overlapping findings | 15% |
| Sales moved clearly | 15% |
| Pattern type | 10% |

- **Long enough to see after-effects**: Longer periods make it easier to see
  how long advertising continues to work. Full credit starts at 28 days.
- **Comparison available**: Another market or channel ran normally during the
  finding.
- **Size of the spend change**: Larger changes are easier to study.
- **No overlapping findings**: Another change in the same market makes the
  result harder to interpret.
- **Sales moved clearly**: Sales during the finding differ from the previous
  eight weeks by more than normal sales variation. This does not show that
  advertising caused the change.
- **Pattern type**: Some patterns usually give a clearer comparison. For
  example, all advertising off is usually easier to study than a budget step.

The app labels Priority as **High** at 0.75 or above, **Medium** from 0.50 to
0.74, and **Low** below 0.50. A finding that touches the first or last day of
the data gets lower Priority because it may continue outside the data.

## Evidence

Evidence measures how clearly the spend data supports the finding. The weights
depend on the type of finding.

| Criterion | All ads off | Only one on | One paused | Pulsing | Late launch | Budget step |
|---|---:|---:|---:|---:|---:|---:|
| How completely spend changed | 20% | 20% | 25% | 25% | 20% | 45% |
| Long enough | 15% | 15% | 20% | 10% | 20% | 20% |
| Unusual for this channel | 15% | 15% | 25% | 30% | 20% | 0% |
| Clean start and stop | 10% | 10% | 10% | 15% | 10% | 25% |
| Impressions agree | 15% | 15% | 20% | 20% | 30% | 10% |
| All channels agree | 25% | 25% | 0% | 0% | 0% | 0% |

- **How completely spend changed**: A full stop is stronger evidence than a
  small reduction. A larger budget step is also stronger evidence.
- **Long enough**: A longer change is less likely to be a short blip. Full
  credit starts at 14 days.
- **Unusual for this channel**: A rare pause is more meaningful than a pause
  that happens often.
- **Clean start and stop**: A sharp change is clearer than a gradual drift.
- **Impressions agree**: Impressions should stop when spend stops. Continuing
  impressions can indicate a tracking problem.
- **All channels agree**: For market-wide findings, the channels that should
  be off need to be off.

The app labels Evidence as **Strong** at 0.80 or above, **Moderate** from 0.60
to 0.79, and **Weak** below 0.60.

## Warnings

Warnings do not remove a finding.

- **Possible data gap**: Data rows are missing during the finding, or every
  channel in every market stopped at once.
- **Possible tracking loss**: Impressions continued while spend was off or
  nearly off.

Priority and Evidence help you choose what to investigate. They do not prove
that advertising caused a sales change.
