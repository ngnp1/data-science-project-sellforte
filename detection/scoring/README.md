# Scoring and ranking

A detector only reports *what* it found: a market, a channel, a pattern and some dates. This package then asks two questions about each finding:

1. **Is it real?** Did spend actually change the way the detector says? The app shows this as **Evidence**.
2. **Is it useful?** Would the period teach us something about what advertising does? The app shows this as **Priority**, and ranks by it.

A finding can be real but not useful. A clear 3-day pause is one example: it is too short to show anything. That is why there are two scores.

Every detector's findings go through this package, so all of them are judged the same way.

```python
from detection.scoring import score_findings, ranked

findings = score_findings(findings, build_panel(media, sales))
order = ranked(findings)   # most useful first
```

## How a score is built

Each score is a set of checks. Every check gives a number from 0 (bad) to 1 (good). The score is a weighted average of the checks, so it is also between 0 and 1. In the app, open a period's **Details** to see each check's value.

## Is it useful? (Priority)

| Check | What it asks | Full marks (1.0) when | Weight |
|---|---|---|---|
| **Long enough** | Is the period long enough to see the effect of ads fade after they stop? | It lasts 4 weeks or more. Two weeks gets 0.5. For on–off pulsing, the shortest pause counts. | 25% |
| **Comparison** | Is there something to compare it with? | Another market kept the same channel running. If not, another channel in the same market kept running: 0.7. If neither: 0.3. | 20% |
| **Size of change** | How big was the change in spend? | Spend dropped to zero, or the budget tripled (or fell to a third). | 15% |
| **No overlap** | Did something else change in that market at the same time? | Nothing overlapped. With an overlap: 0.6. | 15% |
| **Sales moved** | Did sales move clearly during the period? It compares average daily sales with the 8 weeks before, measured against how much sales normally go up and down. | Sales moved at least 3 times their usual ups and downs. With no sales data, or less than 8 weeks of history: 0.5. | 15% |
| **Pattern type** | How useful is this kind of pattern in general? | All advertising off: 1.0. Only one channel on, or pulsing: 0.9. One channel paused: 0.75. Staggered launch: 0.6. Budget step: 0.45. | 10% |

If a period starts on the first day of the data, or ends on the last day, its score is multiplied by 0.7. It may have started earlier or kept going, so part of it is unseen.

The **Priority** label is **High** at 0.75 or more, **Medium** at 0.5 or more, otherwise **Low**.

## Is it real? (Evidence)

| Check | What it asks | Full marks (1.0) when |
|---|---|---|
| **How completely spend changed** | Did spend really stop, or only dip? | Spend fell to exactly zero. For a budget step: the jump was far outside the channel's normal range. |
| **Long enough** | Could this be a short blip? | It lasts 2 weeks or more. |
| **Unusual for this channel** | Does this channel pause like this all the time? | The pause is at least 3 times longer than the channel's usual pauses, or the channel never pauses otherwise. |
| **Clean start and stop** | Did spend switch off and on sharply, or drift? | Spend was at its normal level right up to the start and from the day after the end. |
| **Impressions agree** | Did impressions stop too? If impressions kept coming, the spend data may simply be missing. | Impressions fell to zero as well. If impressions kept coming: 0.2. With no impression data: 0.6. |
| **All channels agree** | For "all advertising off" and "only one channel on": did every channel that should be off really stop? | All of them did. Other patterns are not checked for this. |

The weights depend on the pattern type, because each pattern leaves different evidence. A budget step leans most on how big the jump was (45%). Pulsing leans most on being unusual for the channel (30%). The exact weights are in `CONFIDENCE_WEIGHTS` in [params.py](../params.py).

The **Evidence** label is **Strong** at 0.8 or more, **Moderate** at 0.6 or more, otherwise **Weak**.

## The order in the app

`ranked()` puts the most useful findings first. When two are equally useful, the one with stronger evidence comes first. Findings with no scores go last.

Data warnings don't change the order. Tick **Hide periods with data warnings** in the sidebar to hide them.

## The Sales change column is separate

The **Sales change** column in the app is not part of either score and does not affect the order. It is calculated in [ui/readout.py](../../ui/readout.py) like this:

1. Compare the market's average daily sales during the period with the same number of days just before it. If those days are taken by another finding, it uses the days just after.
2. Do the same for other markets that had nothing unusual going on in those weeks, and take their average.
3. The Sales change is how much more, or less, the market changed than the other markets did. This strips out things that hit every market, such as seasons or holidays.
4. If the change is smaller than twice the market's usual week-to-week swing, the app calls it small. It may be chance.

How it differs from the **Sales moved** check:
- **Sales moved** compares the market with its own past.
- **Sales change** compares the market with other markets.

To view periods by sales impact, set **Sort by** above the table to **Biggest sales change**. Rank still shows the usefulness order.

## Files

| File | What it does |
|---|---|
| [findings.py](findings.py) | `score_findings`: adds the scores, data-quality checks and an explanation to each finding. |
| [score.py](score.py) | The checks behind both scores. |
| [validity.py](validity.py) | Data-quality checks, such as missing days. |
| [explain.py](explain.py) | The plain-language explanation. |
| [rank.py](rank.py) | `ranked`: the order findings are shown in. |

To change a weight or a limit, edit [params.py](../params.py). The Priority and Evidence labels are set in [ui/labels.py](../../ui/labels.py).

These scores are rules of thumb, not probabilities. They don't say how much sales the advertising caused.
