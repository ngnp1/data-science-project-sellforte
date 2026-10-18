# How the detector works

The detector reads daily media spend and returns a list of events. Sales data is optional: it helps rank findings but is not needed to find spend changes.

## Use it in Python

After following the [project setup](../README.md), run this from the repository folder:

```python
import pandas as pd
from detection.pipeline import run_detection

media = pd.read_csv("synthetic_data_generator/data/media.csv")
events = run_detection(media, sid="sample")

for event in events:
    print(event.event_type, event.start, event.end, event.explanation)
```

To include sales, load its CSV and pass it as `run_detection(media, sales, sid="sample")`. See the [input columns](../README.md#use-your-own-data).

## From data to findings

1. **Prepare daily data.** Sum campaign rows by market, channel, and date. Keep track of missing rows so they can trigger warnings.
2. **Estimate normal spend.** Use each channel's positive spend to estimate its usual level. This lets the same rules work across markets of different sizes.
3. **Find changes.** Look for pauses, sustained budget changes, repeated pauses, and channels that start late.
4. **Assign event types.** Check which other channels stayed active. For example, all channels pausing together becomes a dark period. With two channels, a pause in one is labelled a natural holdout.
5. **Compare markets.** Look for peers that kept the relevant channels active throughout the event, and compare launch dates.
6. **Score and explain.** Return the finding, its evidence, possible data problems, and a readable explanation.

## The main rules

| Pattern | How it is found |
|---|---|
| Pause | Spend is at most 15% of the channel's usual positive level, with a small floor for values close to zero. Runs must meet duration and notability rules. |
| Budget change | Compare spending levels around a possible change and check that the new level lasts. Restarting from zero alone is not enough. |
| Pulse | Group nearby pauses of similar length. The rule does not require perfectly regular spacing. Each off-window is stored separately. |
| Late launch | Find an initial inactive period followed by activity, then compare it with other markets. |

The minimum reportable duration is seven days. Exact thresholds live in [params.py](params.py); the pattern rules live in [primitives/](primitives/).

## Read an event

Each [DetectedEvent](model.py) includes its type, market, channel, dates, scores, explanation, and validity warnings. A dark period has no single channel. Both dates are inclusive. For pulses, `components` contains the individual off-windows; `start` and `end` cover the whole group.

Confidence and informativeness are heuristic scores, not probabilities or estimates of advertising's effect on sales. The old calibration is retained for reference but is not applied.

A `censored_start` or `censored_end` flag means the data does not show the event's full boundary. For example, a permanent budget change is reported through the last available day, without claiming it stopped then. Overlapping events reduce informativeness because their effects may be difficult to separate.

Missing rows or a spend pause without a corresponding drop in available impressions can trigger validity warnings. Unknown, infinite, or negative spend values are rejected. Optional missing metrics are treated as unknown evidence.

## Where to make changes

| Files | Responsibility |
|---|---|
| [pipeline.py](pipeline.py) | Runs detection from beginning to end. |
| [io/](io/) | Prepares and rescales daily data. |
| [primitives/](primitives/) | Finds individual patterns. |
| [compose/](compose/) | Labels events and compares markets. |
| [score.py](score.py), [validity.py](validity.py), [explain.py](explain.py) | Ranks findings, checks data quality, and writes explanations. |

Keep detection independent of benchmark definitions and answer files. Tests check this code boundary. See the [project limitations](../README.md#what-the-scores-mean) before interpreting results.
