# How the detector works

The detector reads daily media spend and returns a list of findings. Sales data is optional: it only helps rank findings.

```python
import pandas as pd
from detection.pipeline import run_detection

media = pd.read_csv("synthetic_data_generator/data/media.csv")
sales = pd.read_csv("synthetic_data_generator/data/sales.csv")
for event in run_detection(media, sales, sid="sample"):
    print(event.event_type, event.country_code, event.channel, event.start, event.end)
```

Input columns are listed in the [main README](../README.md#use-your-own-data).

## Steps

1. **Prepare daily data.** Sum campaign rows by market, channel and date, and remember which rows were missing.
2. **Estimate normal spend** per channel from its positive spend, so the same rules work for small and large markets.
3. **Find changes:** pauses, budget steps, repeated pauses and late starts.
4. **Label them** by checking which other channels kept running.
5. **Compare markets** to find peers that kept running, and to compare launch dates.
6. **Score and explain** each finding, and flag possible data problems.

## Main rules

| Pattern | Rule |
|---|---|
| Pause | Spend at most 15% of the channel's usual level, for at least seven days. |
| Budget step | A sharp, lasting change in level. A step that never reverses must also differ from the channel's last quarter of active spend, so a return to normal after a pause or a short bump is not reported. |
| Pulse | Nearby pauses of similar length, grouped into one finding. |
| Late launch | An initial inactive period, compared with other markets. |

With only two channels, one pausing while the other runs is labelled a holdout. From spend alone this cannot be told apart from "only one channel on", so the choice is deliberate.

Thresholds are in [params.py](params.py) and the pattern rules in [primitives/](primitives/).

## A finding

A [DetectedEvent](model.py) has a type, market, channel (`None` for a dark period), inclusive start and end dates, scores, data-quality flags and an explanation. For a pulse, `components` lists each pause. `censored_start` and `censored_end` mean the period touches the edge of the data.

Scores are heuristics, not probabilities, and not estimates of what advertising caused.

## Code map

| Files | Job |
|---|---|
| [pipeline.py](pipeline.py) | Runs everything. |
| [io/](io/) | Prepares and rescales the data. |
| [primitives/](primitives/) | Finds individual patterns. |
| [compose/](compose/) | Labels findings and compares markets. |
| [score.py](score.py), [validity.py](validity.py), [explain.py](explain.py) | Scores, data checks, explanations. |

`detection/` must never import the benchmark or its answers. A test checks this.
