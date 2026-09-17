# The Detection Library

This package finds informative periods in marketing spend data: dark periods, single-channel periods, natural holdouts, step changes, channel pulses, and staggered launches. It takes two CSV files and returns a list of labeled events.

This package never reads the benchmark's answer key. It cannot import any of the benchmark's own packages, by name or by path. A test checks this on every run. See `../benchmark/STRUCTURE.md` for why that boundary matters.

## How to run it

```python
from detection.pipeline import run_detection

events = run_detection(media_df, sales_df, sid="dev_001")
```

`media_df` and `sales_df` are the same two frames the benchmark uses. `sales_df` is optional. Detection runs on spend alone. Sales only feeds a later scoring step.

## The pipeline, in order

1. **Build a panel.** `io/panel.py` turns the two input frames into one daily table, indexed by country and channel.
2. **Normalize each series.** `io/normalize.py` turns raw spend into a scale-free series, so a large market and a small market can be compared the same way.
3. **Find the four primitive patterns.** The `primitives/` package looks for off-runs, level shifts, pulse trains, and onsets. Each one works on a single channel's series and knows nothing about event types.
4. **Label each market.** `compose/label.py` turns the primitives into typed events by watching which channels are active over time.
5. **Compare across markets.** `compose/cross_market.py` checks whether an event has a control group elsewhere, and finds channels that turned on late in some markets.
6. **Score and explain.** Each event gets a confidence score, an informativeness score, a validity check, and a plain-language explanation.

`pipeline.py` runs all six steps and returns one list of events.

## Where each event type comes from

| Event type | Found by | Primitive it uses |
|---|---|---|
| `dark_period` | `compose/label.py` | `primitives/zero_runs.py` |
| `single_channel` | `compose/label.py` | `primitives/zero_runs.py` |
| `natural_holdout` | `compose/label.py` | `primitives/zero_runs.py` |
| `step_change` | `compose/label.py` | `primitives/level_shift.py` |
| `channel_pulse` | `compose/label.py` | `primitives/pulse.py` |
| `staggered_launch` | `compose/cross_market.py` | `primitives/onset.py` |

The first five event types come from watching one market on its own. A staggered launch needs to compare markets against each other. `compose/cross_market.py` finds it separately, after the other five.

## The four primitives

Each primitive works on one channel's daily spend and finds a pattern in it. None of them know what event type they will become. That decision happens later, in `compose/`.

* **`primitives/zero_runs.py`** finds runs of days where a channel was off, and decides which runs are long or unusual enough to matter. `compose/label.py` reads these runs to decide between a dark period, a single-channel period, and a natural holdout, based on how many channels went off together.
* **`primitives/level_shift.py`** finds a channel's spend moving to a new level and holding there. It also finds where that new level ends, if it does.
* **`primitives/pulse.py`** finds a channel switching on and off in a regular pattern, and groups the whole pattern into one event.
* **`primitives/onset.py`** finds a channel that stayed off from day one of the data, or stopped and never came back.

## The composition layer

* **`compose/label.py`** watches which channels are active in a market, day by day. Whenever that set of active channels changes, it marks a new period and decides what kind of event that period represents.
* **`compose/cross_market.py`** takes the events `label.py` found and checks each one against the other markets: did a peer market keep the channel running? It also builds staggered launch events by comparing onset dates across markets.

## Scoring and explanation

* **`score.py`** computes a confidence score, how sure the detector is, and an informativeness score, how useful the event is for analysis. Each score comes from several smaller signals.
* **`validity.py`** checks an event for signs of a data problem rather than a real marketing event. Missing rows are one sign. A spend drop with no matching change in impressions is another.
* **`explain.py`** turns one event into a plain-language sentence, stating what happened, when, and how it compares to that channel's normal spend.
* **`calibrate.py`** adjusts the confidence score so it matches how often the detector is actually right. It learns this from results on the development split. `calibration_fit.py` stores the fitted result.

## Two files with no events in them

* **`model.py`** defines `DetectedEvent`, the shape every event takes on its way out of this package. It also lists the six event types by name.
* **`params.py`** holds every threshold and constant this library uses, such as how long an off-run has to be before it counts. Nothing else in this package should hardcode a number that belongs here.

## Two files with no algorithm in them

* **`io/panel.py`** and **`io/normalize.py`** hold no detection logic. They only reshape and rescale the input data so the primitives can compare across channels and markets fairly.
* **`calibration_fit.py`** holds no logic either. It is a generated file: the frozen output of one run of `calibrate.py` against the development split.
