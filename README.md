# Finding useful periods in marketing data

This school project finds changes in daily advertising spend that may help someone study marketing performance. It includes a detector, a local viewer, and synthetic data with known events for testing.

| Event | What it means |
|---|---|
| Dark period | All channels in a market pause. |
| Single-channel period | Only one channel keeps running. |
| Natural holdout | One channel pauses while others continue. |
| Step change | A channel's budget moves to a different level. |
| Channel pulse | A channel pauses several times, with spending between pauses. |
| Staggered launch | A channel starts later in one market than in others. |

## Run the viewer

Use Python 3.12, the version tested for this project. Run these commands from the repository folder:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py --server.address 127.0.0.1
```

The viewer opens with the included sample, on one page read from top to bottom:

1. **Summary and period map.** How many findings there are, and a timeline with one row per market showing where each finding sits, labelled with what happened and for how long. Click a bar to inspect it.
2. **Ranked findings.** The three most informative findings appear as cards, and the rest follow in a table. Click a card's *Inspect* button or a table row to inspect a finding. *Download findings* saves the table as CSV.
3. **Inspect panel.** What the period lets you measure, then four tabs. *Readout* shows a rough sales readout, then sales compared with other markets with the reference and finding averages marked, above one weekly spend strip per channel with the finding's own channels highlighted; pulses also get a chart of sales around each pause. *Why flagged* shows the scores in plain words. *Caveats* lists what to keep in mind, and *Explanation* shows the detector's own text. A data-quality warning always shows above the tabs.

Use the sidebar to filter by market or pattern type, hide findings with data warnings, or upload your own CSV files.

The look comes from [shadcn/ui](https://ui.shadcn.com) components, through the `streamlit-shadcn-ui` package, and a matching theme in `.streamlit/config.toml`.

## Use your own data

| File | Required columns |
|---|---|
| Media CSV | `date`, `country_code`, `advertising_channel`, `media_investment` |
| Sales CSV (optional) | `date`, `country_code`, `turnover` |

Use dates such as `2024-06-24`. Media spend must be a finite, non-negative number; missing or invalid spend values are rejected. Media can also include `impressions` and `clicks`.

Multiple campaign rows are summed for each day, market, and channel. Missing daily rows are flagged because a missing record does not prove that advertising stopped. Event start and end dates both belong to the reported window.

## Plug in another detector

The app and the evaluation harness run any detector registered in `detectors/`. Our rule-based library is one implementation (`detectors/rule_based.py`). To add another, create `detectors/my_detector.py`:

```python
from detectors.base import Detector

class MyDetector(Detector):
    key = "my_detector"
    name = "My detector"
    event_types = frozenset({"dark_period", "natural_holdout"})

    def detect(self, media, sales):
        ...  # return a list of detection.model.DetectedEvent

DETECTORS = [MyDetector()]
```

Then add `"detectors.my_detector"` to `MODULES` in `detectors/registry.py`. A test fails if a module in `detectors/` is missing from that list, so a detector cannot be forgotten.

A finding must name its market, type, channel (none for a dark period) and inclusive dates, plus the off-windows for a pulse. Scores and explanations are optional: the app shows "Unscored" and says when a detector gives no scores or explanation. Data-quality checks are optional too, but a finding's `validity` defaults to "ok", so a detector that does not check data quality has its findings shown as OK. Once two or more detectors are registered, a **Detector** picker appears in the sidebar. To score a detector on the development split, add `detect = for_detector("my_detector")` (from `benchmark.eval.adapter`) to a module and run `python -m benchmark.eval.run_dev --detector mymodule:detect --load-data`. The truth-isolation tests scan every registered detector's source, and each run record hashes `detectors/` as well as `detection/`.

## Check the results

With the virtual environment active, run:

```bash
python -m scripts.evaluate_sample
python -m pytest -q -m "not slow"
```

The revised detector finds all six events in the included sample, with no extra findings and exact pulse windows. Precision, recall, and F1 are all 1.0 on this sample. Because this is a known example used during development, it does **not** show how well the detector handles unseen data.

The full benchmark CSVs are not included. Tests needing those files or R skip when they are unavailable. Running the viewer and sample checks does not require R. The saved final benchmark results describe an older detector, not the current version.

## What the scores mean

Each event has two scores from 0 to 1:

- **Confidence:** how strongly the data matches the detected pattern. A score of 0.8 does not mean an 80% chance of being correct.
- **Informativeness:** how useful the period might be for further analysis. Longer, clearer periods with possible comparison markets generally score higher.

These are rule-based scores. They do not measure the sales caused by advertising. A comparison market is only a candidate; the detector does not prove it is a fair control group.

The viewer shows each band, not the raw score: priority is *High* from 0.75 and *Medium* from 0.5; evidence is *Strong* from 0.8 and *Moderate* from 0.6.

## How the viewer reads sales

For each finding the viewer compares the market's average daily sales during the period with a reference window. That window is the same number of days just before the period, up to 90 days. When the data starts too late for that, it uses the days just after the period. A pulse uses its own on-days between pauses. The same before/during ratio in **comparison markets** is divided out: comparison markets are other markets with sales data and no finding of their own over those weeks, weighted equally whatever their size. What is left is the change the other markets did not share.

To judge whether that change stands out, the viewer repeats the same calculation at every other weekly offset in the history. The typical size of those results is the **typical swing**. A change smaller than twice the typical swing is shown as *unclear*. When the change is readable and spend moved by at least 10%, the viewer also shows a **rough return**: the extra sales per extra unit of spend. It withholds the return when other channels' spend moved too, when sales moved the opposite way to spend, or when another finding overlaps the period. The reference window also avoids other findings in the same market. For a period with all advertising off, the sales during the period are shown as the **baseline**.

These are rough readings, not causal estimates. Averages over the whole window understate a pause's effect, because advertising keeps working for a while after it stops. On the included sample, the rough returns land within a factor of two of the simulator's true ROI. On harder synthetic data they can be much further off, so treat a rough return as a first look, not a measurement.

Pauses shorter than seven days are not reported. Events touching the start or end of the data may continue beyond what we can see. Overlapping events are harder to interpret. Long periods of very low spend can distort the estimate of normal spend, and an initially inactive channel may be a late launch or simply have missing earlier history. Real company data and the usefulness of the ranking still need validation.

## Read more

- [How detection works](detection/README.md)
- [Benchmark data and setup](benchmark/BENCHMARK.md)
- [How evaluation works](benchmark/eval/README.md)
- [Generate a new sample](synthetic_data_generator/README.md)
