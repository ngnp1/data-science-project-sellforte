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

The viewer opens with the included sample. Filter the findings by market or event type, then select an event to see its chart and explanation. You can also upload CSV files and download the findings.

## Use your own data

| File | Required columns |
|---|---|
| Media CSV | `date`, `country_code`, `advertising_channel`, `media_investment` |
| Sales CSV (optional) | `date`, `country_code`, `turnover` |

Use dates such as `2024-06-24`. Media spend must be a finite, non-negative number; missing or invalid spend values are rejected. Media can also include `impressions` and `clicks`.

Multiple campaign rows are summed for each day, market, and channel. Missing daily rows are flagged because a missing record does not prove that advertising stopped. Event start and end dates both belong to the reported window.

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

Pauses shorter than seven days are not reported. Events touching the start or end of the data may continue beyond what we can see. Overlapping events are harder to interpret. Long periods of very low spend can distort the estimate of normal spend, and an initially inactive channel may be a late launch or simply have missing earlier history. Real company data and the usefulness of the ranking still need validation.

## Read more

- [How detection works](detection/README.md)
- [Benchmark data and setup](benchmark/BENCHMARK.md)
- [How evaluation works](benchmark/eval/README.md)
- [Generate a new sample](synthetic_data_generator/README.md)
