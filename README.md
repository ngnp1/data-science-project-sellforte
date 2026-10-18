# Finding informative periods in marketing data

Marketing data is hard to read because channels usually run together. This tool scans daily ad spend and finds the short periods that work like accidental experiments, then shows them in an app you can explore.

| Pattern | What happened |
|---|---|
| All advertising off (dark period) | Every channel in a market paused. |
| Only one channel on | One channel kept running while the others stopped. |
| One channel paused (natural holdout) | One channel stopped while the others continued. |
| Budget step | A channel's spend jumped to a new level. |
| On–off pulsing | A channel paused several times, with spend in between. |
| Staggered launch | A channel started later in one market than in others. |

## Run the app

You need Python 3.12. From the repository folder:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app opens in your browser with an included sample dataset. You do not need R.

## Use the app

The page reads from top to bottom:

1. **Where the periods are.** A timeline with one row per market. Each bar is a finding, labelled with what happened and for how long. Click a bar to inspect it.
2. **Ranked findings.** The three most useful findings appear as cards, and the rest follow in a table. Click *Inspect* or a table row to open a finding. *Download findings* saves the table as CSV.
3. **The selected finding.** It shows what the period lets you measure, followed by four tabs:
   - *Readout:* how sales moved compared with other markets, plus spend per channel with the finding's channels highlighted.
   - *Why flagged:* the scores behind the finding, in plain words.
   - *Caveats:* what to keep in mind.
   - *Explanation:* the detector's own description.

Use the sidebar to filter by market or pattern, hide findings with data warnings, or upload your own files.

## Use your own data

Choose **Upload CSV files** in the sidebar.

| File | Required columns |
|---|---|
| Media (required) | `date`, `country_code`, `advertising_channel`, `media_investment` |
| Sales (optional) | `date`, `country_code`, `turnover` |

- Write dates as `2024-06-24`.
- Spend must be a non-negative number.
- Several rows for the same day, market and channel are added together.
- Media may also include `impressions` and `clicks`, which help spot tracking problems.
- Without sales data, the app still finds periods but cannot show sales readings.

## Read the numbers

- **Priority** (High, Medium or Low) shows how useful a period is likely to be for analysis.
- **Evidence** (Strong, Moderate or Weak) shows how clearly the data shows the pattern. Both are rule-based scores, not probabilities.
- **Sales vs comparison** is the change in a market's sales during the period that other markets did not share. It is marked *unclear* when it is within the market's normal ups and downs.
- **Rough return** is the extra sales per extra unit of spend. It appears only when the change is clear and nothing else obviously explains it.
- **Baseline** shows sales while all advertising was off.

These are rough readings, not proof of what advertising caused. Averages over a pause understate its effect, because advertising keeps working for a while after it stops.

**Limits:**
- Pauses shorter than seven days are not reported.
- A period that touches the start or end of the data may continue beyond it.
- Overlapping findings are hard to separate.
- The tool has been tested on simulated data, not yet on real company data.

## Check the results

```bash
python -m scripts.evaluate_sample
python -m pytest -q -m "not slow"
```

The first command scores the detector on the included sample, which contains six known events. All six are found and nothing extra is reported. The sample was used during development, so this is a regression check, not a measure of performance on new data.

## For developers

- [How the detector works](detection/README.md)
- [Adding another detector](detectors/README.md)
- [Benchmark data and folder layout](benchmark/BENCHMARK.md)
- [How evaluation works](benchmark/eval/README.md)
- [Generating synthetic data](synthetic_data_generator/README.md)
