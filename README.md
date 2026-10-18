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

The app opens in your browser with an included sample dataset. You do not need R. To analyse your own data, choose **Upload CSV files** in the sidebar ([required columns](detection/README.md#input-data)).

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
