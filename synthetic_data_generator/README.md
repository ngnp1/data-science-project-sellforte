# Synthetic MMM Data Generator

Generates synthetic marketing and sales data for the "Detecting Informative
Periods in Marketing Data" project (Sellforte / Aalto). Output matches the
schema of the `media.csv` / `sales.csv` samples Sellforte provided, with known
"informative periods" (dark period, single-channel period, natural holdout,
step change, plus bonus patterns) injected so you can test detection
algorithms against a known answer key before the real dataset arrives.

## How it works

Two stages, one shared config:

1. **R: [`generate_with_simmmulator.R`](generate_with_simmmulator.R)**
   Uses Meta's open-source [siMMMulator](https://github.com/facebookexperimental/siMMMulator)
   package to simulate baseline sales, ad spend, adstock decay, and
   diminishing returns for every channel and country listed in
   [`config.yaml`](config.yaml). Before decay and saturation run, it injects
   the patterns listed in [`events_config.yaml`](events_config.yaml) directly
   into daily spend. Output: `raw_daily_wide.csv` (not committed, regenerate
   it, see below).

2. **Python: [`reformat.py`](reformat.py)**
   Reads the same `config.yaml`, and reshapes siMMMulator's wide per-country
   output into the long `media.csv` / `sales.csv` format. Writes
   `ground_truth.csv` (the answer key) and `true_roi.csv` (ground-truth ROI
   per channel).

**`config.yaml` and `events_config.yaml` are the only two config files, and
both scripts read them directly.** Nothing about countries, channels, spend
levels, or injected patterns is hardcoded separately in the R or Python code.
See [`DETAILS.md`](DETAILS.md) for the full field reference.

## Setup

R needs the `siMMMulator` and `yaml` packages:
```r
install.packages(c("remotes", "yaml"))
remotes::install_github("facebookexperimental/siMMMulator")
```

Python needs pandas, numpy, and PyYAML:
```bash
python3 -m venv .venv
.venv/bin/pip install pandas numpy pyyaml
```

## Usage

Run from inside `synthetic_data_generator/`:

```bash
Rscript generate_with_simmmulator.R
.venv/bin/python reformat.py
```

This regenerates `raw_daily_wide.csv` and `data/media.csv`,
`data/sales.csv`, `data/ground_truth.csv`, `data/true_roi.csv`. The `data/`
files in this repo are already generated, so you can use the dataset without
running R at all.

To change what gets generated (add a country, tune a channel's decay rate,
add a new injected pattern), edit `config.yaml` or `events_config.yaml` and
rerun both commands. No code changes needed for any of that.

## Output files: what you actually need

| File | What it is | Required for the assignment? |
|---|---|---|
| `data/media.csv` | Daily media spend, impressions, clicks, conversions per channel, campaign, country | Yes, this is the data your detector reads. |
| `data/sales.csv` | Daily turnover per country, customer type, sales channel | Yes, same as above. |
| `data/ground_truth.csv` | Answer key: which periods were injected, where, and why | Testing aid only. Lets you check your detector's recall and precision before Sellforte's real, unlabeled dataset arrives. Not part of the real data. |
| `data/true_roi.csv` | Ground-truth ROI per channel | Optional. Only meaningful because we generated the data ourselves. Real data has no ground-truth ROI, that's the whole point of MMM. |

## Injected informative periods

Defined in [`events_config.yaml`](events_config.yaml), applied to simulated
spend before conversions are calculated:

| pattern_id | pattern_type | country | channel | window |
|---|---|---|---|---|
| DE_DARK_01 | dark_period | DE | ALL | 2024-06-24 to 2024-08-17 |
| FI_SINGLE_SEARCH_01 | single_channel | FI | Google Search | 2024-09-07 to 2024-10-06 |
| AT_FACEBOOK_HOLDOUT_01 | natural_holdout | AT | Facebook | 2025-02-04 to 2025-03-18 |
| CH_SEARCH_STEP_01 | step_change (3x) | CH | Google Search | 2025-05-15 to 2025-07-09 |
| DE_RADIO_PULSE_01/02/03 | channel_pulse | DE | Radio | three 2-week off periods |
| US_INSTAGRAM_LAUNCH_01 | staggered_launch | US | Instagram | off for first 90 days |

The first four are the required patterns from the assignment brief. The last
two are bonus patterns; the brief explicitly welcomes extra ones.
