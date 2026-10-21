# Generate synthetic marketing data

This generator creates daily ad spend and sales with known events, such as a pause or a budget increase, so we can check that the detector finds them. A ready-made sample is in [data/](data/). The app and the sample check need no R.

## How it works

1. [generate_with_simmmulator.R](generate_with_simmmulator.R) uses [siMMMulator](https://github.com/facebookexperimental/siMMMulator) to simulate spend and sales, applying the configured events to spend.
2. [reformat.py](reformat.py) writes the project's CSV format and the answer files.

Markets, channels and simulation settings are in [config.yaml](config.yaml), and the inserted events are in [events_config.yaml](events_config.yaml). Every field is described in the [settings reference](DETAILS.md).

## Setup

Use the Python environment from the [main README](../README.md#run-the-app). Then install R, and in an R console run:

```r
install.packages(c("remotes", "yaml", "dplyr"))
remotes::install_github("facebookexperimental/siMMMulator")
```

## Generate a separate sample

```bash
cd synthetic_data_generator
Rscript generate_with_simmmulator.R --outdir generated
python reformat.py --outdir generated
```

This writes to `generated/` and leaves `data/` alone.

| Output | Contents |
|---|---|
| `media.csv` | Daily spend and media measurements by market and channel. |
| `sales.csv` | Daily turnover by market, customer type and sales channel. |
| `ground_truth.csv` | The inserted events (for evaluation only). |
| `true_roi.csv` | Simulated revenue ÷ spend per channel. |
| `raw_daily_wide.csv` | Intermediate output from R. |

## The included sample

It contains six events:
- a German dark period;
- a Finnish period with only Google Search running;
- an Austrian Facebook holdout;
- a Swiss Google Search budget increase;
- three German Radio pauses (one pulse event);
- a late US Instagram launch.

For exact dates, use the day offsets in `events_config.yaml`. For many scenarios at once, use the [benchmark generator](../benchmark/BENCHMARK.md).
