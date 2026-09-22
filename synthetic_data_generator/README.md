# Generate synthetic marketing data

This generator creates daily advertising spend and sales with known changes, such as a pause or a budget increase. The known events let us check whether the detector finds the right periods.

The repository already includes a sample in [data/](data/). You can run the viewer and sample checks without installing R.

## How generation works

1. [The R script](generate_with_simmmulator.R) uses [siMMMulator](https://github.com/facebookexperimental/siMMMulator) to simulate spend and sales. It changes spend according to the configured events before calculating media response, lingering ad effects, and saturation.
2. [The Python script](reformat.py) reshapes the output into the project's CSV format and writes the answers used for checking detections.

Both scripts read [config.yaml](config.yaml) for markets, channels, and simulation settings, and [events_config.yaml](events_config.yaml) for inserted events. See the [parameter reference](DETAILS.md) when changing them.

## Setup

Use the Python environment from the [main setup](../README.md#run-the-viewer). No second virtual environment is needed.

To generate new data, also install R and run these commands in an R console:

```r
install.packages(c("remotes", "yaml", "dplyr"))
remotes::install_github("facebookexperimental/siMMMulator")
```

## Generate a separate sample

With the Python environment active, start in the repository folder:

```bash
cd synthetic_data_generator
Rscript generate_with_simmmulator.R --outdir generated
python reformat.py --outdir generated
```

Both commands use the same output directory. This writes a new sample to `generated/` and leaves the committed `data/` sample in place. To change the simulation, edit the YAML settings and rerun both commands.

| Output | Contents |
|---|---|
| `raw_daily_wide.csv` | Intermediate output from R, read by Python. |
| `media.csv` | Daily spend and media measurements by market and channel. |
| `sales.csv` | Daily turnover split by market, customer type, and sales channel. |
| `ground_truth.csv` | The inserted events, for evaluation only. |
| `true_roi.csv` | Simulated attributed revenue divided by spend for each channel; not an estimate from real data. |

## Events in the included sample

The configuration includes a German dark period, a Finnish period with only Google Search active, an Austrian Facebook holdout, a Swiss Google Search budget increase, three German Radio pauses, and a late US Instagram launch.

Those are six grouped events: the three Radio pauses form one pulse train. Exact offsets and settings are in [events_config.yaml](events_config.yaml). For precise dates, use its day offsets; the legacy `ground_truth.csv` end dates do not consistently use the detector's inclusive convention.

For many scenarios instead of one sample, use the [benchmark generator](../benchmark/BENCHMARK.md).
