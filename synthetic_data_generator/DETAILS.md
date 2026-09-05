# Parameter Reference

Everything simulation-wide lives in [`config.yaml`](config.yaml). Informative
periods live in [`events_config.yaml`](events_config.yaml). Both R
(`generate_with_simmmulator.R`) and Python (`reformat.py`) read these two
files directly, nothing is duplicated or hardcoded per language.

## Top level: `config.yaml`

```yaml
years: 2
start_date: "2024/01/01"
revenue_per_conv: 40
customer_types: [New, Returning]
sales_channels: [Ecom, Stores]
baseline: {...}
campaign_spend: {...}
countries: [...]
channels: [...]
```

- `years` / `start_date`: how much data to generate and when it starts.
- `revenue_per_conv`: a single global scalar. siMMMulator does not support a
  different revenue-per-conversion per channel, so every channel's
  `conversion_value` uses this same multiplier.
- `customer_types` / `sales_channels`: used only by `reformat.py`, to split
  each day's total turnover into `sales.csv` rows.

## Baseline sales: `baseline`

```yaml
baseline:
  daily_mean: 15000
  trend_p: 0.5
  temp_var: 2
  temp_coef_mean: 100
  temp_coef_sd: 500
  error_std: 100
```

Passed straight through to siMMMulator's `step_1_create_baseline()`.
`daily_mean`, `temp_coef_mean`, `temp_coef_sd`, and `error_std` are all
multiplied by each country's `market_size` before use.

## Ad spend: `campaign_spend`

```yaml
campaign_spend:
  daily_total_mean: 18500
  daily_total_std: 4000
```

Total daily budget across all channels, before it's split by each channel's
`spend_share_min`/`spend_share_max` (see below). Also scaled by
`market_size`.

## Countries: `countries`

```yaml
countries:
  - code: DE
    name: Germany
    market_size: 1.00
```

- `code`: ISO country code, used as `country_code` in the output and as the
  `country` field in `events_config.yaml`.
- `name`: full name, used as `country` in `sales.csv`.
- `market_size`: multiplier applied to baseline sales and ad spend for that
  country. Keeps relative country size realistic, which is why detection
  algorithms need to normalize before comparing across countries, per the
  assignment hints.

To add a country: add an entry here, and optionally reference its `code` in
`events_config.yaml` to inject a pattern into it.

## Channels: `channels`

```yaml
channels:
  - name: TV
    type: impression
    platform: TV
    true_cvr: 0.00003
    true_cpm: 5
    mean_noisy_cpm_cpc: 0
    std_noisy_cpm_cpc: 0.3
    mean_noisy_cvr: 0
    std_noisy_cvr: 0.00001
    decay: 0.55
    alpha_saturation: 2
    gamma_saturation: 0.4
    spend_share_min: 0.36
    spend_share_max: 0.44
    assumed_ctr: 0.001
```

Each channel needs:

| Field | Meaning |
|---|---|
| `name` | Used as `advertising_channel` in `media.csv`. |
| `type` | `impression` or `click`. Determines whether siMMMulator uses `true_cpm` or `true_cpc` for that channel. |
| `platform` | Used as `ad_platform` in `media.csv` (e.g. both `Google Search` and `Google Discovery` can share platform `Google Ads`). |
| `true_cvr` | Underlying conversion rate. Impression-type channels are per-impression, so tiny (e.g. `3e-05`). Click-type channels are per-click, so larger (e.g. `0.02`). |
| `true_cpm` | Cost per 1000 impressions. Required for `type: impression`, omit for `type: click`. |
| `true_cpc` | Cost per click. Required for `type: click`, omit for `type: impression`. |
| `mean_noisy_cpm_cpc` / `std_noisy_cpm_cpc` | Noise added around `true_cpm`/`true_cpc`. |
| `mean_noisy_cvr` / `std_noisy_cvr` | Noise added around `true_cvr`. |
| `decay` | Adstock decay rate, 0 to 1. Higher means the ad effect lingers longer after spend stops. TV=0.55 (long memory) vs Search=0.10 (short, intent-driven) in the current config. This is exactly what the `channel_pulse` pattern is designed to let you measure. |
| `alpha_saturation` / `gamma_saturation` | Diminishing-returns (Hill) curve shape. Higher spend gives a proportionally smaller marginal return. |
| `spend_share_min` / `spend_share_max` | Min/max share of the daily budget this channel can get. Required for every channel except the last one in the list, which automatically receives whatever remains. Keep the max values comfortably under 1.0 in total, or the last channel can go negative. |
| `assumed_ctr` | Python-only, cosmetic. See below. |

You do not need to list impression channels before click channels: the
scripts split channels by `type` and reorder them internally (impressions
first, as siMMMulator requires), so channels can appear in any order in the
file.

### `assumed_ctr`, the one cosmetic field

siMMMulator only ever simulates impressions or clicks per channel, never
both. `reformat.py` uses `assumed_ctr` to back out a synthetic value for
whichever column the channel didn't get, purely so every row in `media.csv`
has both columns populated, matching the sample schema. This has zero effect
on spend, decay, saturation, or conversions, R never reads this field.

## Injected patterns: `events_config.yaml`

```yaml
- pattern_id: DE_DARK_01
  pattern_type: dark_period
  country: DE
  channel: ALL
  start_day: 175
  end_day: 230
  multiplier: 0
  description: All German advertising is off for ~8 weeks.
```

- `pattern_id`: unique label, free text, shown as-is in `ground_truth.csv`.
- `pattern_type`: one of:
  - `dark_period`: `channel` should be `ALL`. Zeroes every channel's spend
    in that country and window.
  - `single_channel`: `channel` is the one channel that stays on. Every
    other channel is zeroed for that country and window.
  - `natural_holdout`: `channel` drops to zero, all other channels
    unaffected.
  - `step_change`: `channel`'s spend is multiplied by `multiplier` (e.g. `3`
    triples it) for the window.
  - anything else (e.g. `channel_pulse`, `staggered_launch`) is treated like
    `natural_holdout`: spend for that one channel is multiplied by
    `multiplier` in the window. Use multiple entries with different windows
    to build an on/off/on pulse (see the three `DE_RADIO_PULSE_0x` entries).
- `country`: must match a `code` in `config.yaml`'s `countries` list.
- `channel`: must match a `name` in `config.yaml`'s `channels` list, or
  `ALL`.
- `start_day` / `end_day`: 0-indexed day offsets from `start_date`, using
  Python-slice convention. The affected days are `start_day, start_day+1,
  ..., end_day-1`. `end_day` itself is not modified. This matches how
  `ground_truth.csv`'s `end_date` is computed, and matters if you're
  checking boundaries precisely.
- `multiplier`: `0` to zero out spend, or any factor for `step_change`/
  pulses. Left blank in `ground_truth.csv` output unless `pattern_type` is
  `step_change`.
- `description`: free text.

### Why injection works as a simple day filter

`generate_with_simmmulator.R` always runs siMMMulator with
`frequency_of_campaigns = 1`. Under that setting, siMMMulator's internal
`campaign_id` is the day number (1-indexed): each "campaign" lasts exactly
one day. That means injecting a pattern is just:

```r
day_range <- (start_day + 1):end_day     # convert 0-indexed offset to campaign_id
df_ads_step2$spend_channel[df_ads_step2$campaign_id %in% day_range & ...] <- 0
```

If you change `years`/`frequency_of_campaigns` behaviour so campaigns no
longer last exactly one day, this mapping breaks: `campaign_id` no longer
equals the day number, and `inject_events()` in
`generate_with_simmmulator.R` needs to be rewritten to map `campaign_id` to
calendar dates before filtering.

## A note on YAML and R number types

siMMMulator's input checks require R type `double` for numeric arguments.
YAML parses whole numbers like `2`, `40`, `15000` as R integers, not
doubles, which fails those checks. `generate_with_simmmulator.R` wraps every
numeric value pulled from `config.yaml` in `as.numeric()` to guard against
this. If you add a new numeric field to the config and wire it into the R
script, wrap it the same way.
