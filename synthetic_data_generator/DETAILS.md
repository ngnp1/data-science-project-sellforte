# Parameter Reference

How to change countries, channels, spend behaviour, and injected patterns.
Config lives at the top of `generate_with_simmmulator.R`, plus
`events_config.csv` for the injected periods.

## Countries: `COUNTRIES`

```r
COUNTRIES <- list(
  list(code = "DE", name = "Germany", market_size = 1.00),
  ...
)
```

- `code`: ISO country code, used as `country_code` in the output and as the
  `country` field in `events_config.csv`.
- `name`: full name, used as `country` in `sales.csv`.
- `market_size`: multiplier applied to baseline sales and ad spend for that
  country (`base_p`, `campaign_spend_mean/std` are all multiplied by this).
  Keeps relative country size realistic, which is why detection algorithms
  need to normalize before comparing across countries, per the assignment
  hints.

To add a country: add an entry here, and optionally reference its `code` in
`events_config.csv` to inject a pattern into it.

## Channels: `CHANNELS_IMPRESSIONS` / `CHANNELS_CLICKS`

```r
CHANNELS_IMPRESSIONS <- c("TV", "Radio", "Google Discovery", "Facebook", "Instagram")
CHANNELS_CLICKS <- c("Google Search")
```

Order matters everywhere. siMMMulator requires every per-channel vector
(`TRUE_CVR`, `TRUE_CPM`, `TRUE_CPC`, noise vectors, decay, saturation) to
list impression channels first, then click channels, in this exact order.
If you add or reorder a channel, every vector below must be updated to
match, both length and position.

`PLATFORM_OF` maps each channel to the `ad_platform` value used in
`media.csv` (for example, `"Google Search"` and `"Google Discovery"` both
map to platform `"Google Ads"`).

## Per-channel behaviour

| Vector | Meaning | Notes |
|---|---|---|
| `TRUE_CVR` | Underlying conversion rate per channel | Impression-channel CVRs are per-impression, tiny, e.g. `3e-05`. Click-channel CVRs are per-click, larger, e.g. `0.02` for Search. Noise added via `MEAN_NOISY_CVR`/`STD_NOISY_CVR`. |
| `TRUE_CPM` | Cost per 1000 impressions | `NA` for click-type channels. |
| `TRUE_CPC` | Cost per click | `NA` for impression-type channels. |
| `TRUE_LAMBDA_DECAY` | Adstock decay rate, 0 to 1 | Higher means the ad effect lingers longer after spend stops. TV=0.55 (long memory) vs Search=0.10 (short, intent-driven) in the current config. This is exactly what the `channel_pulse` pattern is designed to let you measure. |
| `ALPHA_SATURATION` / `GAMMA_SATURATION` | Diminishing-returns (Hill) curve shape | Higher spend gives a proportionally smaller marginal return. |
| `MAX_MIN_PROPORTION` | Min/max share of daily budget per channel | Pairs of (min, max) for every channel except the last (`Google Search` gets the remainder). Keep the max values comfortably under 1.0 in total, or the last channel can go negative. |

`REVENUE_PER_CONV` is a single global scalar. siMMMulator does not support a
different revenue-per-conversion per channel, so all channels'
`conversion_value` use the same multiplier.

## Injected patterns: `events_config.csv`

```
pattern_id,pattern_type,country,channel,start_day,end_day,multiplier,description
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
    `multiplier` in the window. Use multiple rows with different windows to
    build an on/off/on pulse (see the three `DE_RADIO_PULSE_0x` rows).
- `country`: must match a `code` in `COUNTRIES`.
- `channel`: must match a name in `CHANNELS_IMPRESSIONS`/`CHANNELS_CLICKS`,
  or `ALL`.
- `start_day` / `end_day`: 0-indexed day offsets from `START_DATE`, using
  Python-slice convention. The affected days are `start_day, start_day+1,
  ..., end_day-1`. `end_day` itself is not modified. This matches how
  `ground_truth.csv`'s `end_date` is computed, and matters if you're
  checking boundaries precisely.
- `multiplier`: `0` to zero out spend, or any factor for `step_change`/
  pulses. Left blank in `ground_truth.csv` output unless `pattern_type` is
  `step_change`.
- `description`: free text, quote it if it contains a comma.

### Why injection works as a simple day filter

`generate_with_simmmulator.R` always runs siMMMulator with
`frequency_of_campaigns = 1`. Under that setting, siMMMulator's internal
`campaign_id` is the day number (1-indexed): each "campaign" lasts exactly
one day. That means injecting a pattern is just:

```r
day_range <- (start_day + 1):end_day     # convert 0-indexed offset to campaign_id
df_ads_step2$spend_channel[df_ads_step2$campaign_id %in% day_range & ...] <- 0
```

If you change `frequency_of_campaigns` to anything other than `1`, this
mapping breaks: `campaign_id` no longer equals the day number, and
`inject_events()` in `generate_with_simmmulator.R` needs to be rewritten to
map `campaign_id` to calendar dates before filtering.

## Cosmetic-only details

These don't affect the simulation itself.

`reformat.py`'s `ASSUMED_CTR` dict backs out a synthetic "impressions" value
for Search (or "clicks" for impression channels) purely so every row in
`media.csv` has both columns populated, matching the sample schema.
siMMMulator itself only ever simulates impressions or clicks per channel,
never both. This derived value has zero effect on spend, decay, saturation,
or conversions.

`stable_campaign_id()` in both scripts hashes `(platform, channel, country)`
into a stable-looking numeric ID. It's cosmetic, matching the sample's
`campaign_id` column style, and isn't used anywhere in the simulation logic.
