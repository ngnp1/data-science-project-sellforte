# Generate multi-country synthetic marketing data using the siMMMulator R package,
# with informative periods (dark period, single-channel, natural holdout, step change,
# + bonus patterns) injected directly into the simulated ad spend before the rest of
# the siMMMulator pipeline (decay, saturation, conversions) runs on top of it.
#
# Output (raw, wide, per-country) is written to raw_daily_wide.csv and then reshaped
# into media.csv / sales.csv / ground_truth.csv by reformat.py (Python).
#
# Usage: Rscript generate_with_simmmulator.R

library(siMMMulator)
library(dplyr)
library(yaml)

set.seed(42)

# ---------------------------------------------------------------------------
# Config -- everything simulation-wide lives in config.yaml, informative
# periods live in events_config.yaml. See DETAILS.md for the full reference.
# ---------------------------------------------------------------------------
config <- yaml::read_yaml("config.yaml")
events <- yaml::read_yaml("events_config.yaml")

# yaml parses whole numbers (2, 40, 15000, ...) as R integers, but siMMMulator's
# input checks require type "double" -- as.numeric() everything pulled from yaml.
num <- function(x) as.numeric(x)
field <- function(ch, name, default = NA) if (is.null(ch[[name]])) default else num(ch[[name]])

YEARS <- num(config$years)
START_DATE <- config$start_date
REVENUE_PER_CONV <- num(config$revenue_per_conv)

BASELINE <- lapply(config$baseline, num)
CAMPAIGN_SPEND <- lapply(config$campaign_spend, num)

COUNTRIES <- lapply(config$countries, function(c) {
  c$market_size <- num(c$market_size)
  c
})

# a channel's position in config.yaml determines its position in every
# vector below; impression-type and click-type channels are split out and
# concatenated impressions-first regardless of how they're ordered in the file
impression_channels <- Filter(function(ch) ch$type == "impression", config$channels)
click_channels <- Filter(function(ch) ch$type == "click", config$channels)
channels_ordered <- c(impression_channels, click_channels)

CHANNELS_IMPRESSIONS <- sapply(impression_channels, function(ch) ch$name)
CHANNELS_CLICKS <- sapply(click_channels, function(ch) ch$name)
CHANNELS <- c(CHANNELS_IMPRESSIONS, CHANNELS_CLICKS)

PLATFORM_OF <- setNames(sapply(channels_ordered, function(ch) ch$platform), CHANNELS)

TRUE_CVR <- sapply(channels_ordered, function(ch) num(ch$true_cvr))
TRUE_CPM <- sapply(channels_ordered, function(ch) field(ch, "true_cpm"))
TRUE_CPC <- sapply(channels_ordered, function(ch) field(ch, "true_cpc"))
MEAN_NOISY_CPM_CPC <- sapply(channels_ordered, function(ch) num(ch$mean_noisy_cpm_cpc))
STD_NOISY_CPM_CPC <- sapply(channels_ordered, function(ch) num(ch$std_noisy_cpm_cpc))
MEAN_NOISY_CVR <- sapply(channels_ordered, function(ch) num(ch$mean_noisy_cvr))
STD_NOISY_CVR <- sapply(channels_ordered, function(ch) num(ch$std_noisy_cvr))
TRUE_LAMBDA_DECAY <- sapply(channels_ordered, function(ch) num(ch$decay))
ALPHA_SATURATION <- sapply(channels_ordered, function(ch) num(ch$alpha_saturation))
GAMMA_SATURATION <- sapply(channels_ordered, function(ch) num(ch$gamma_saturation))

# min/max spend share for every channel except the last (which receives
# whatever remains)
all_but_last <- channels_ordered[-length(channels_ordered)]
MAX_MIN_PROPORTION <- unlist(lapply(all_but_last, function(ch) {
  c(num(ch$spend_share_min), num(ch$spend_share_max))
}))

# ---------------------------------------------------------------------------
# Inject an informative period directly into step-2 spend.
# With frequency_of_campaigns = 1, campaign_id IS the day number (1-indexed),
# so a [start_day, end_day) window (0-indexed, Python-slice convention) maps to
# campaign_id in (start_day+1) .. end_day.
# ---------------------------------------------------------------------------
inject_events <- function(df_ads_step2, country_code) {
  for (ev in events) {
    if (ev$country != country_code) next
    day_range <- (ev$start_day + 1):ev$end_day

    if (ev$pattern_type == "single_channel") {
      keep_channel <- ev$channel
      idx <- df_ads_step2$campaign_id %in% day_range & df_ads_step2$channel != keep_channel
      df_ads_step2$spend_channel[idx] <- 0
    } else {
      if (ev$channel == "ALL") {
        idx <- df_ads_step2$campaign_id %in% day_range
      } else {
        idx <- df_ads_step2$campaign_id %in% day_range & df_ads_step2$channel == ev$channel
      }
      df_ads_step2$spend_channel[idx] <- df_ads_step2$spend_channel[idx] * ev$multiplier
    }
  }
  df_ads_step2
}

# ---------------------------------------------------------------------------
# Run the full siMMMulator pipeline for one country
# ---------------------------------------------------------------------------
run_country <- function(country) {
  cat("\n==== Generating data for", country$name, "====\n")
  ms <- country$market_size

  my_variables <- step_0_define_basic_parameters(
    years = YEARS,
    channels_impressions = CHANNELS_IMPRESSIONS,
    channels_clicks = CHANNELS_CLICKS,
    frequency_of_campaigns = 1,
    true_cvr = TRUE_CVR,
    revenue_per_conv = REVENUE_PER_CONV,
    start_date = START_DATE
  )

  df_baseline <- step_1_create_baseline(
    my_variables = my_variables,
    base_p = BASELINE$daily_mean * ms,
    trend_p = BASELINE$trend_p,
    temp_var = BASELINE$temp_var,
    temp_coef_mean = BASELINE$temp_coef_mean * ms,
    temp_coef_sd = BASELINE$temp_coef_sd * ms,
    error_std = BASELINE$error_std * ms
  )

  df_ads_step2 <- step_2_ads_spend(
    my_variables = my_variables,
    campaign_spend_mean = CAMPAIGN_SPEND$daily_total_mean * ms,
    campaign_spend_std = CAMPAIGN_SPEND$daily_total_std * ms,
    max_min_proportion_on_each_channel = MAX_MIN_PROPORTION
  )

  df_ads_step2 <- inject_events(df_ads_step2, country$code)

  df_ads_step3 <- step_3_generate_media(
    my_variables = my_variables,
    df_ads_step2 = df_ads_step2,
    true_cpm = TRUE_CPM,
    true_cpc = TRUE_CPC,
    mean_noisy_cpm_cpc = MEAN_NOISY_CPM_CPC,
    std_noisy_cpm_cpc = STD_NOISY_CPM_CPC
  )

  df_ads_step4 <- step_4_generate_cvr(
    my_variables = my_variables,
    df_ads_step3 = df_ads_step3,
    mean_noisy_cvr = MEAN_NOISY_CVR,
    std_noisy_cvr = STD_NOISY_CVR
  )

  df_ads_step5a <- step_5a_pivot_to_mmm_format(
    my_variables = my_variables,
    df_ads_step4 = df_ads_step4
  )

  df_ads_step5b <- step_5b_decay(
    my_variables = my_variables,
    df_ads_step5a_before_mmm = df_ads_step5a,
    true_lambda_decay = TRUE_LAMBDA_DECAY
  )

  df_ads_step5c <- step_5c_diminishing_returns(
    my_variables = my_variables,
    df_ads_step5b = df_ads_step5b,
    alpha_saturation = ALPHA_SATURATION,
    gamma_saturation = GAMMA_SATURATION
  )

  df_ads_step6 <- step_6_calculating_conversions(
    my_variables = my_variables,
    df_ads_step5c = df_ads_step5c
  )

  df_ads_step7 <- step_7_expanded_df(
    my_variables = my_variables,
    df_ads_step6 = df_ads_step6,
    df_baseline = df_baseline
  )

  df_ads_step7$country_code <- country$code
  df_ads_step7$country_name <- country$name
  df_ads_step7
}

all_countries_df <- bind_rows(lapply(COUNTRIES, run_country))

write.csv(all_countries_df, "raw_daily_wide.csv", row.names = FALSE)

cat("\nDone. Wrote raw_daily_wide.csv (", nrow(all_countries_df), "rows ).\n")
cat("Next: run `python reformat.py` to produce media.csv / sales.csv / ground_truth.csv\n")
