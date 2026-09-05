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

set.seed(42)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
YEARS <- 2
START_DATE <- "2024/01/01"
REVENUE_PER_CONV <- 40

COUNTRIES <- list(
  list(code = "DE", name = "Germany",       market_size = 1.00),
  list(code = "AT", name = "Austria",       market_size = 0.25),
  list(code = "CH", name = "Switzerland",   market_size = 0.30),
  list(code = "US", name = "United States", market_size = 1.30),
  list(code = "FI", name = "Finland",       market_size = 0.20)
)

# channels_impressions first, then channels_clicks -- order matters for every
# vector argument passed to siMMMulator below.
CHANNELS_IMPRESSIONS <- c("TV", "Radio", "Google Discovery", "Facebook", "Instagram")
CHANNELS_CLICKS <- c("Google Search")
CHANNELS <- c(CHANNELS_IMPRESSIONS, CHANNELS_CLICKS)

PLATFORM_OF <- c(
  "TV" = "TV", "Radio" = "Radio",
  "Google Discovery" = "Google Ads", "Google Search" = "Google Ads",
  "Facebook" = "Meta", "Instagram" = "Meta"
)

TRUE_CVR   <- c(0.00003, 0.00002, 0.00006, 0.00005, 0.00004, 0.02)
TRUE_CPM   <- c(5, 3, 8, 10, 12, NA)
TRUE_CPC   <- c(NA, NA, NA, NA, NA, 0.8)
MEAN_NOISY_CPM_CPC <- rep(0, 6)
STD_NOISY_CPM_CPC  <- c(0.3, 0.2, 0.5, 0.5, 0.6, 0.05)
MEAN_NOISY_CVR <- rep(0, 6)
STD_NOISY_CVR  <- c(0.00001, 0.000008, 0.00002, 0.000015, 0.000012, 0.005)
TRUE_LAMBDA_DECAY <- c(0.55, 0.35, 0.20, 0.30, 0.25, 0.10)
ALPHA_SATURATION  <- rep(2, 6)
GAMMA_SATURATION  <- c(0.4, 0.3, 0.3, 0.3, 0.3, 0.2)

# min/max proportion of daily budget for the first 5 channels (Google Search,
# the last channel, automatically receives whatever remains)
MAX_MIN_PROPORTION <- c(
  0.36, 0.44,  # TV
  0.08, 0.12,  # Radio
  0.05, 0.09,  # Google Discovery
  0.11, 0.15,  # Facebook
  0.05, 0.09   # Instagram
)

events <- read.csv("events_config.csv", stringsAsFactors = FALSE)

# ---------------------------------------------------------------------------
# Inject an informative period directly into step-2 spend.
# With frequency_of_campaigns = 1, campaign_id IS the day number (1-indexed),
# so a [start_day, end_day) window (0-indexed, Python-slice convention) maps to
# campaign_id in (start_day+1) .. end_day.
# ---------------------------------------------------------------------------
inject_events <- function(df_ads_step2, country_code) {
  ev_country <- events[events$country == country_code, ]
  for (i in seq_len(nrow(ev_country))) {
    ev <- ev_country[i, ]
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
    base_p = 15000 * ms,
    trend_p = 0.5,
    temp_var = 2,
    temp_coef_mean = 100 * ms,
    temp_coef_sd = 500 * ms,
    error_std = 100 * ms
  )

  df_ads_step2 <- step_2_ads_spend(
    my_variables = my_variables,
    campaign_spend_mean = 18500 * ms,
    campaign_spend_std = 4000 * ms,
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

channels_meta <- data.frame(
  channel = CHANNELS,
  platform = unname(PLATFORM_OF[CHANNELS]),
  type = c(rep("impression", length(CHANNELS_IMPRESSIONS)), rep("click", length(CHANNELS_CLICKS)))
)
write.csv(channels_meta, "channels_meta.csv", row.names = FALSE)

run_meta <- data.frame(revenue_per_conv = REVENUE_PER_CONV, start_date = START_DATE, years = YEARS)
write.csv(run_meta, "run_meta.csv", row.names = FALSE)

cat("\nDone. Wrote raw_daily_wide.csv (", nrow(all_countries_df), "rows ),",
    "channels_meta.csv, run_meta.csv.\n")
cat("Next: run `python reformat.py` to produce media.csv / sales.csv / ground_truth.csv\n")
