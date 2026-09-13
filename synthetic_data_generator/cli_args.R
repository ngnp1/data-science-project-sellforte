# Command-line argument parsing for generate_with_simmmulator.R.
#
# Kept in its own file so it can be sourced and tested without running a
# simulation. Uses only base R -- the generator gains no new package
# dependency.
#
# All options are optional, and the defaults reproduce the script's original
# behaviour exactly, so `Rscript generate_with_simmmulator.R` is unchanged.
#
#   --seed    integer RNG seed                        (default 42)
#   --config  path to the simulation config           (default config.yaml)
#   --events  path to the injected-pattern config     (default events_config.yaml)
#   --outdir  directory for raw_daily_wide.csv        (default .)

parse_cli_args <- function(argv) {
  opts <- list(
    seed = "42",
    config = "config.yaml",
    events = "events_config.yaml",
    outdir = "."
  )

  i <- 1
  while (i <= length(argv)) {
    if (!startsWith(argv[i], "--")) {
      stop(sprintf("expected an option starting with '--', got '%s'", argv[i]))
    }
    key <- substring(argv[i], 3)
    if (!key %in% names(opts)) {
      stop(sprintf("unknown option '--%s'; known options are: %s",
                   key, paste(names(opts), collapse = ", ")))
    }
    if (i + 1 > length(argv)) {
      stop(sprintf("option '--%s' needs a value", key))
    }
    opts[[key]] <- argv[i + 1]
    i <- i + 2
  }

  seed <- suppressWarnings(as.integer(opts$seed))
  if (is.na(seed)) stop("--seed must be an integer")
  opts$seed <- seed

  opts
}
