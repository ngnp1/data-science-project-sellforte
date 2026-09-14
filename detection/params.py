"""Every threshold the detector uses, with the failure mode of getting it wrong.

Defaults come from spec section 7's parameter table. They were chosen from
first principles -- weekly multiples, a conventional robust cut -- and are to be
adjusted ONLY against the development split, never the sealed test split.
"""

# --- P1, off-runs -----------------------------------------------------------

# A day counts as "off" when spend <= max(EPS_ABS, RHO * active_level).
# Too low misses near-zero events; too high reads ordinary low-spend days as off.
#
# The spec's parameter table proposed a value equal to the TOP of the near-zero
# band the spec itself defines (a near-zero event is 0.02x to 0.08x of normal
# spend). A threshold sitting inside the band it must classify decides those
# days by the noise realisation rather than by the event, so the value has to
# clear the band's upper bound with margin -- this one is roughly double it.
#
# Measured on the development split, changing it from the spec's value:
#   precision 0.867 -> 0.934, recall 0.693 -> 0.760, F1 0.770 -> 0.838,
#   null-scenario false positives unchanged at 0.000 per country-year.
# natural_holdout recall went 0.550 -> 0.800 and step_change false positives
# went 4 -> 0: the near-zero holdouts that fell through this mask were being
# re-detected as level shifts, so one threshold caused both failures.
#
# This is a domain-derived bound (clear the defined near-zero band), not a fit
# to individual scenarios. On real data the band is a business question -- what
# counts as "spend paused" versus "spend low" -- and this is the first
# parameter to revisit.
RHO = 0.15

# Absolute floor, so float noise around zero cannot register as spend.
EPS_ABS = 1e-6

# Shortest reportable event. At 7 the benchmark's deliberate 5-day edge case is
# undetectable and should surface as an honest false negative; lowering it to
# catch that case floods the output with noise.
MIN_DAYS = 7

# A run is notable if it is also RUN_RATIO times the series' own p90 off-run.
# This is the intermittent-channel guard: a flighting channel whose normal gaps
# are 3 days needs ~9 off-days to register, while a channel that is otherwise
# never off needs only MIN_DAYS. A single global threshold fails one of those
# two cases whichever value is picked.
RUN_RATIO = 3.0

# Below this many other runs the series has no usable distribution, so the
# ratio rule is skipped and MIN_DAYS alone applies.
MIN_RUNS_FOR_RATIO = 5

# --- P2, level shifts -------------------------------------------------------

# Half-width of the comparison windows, in days. A multiple of 7 so day-of-week
# structure cancels. Shorter gives a noisier z; longer misses short steps.
W = 21

# Robust z threshold. The dominant precision/recall lever for step changes.
Z_THRESH = 3.5

# Lower bound on the MAD scale, so a near-constant window cannot produce an
# unbounded z.
SIGMA_FLOOR = 0.05

# The new level must still hold this many days later, at half the original
# delta. Rejects spikes.
PERSIST = 14

# Fraction of the total level change that must land inside SHARPNESS_WINDOW.
# THIS is the gradual-ramp defence: a genuine step concentrates its change into
# a couple of days (sharpness near 1) while a 50-day ramp spreads it out
# (near 0.1). Persistence alone does NOT reject a ramp, because a ramp's new
# level genuinely does hold.
SHARPNESS = 0.6
SHARPNESS_WINDOW = 3

# Fraction of the opening delta the level must still show PERSIST days later
# for the shift to count as held. Higher rejects real steps that drift back
# slightly; lower lets a decaying spike pass as a step.
PERSIST_FRACTION = 0.5

# An opposite-sign shift closes a step episode only if its magnitude is
# comparable -- within this factor either way of the opening shift. Wider pairs
# a step's end with an unrelated later shift; narrower leaves real episodes
# open-ended, running to the series end.
EPISODE_MATCH_BAND = 2.0

# Centred rolling median width used for level work, to remove day-of-week
# structure. Run detection stays on raw daily values so boundaries land on
# exact dates.
ROLLING = 7

# Median-absolute-deviation to standard-deviation factor for a normal
# distribution. Lives here because tests/detection/test_params.py asserts no
# logic module hardcodes a constant.
MAD_TO_SIGMA = 1.4826

# --- P3, pulses -------------------------------------------------------------

# Minimum notable off-runs on one series to call it a pulse train.
PULSE_MIN_RUNS = 2

# Runs must be of similar length: IQR of lengths divided by median <= this.
PULSE_LEN_IQR_RATIO = 0.5

# --- P4, onsets and the cross-market layer ----------------------------------

# Spread in onset dates across markets, in days, above which the channel is
# reported as a staggered launch rather than coincidental start-up jitter.
ONSET_SPREAD = 14

# --- Section 8, confidence sub-scores ------------------------------------

# |z| at which magnitude evidence for a step change saturates. A z of this size
# is already overwhelming; above it the extra certainty is not worth reporting
# as a difference.
Z_SATURATION = 8.0

# Duration evidence saturates at this multiple of MIN_DAYS. At 2 a 14-day event
# is fully evidenced; lower makes every reportable event look equally long.
DURATION_SATURATION_MULT = 2.0

# Distinctiveness saturates at this ratio of the run's length to the series' own
# p90 off-run. At 3 a run three times the usual gap is maximally distinctive.
DISTINCTIVENESS_SATURATION = 3.0

# Spend at zero while impressions keep flowing is the signature of tracking
# loss, not a real pause. It is not zero, because the spend feed may simply be
# late, but it must sit far below a corroborated stop.
CORROBORATION_CONTRADICTED = 0.2

# No impressions column, or impressions that are all zero for this series, means
# corroboration is unavailable rather than contradicted.
CORROBORATION_UNKNOWN = 0.6

# Confidence sub-score weights per event type. They differ because the evidence
# differs: a dark period's whole claim is that every channel stopped together,
# so consistency carries real weight there and none at all for a step change,
# which makes no claim about its neighbours. Each row sums to 1.
CONFIDENCE_WEIGHTS = {
    "dark_period": {"magnitude_evidence": 0.2, "duration_evidence": 0.15,
                    "distinctiveness": 0.15, "edge_sharpness": 0.1,
                    "corroboration": 0.15, "consistency": 0.25},
    "single_channel": {"magnitude_evidence": 0.2, "duration_evidence": 0.15,
                       "distinctiveness": 0.15, "edge_sharpness": 0.1,
                       "corroboration": 0.15, "consistency": 0.25},
    "natural_holdout": {"magnitude_evidence": 0.25, "duration_evidence": 0.2,
                        "distinctiveness": 0.25, "edge_sharpness": 0.1,
                        "corroboration": 0.2, "consistency": 0.0},
    "channel_pulse": {"magnitude_evidence": 0.25, "duration_evidence": 0.1,
                      "distinctiveness": 0.3, "edge_sharpness": 0.15,
                      "corroboration": 0.2, "consistency": 0.0},
    "staggered_launch": {"magnitude_evidence": 0.2, "duration_evidence": 0.2,
                         "distinctiveness": 0.2, "edge_sharpness": 0.1,
                         "corroboration": 0.3, "consistency": 0.0},
    "step_change": {"magnitude_evidence": 0.45, "duration_evidence": 0.2,
                    "distinctiveness": 0.0, "edge_sharpness": 0.25,
                    "corroboration": 0.1, "consistency": 0.0},
}

# --- Section 8, informativeness ------------------------------------------

# Type prior. Dark periods read the baseline directly, single-channel periods
# give unambiguous attribution, and pulses are the only place adstock decay is
# observable -- so all three outrank a plain step change for an analyst
# choosing what to look at.
TYPE_PRIOR = {
    "dark_period": 1.0,
    "single_channel": 0.9,
    "channel_pulse": 0.9,
    "natural_holdout": 0.75,
    "staggered_launch": 0.6,
    "step_change": 0.45,
}

# Control availability, spec section 8: peers, then sibling channels, then none.
CONTROL_SCORE = {"peers": 1.0, "sibling_channels": 0.7, "none": 0.3}

# Informativeness driver weights. One row, not per type -- the type's own
# influence enters through TYPE_PRIOR.
INFORMATIVENESS_WEIGHTS = {
    "duration_adequacy": 0.25,
    "contrast": 0.15,
    "cleanliness": 0.15,
    "control_availability": 0.2,
    "type_prior": 0.25,
}

# Assumed adstock half-life in days. A window shorter than a few half-lives
# cannot show the decay it is supposed to reveal. Stated as an assumption
# because the real value is a modelling question, not a measurement.
ADSTOCK_HALF_LIFE = 7.0

# Informativeness saturates once the window covers this many half-lives.
ADSTOCK_WINDOWS_FOR_FULL_CREDIT = 4.0

# Multiplier applied when the event is censored at a series edge -- its true
# extent is unknown, so it is worth less than the same event fully observed.
CENSORING_PENALTY = 0.7

# Multiplier applied when another event overlaps the window and confounds it.
CONFOUNDED_PENALTY = 0.6
