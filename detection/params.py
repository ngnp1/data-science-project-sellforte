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
