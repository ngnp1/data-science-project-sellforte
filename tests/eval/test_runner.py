from benchmark.eval import detectors_for_testing as D
from benchmark.eval import truth as T
from benchmark.eval.report import render_markdown
from benchmark.eval.runner import evaluate_split

SOME = T.list_scenarios("dev")[:8]


def test_evaluate_split_covers_every_requested_scenario():
    got = evaluate_split(D.never_detect, "dev", sids=SOME)
    assert got["n_scenarios"] == len(SOME)
    assert set(got["per_scenario"]) == set(SOME)


def test_perfect_oracle_scores_one_over_a_whole_split():
    got = evaluate_split(D.perfect_oracle, "dev", sids=SOME)
    assert got["overall"]["f1"] == 1.0


def test_never_detect_has_a_zero_false_positive_rate_on_nulls():
    got = evaluate_split(D.never_detect, "dev")
    assert got["null_fp_rate"] == 0.0


def test_detect_everything_has_a_nonzero_false_positive_rate_on_nulls():
    got = evaluate_split(D.detect_everything, "dev")
    assert got["null_fp_rate"] > 0.0


def test_breakdowns_are_present_for_the_interpretable_axis():
    got = evaluate_split(D.perfect_oracle, "dev", sids=SOME)
    assert "noise_level" in got["breakdowns"]


def test_report_renders_without_crashing_and_names_the_headline_numbers():
    got = evaluate_split(D.perfect_oracle, "dev", sids=SOME)
    md = render_markdown(got)
    assert "Precision" in md and "Recall" in md and "F1" in md
    assert "false positive" in md.lower()
    assert "Reliability" in md and "Operating curve" in md


def test_report_says_so_plainly_when_nothing_carries_a_confidence():
    got = evaluate_split(D.never_detect, "dev", sids=SOME)
    md = render_markdown(got)
    assert "calibration cannot be assessed" in md.lower()


def _minimal_results(breakdowns):
    """The smallest `results` dict render_markdown can render, so breakdown
    rendering can be tested in isolation from a real split."""
    return {
        "split": "dev",
        "n_scenarios": 1,
        "overall": {"precision": 1.0, "recall": 1.0, "f1": 1.0,
                    "n_tp": 1, "n_fp": 0, "n_fn": 0},
        "per_type": {},
        "iou": {"mean_iou": 1.0, "median_iou": 1.0, "n": 1},
        "boundary": {"start_median": 0.0, "start_p90": 0.0,
                     "end_median": 0.0, "end_p90": 0.0, "n": 1},
        "accuracy": {"channel_accuracy": 1.0, "market_accuracy": 1.0,
                     "n_relaxed_matches": 1},
        "day_level": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
        "confusion": {},
        "reliability": [],
        "operating": [],
        "null_fp_rate": 0.0,
        "null_country_years": 0.0,
        "per_scenario": {},
        "breakdowns": breakdowns,
    }


def test_breakdown_row_with_no_counts_at_all_renders_as_not_available():
    """A `null`-family-style bucket has zero TP, FP and FN -- there is nothing
    to detect and nothing was detected. `metrics.prf` reports 0.0/0.0/0.0 for
    that case by its own frozen convention, but the report must not render
    that as `0.000`, which reads as total failure on exactly the bucket where
    the detector is behaving perfectly."""
    breakdowns = {
        "family": {
            "null": {"n_tp": 0, "n_fp": 0, "n_fn": 0, "n_scenarios": 4,
                     "precision": 0.0, "recall": 0.0, "f1": 0.0},
            "_warning": "",
        },
    }
    md = render_markdown(_minimal_results(breakdowns))
    assert "| null | 4 | n/a | n/a | n/a |" in md
    assert "| null | 4 | 0.000 | 0.000 | 0.000 |" not in md


def test_breakdown_row_with_a_real_miss_still_renders_zero():
    """A bucket with real truth events that were entirely missed (n_fn > 0)
    must still show 0.000 -- the n/a fix must not hide a genuine failure."""
    breakdowns = {
        "family": {
            "dark": {"n_tp": 0, "n_fp": 0, "n_fn": 3, "n_scenarios": 3,
                    "precision": 0.0, "recall": 0.0, "f1": 0.0},
            "_warning": "",
        },
    }
    md = render_markdown(_minimal_results(breakdowns))
    assert "| dark | 3 | 0.000 | 0.000 | 0.000 |" in md
    assert "n/a" not in md


def test_confounded_axis_warning_sits_between_its_heading_and_its_table():
    """The honesty guarantee: the warning must appear ON the axis, between its
    heading and its table rows -- not merely somewhere in the document. This
    would fail if the warning were rendered in an appendix, or applied to
    every axis indiscriminately."""
    got = evaluate_split(D.perfect_oracle, "dev", sids=SOME)
    md = render_markdown(got)
    lines = md.splitlines()

    def table_header_after(start: int) -> int:
        return next(i for i in range(start, len(lines))
                   if lines[i].startswith("| value |"))

    for axis in ("trend_p", "market_spread"):
        heading = lines.index(f"### {axis} ⚠️")
        table = table_header_after(heading)
        between = lines[heading + 1:table]
        assert any("confounded" in line.lower() for line in between), (
            f"no confound warning found between the {axis} heading and its table")

    noise_heading = lines.index("### noise_level")
    noise_table = table_header_after(noise_heading)
    between_noise = lines[noise_heading + 1:noise_table]
    assert not any("confounded" in line.lower() for line in between_noise), (
        "noise_level is not a confounded axis and must not carry the warning")


def test_boundary_error_on_zero_matches_renders_as_not_available():
    """I3b. `metrics.boundary_error` returns zeros with `n: 0` when nothing
    matched. Rendered bare, `never_detect` -- which matched nothing at all --
    shows 0.0-day median and p90 boundary error: flawless localisation, on a
    detector that localised nothing. Same degenerate-row class as the
    breakdown buckets, and it gets the same `n/a` treatment, with `n` beside
    it so the reader can see what the figure rests on."""
    results = _minimal_results({})
    results["boundary"] = {"start_median": 0.0, "start_p90": 0.0,
                           "end_median": 0.0, "end_p90": 0.0, "n": 0}
    md = render_markdown(results)
    assert "| start | n/a | n/a | 0 |" in md
    assert "| end | n/a | n/a | 0 |" in md
    assert "| start | 0.0 | 0.0 |" not in md


def test_boundary_error_with_real_matches_still_renders_the_numbers():
    """The n/a fix must not hide a genuine measurement -- a detector that
    matched events and localised them perfectly must still read 0.0."""
    results = _minimal_results({})
    results["boundary"] = {"start_median": 0.0, "start_p90": 3.0,
                           "end_median": 1.0, "end_p90": 4.0, "n": 17}
    md = render_markdown(results)
    assert "| start | 0.0 | 3.0 | 17 |" in md
    assert "| end | 1.0 | 4.0 | 17 |" in md
    assert "n/a" not in md


def test_never_detect_does_not_report_flawless_boundary_localisation():
    """End to end on the real dev split: the detector that matches nothing
    must not read as perfectly localised."""
    got = evaluate_split(D.never_detect, "dev", sids=SOME)
    assert got["boundary"]["n"] == 0
    md = render_markdown(got)
    boundary = md.split("## Boundary error")[1].split("##")[0]
    assert "n/a" in boundary
    assert "0.0" not in boundary


def test_report_renders_the_event_level_breakdown_axes():
    """I7. Spec section 9 item 10 names duration, magnitude and near-zero vs
    exact-zero. They cannot be computed from meta.json, so they are bucketed
    per truth event -- and they have to reach the RENDERED report, which is
    what a later plan's author reads, not the results dict."""
    results = _minimal_results({})
    results["event_breakdowns"] = {
        "duration": [
            {"value": "<14 days", "n_events": 4, "n_matched": 1, "recall": 0.25},
            {"value": "90+ days", "n_events": 6, "n_matched": 6, "recall": 1.0},
        ],
        "zero_kind": [
            {"value": "exact zero", "n_events": 15, "n_matched": 15, "recall": 1.0},
            {"value": "near zero", "n_events": 5, "n_matched": 1, "recall": 0.2},
        ],
    }
    md = render_markdown(results)
    assert "## Event-level breakdowns" in md
    assert "### duration" in md and "### zero_kind" in md
    assert "| <14 days | 4 | 1 | 0.250 |" in md
    assert "| near zero | 5 | 1 | 0.200 |" in md
    # Recall-only, and the report must say why rather than leaving a reader to
    # wonder where precision went.
    assert "recall-only" in md.lower()


def test_event_breakdowns_reach_the_report_from_a_real_split():
    """End to end: the axes are computed by the runner and rendered, not just
    renderable in principle."""
    got = evaluate_split(D.shifted_oracle(10), "dev", sids=SOME)
    md = render_markdown(got)
    assert "## Event-level breakdowns" in md
    for axis in ("duration", "magnitude", "zero_kind"):
        assert f"### {axis}" in md
    assert "exact zero" in md


def test_the_report_warns_that_f1_cannot_separate_the_two_pathologies():
    """I9. The confounded-axis warnings travel with their tables; this one --
    that `never_detect` and `detect_everything` are indistinguishable on
    P/R/F1 -- landed only in the harness README, which the author of a later
    report has no reason to open. It has to sit beside the headline numbers
    themselves."""
    md = render_markdown(_minimal_results({}))
    headline = md.split("## Headline")[1].split("## Boundary error")[0]
    lowered = headline.lower()
    assert "f1" in lowered
    assert "false-positive rate" in lowered or "false positive rate" in lowered
    assert any(w in lowered for w in ("cannot tell", "cannot distinguish")), (
        "the headline does not warn that P/R/F1 cannot separate a silent "
        "detector from an indiscriminate one")
    assert "standalone headline" in lowered
