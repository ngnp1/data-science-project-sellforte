# Detecting informative marketing periods

Find dark periods, single-channel periods, natural holdouts, budget steps,
pulse trains, and staggered launches in daily marketing data.

## Run locally

Use Python 3.12 (the tested version). From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py --server.address 127.0.0.1
```

The viewer starts with the included synthetic sample. Filter by market/type,
select an event to see its spend windows and explanation, or upload your own
media CSV and optional sales CSV. Download the findings as CSV.

Media requires `date`, `country_code`, `advertising_channel`, and
`media_investment`. Spend must be numeric, finite, and non-negative. Missing
spend values are rejected rather than interpreted as zero. `impressions` and
`clicks` are optional. Sales requires `date`, `country_code`, and `turnover`.
Campaign rows are aggregated by country/channel/day; omitted dates are flagged
as missing. All returned event dates are inclusive.

## Verify the included sample

```bash
python -m scripts.evaluate_sample
python -m pytest -q -m "not slow"
```

The revised detector finds all six configured sample events with no extra
events: precision/recall/F1 are 1.0, including exact pulse component windows.
This sample is a known regression fixture, **not an unseen benchmark**.
The previous detector found five events, with three false positives and one
missed grouped pulse (F1 0.7143).

Tests requiring the separately generated benchmark datasets or R are explicitly
skipped when those resources are unavailable. The sample and synthetic unit
tests run without R. The full benchmark CSVs are not included in this repository;
see [benchmark/BENCHMARK.md](benchmark/BENCHMARK.md) for generation instructions.
R also needs `siMMMulator`, `dplyr`, and `yaml`. The generator uses the Python
interpreter that invokes it, so a second virtual environment is unnecessary.
The historical sealed-test results and calibration remain archived and do not
describe the revised detector. Do not overwrite the seal or represent a
regenerated dataset as the original sealed run.

## Scores and limits

Evidence/confidence and informativeness are heuristic scores in [0, 1], not
probabilities or causal-effect estimates. The old constant calibration is not
applied. Corrected evidence includes step sharpness and pulse-only corroboration.
Overlapping events and censored boundaries reduce informativeness. Controls are
conservative comparison candidates: a peer must remain active throughout the
window, and sibling controls must actually exist and remain active.

Permanent budget changes are reported through the last observed date with a
censoring caveat. A restart from zero alone is not enough to infer a permanent
budget increase. Pulse grouping supports six or more pauses and separates
unrelated long shutdowns; gaps shorter than seven days are not reported.

Still unvalidated: performance on real company data, causal comparability of
controls, and whether the usefulness ranking improves marketing-effect
estimation. Long near-zero periods can bias the active-spend baseline. Initial
dormancy can be ambiguous between launch and missing prior history. In a
two-channel market, a one-channel pause is labelled a holdout.

See [detection/README.md](detection/README.md) for the pipeline structure and
[benchmark/eval/README.md](benchmark/eval/README.md) for evaluation conventions.
