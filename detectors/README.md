# Adding another detector

The app and the evaluation harness run any detector registered here. Our rule-based library ([`detection/`](../detection/README.md)) is one of them, wrapped in [rule_based.py](rule_based.py).

## Registered detectors

| Key | Name | Code |
|---|---|---|
| `rule_based` | Rule-based (ours) | [`detection/`](../detection/README.md) |
| `team` | Combined | [`team/`](../team/__init__.py): single channel by Nam Phu, staggered launch by tvh-ds, everything else by Triet |

Score either one on the sample with `python -m scripts.evaluate_sample --detector <key>`.

## Write one

Create `detectors/my_detector.py`:

```python
from detectors.base import Detector

class MyDetector(Detector):
    key = "my_detector"          # stable id
    name = "My detector"         # shown in the app
    event_types = frozenset({"dark_period", "natural_holdout"})

    def detect(self, media, sales):
        ...  # return a list of detection.model.DetectedEvent

DETECTORS = [MyDetector()]
```

Then add `"detectors.my_detector"` to `MODULES` in [registry.py](registry.py). If you forget, a test fails.

## What a finding must contain

- **Required:** market, event type, inclusive start and end dates, the channel (`None` for a dark period), and the off-windows (`components`) for a pulse.
- **Optional:** scores, explanation, evidence. When they are missing, the app shows "Unscored" or says no explanation was given.
- `validity` defaults to `"ok"`. A detector that does not check data quality therefore has its findings shown as OK.
- To get the same scores, data-quality checks and explanations as ours, pass your findings through `detection.scoring.score_findings(findings, build_panel(media, sales))` ([how scoring works](../detection/scoring/README.md)). The Team detector does this.

Once two or more detectors are registered, a **Detector** picker appears in the app's sidebar. It starts on Team.

## Score it

Put `detect = for_detector("my_detector")` (from `benchmark.eval.adapter`) in a module, then run:

```bash
python -m benchmark.eval.run_dev --detector mymodule:detect --load-data
```

Detector code must not import the benchmark's answers. The truth-isolation tests scan every registered detector, and run records hash `detectors/` as well as `detection/`.
