# The Frozen Benchmark

100 scenarios: 45 development, 55 hold-out test. Test split sealed 2026-09-13T16:09:20+00:00, spec_hash `98e8eb299965a59a2059f2d09426c15ba0ef614cdbe79d6f465969e49cd3cbb7`.

Dataset directories hold only `media.csv` and `sales.csv`. Ground truth, `meta.json`, `scenario.json` and `true_roi.csv` live under `<split>_truth/`, which detector code must never import.

## Regenerating

```bash
python -m benchmark.harness.generate --split dev
python -m benchmark.harness.generate --split test --seal
```

Generation is deterministic: the same code produces the same 100 scenarios, so `spec_hash` is a full description of the benchmark.

## Families

| family | dev | test |
|---|---|---|
| null | 4 | 5 |
| dark | 3 | 4 |
| single_channel | 3 | 4 |
| holdout | 3 | 4 |
| step | 5 | 6 |
| pulse | 3 | 4 |
| launch | 3 | 4 |
| cross_market | 3 | 4 |
| mixed | 8 | 10 |
| edge | 10 | 10 |
| **total** | **45** | **55** |

## Coverage

- **noise_level**: `high`x28, `low`x41, `med`x31
- **market_spread**: `extreme`x33, `moderate`x33, `tight`x34
- **n_countries**: `1`x10, `2`x29, `3`x24, `5`x22, `8`x15
- **n_channels**: `12`x19, `2`x19, `4`x34, `6`x15, `9`x13
- **trend_p**: `0.0`x37, `0.5`x36, `1.0`x27
- **years**: `1`x15, `2`x85

## Negative controls

Two pattern types change the data but must never be reported as detections; anything found inside their windows counts as a false positive:

- `ramp_block` — gradual drift, not a step change
- `intermittent_baseline` — a flighting channel whose short gaps are normal

Plus 9 `null` scenarios carrying no events at all, which are the only way to measure a false-positive rate.
