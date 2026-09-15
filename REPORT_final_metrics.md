# Evaluation — test split

55 scenarios.

## Headline

| metric | value |
|---|---|
| Precision | 0.978 |
| Recall | 0.918 |
| F1 | 0.947 |
| TP / FP / FN | 90 / 2 / 8 |
| Mean IoU | 0.991 |
| Median IoU | 1.000 |
| False positives per country-year (null scenarios) | 0.000 |
| Channel accuracy (relaxed match) | 1.000 |
| Market accuracy (relaxed match) | 1.000 |
| Day-level F1 | 0.920 |

> **Precision, recall and F1 cannot tell a silent detector from an indiscriminate one.** A detector that reports nothing and a detector that reports everything both land at 0.000 on all three here. The null-scenario false-positive rate above is the only number that separates them, so F1 must never be quoted from this report as a standalone headline.

## Boundary error (days)

| | median | p90 | n (matched pairs) |
|---|---|---|---|
| start | 0.0 | 1.0 | 90 |
| end | 0.0 | 0.0 | 90 |

## Per event type

| type | precision | recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| channel_pulse | 1.000 | 1.000 | 1.000 | 13 | 0 | 0 |
| dark_period | 1.000 | 1.000 | 1.000 | 15 | 0 | 0 |
| natural_holdout | 0.957 | 0.846 | 0.898 | 22 | 1 | 4 |
| single_channel | 1.000 | 1.000 | 1.000 | 12 | 0 | 0 |
| staggered_launch | 0.941 | 1.000 | 0.970 | 16 | 1 | 0 |
| step_change | 1.000 | 0.750 | 0.857 | 12 | 0 | 4 |

## Type confusion (truth → predicted)

| truth | predicted | n |
|---|---|---|
| channel_pulse | channel_pulse | 13 |
| dark_period | dark_period | 15 |
| natural_holdout | natural_holdout | 22 |
| natural_holdout | staggered_launch  ← substitution | 1 |
| single_channel | single_channel | 12 |
| staggered_launch | staggered_launch | 16 |
| step_change | step_change | 12 |

## Reliability — is the confidence score honest?

| confidence bin | n | mean confidence | empirical precision |
|---|---|---|---|
| 0.9–1.0 | 92 | 0.943 | 0.978 |

## Operating curve — the precision/recall trade-off

| confidence cut | detections kept | precision | recall | F1 |
|---|---|---|---|---|
| 0.00 | 92 | 0.978 | 0.918 | 0.947 |
| 0.05 | 92 | 0.978 | 0.918 | 0.947 |
| 0.10 | 92 | 0.978 | 0.918 | 0.947 |
| 0.15 | 92 | 0.978 | 0.918 | 0.947 |
| 0.20 | 92 | 0.978 | 0.918 | 0.947 |
| 0.25 | 92 | 0.978 | 0.918 | 0.947 |
| 0.30 | 92 | 0.978 | 0.918 | 0.947 |
| 0.35 | 92 | 0.978 | 0.918 | 0.947 |
| 0.40 | 92 | 0.978 | 0.918 | 0.947 |
| 0.45 | 92 | 0.978 | 0.918 | 0.947 |
| 0.50 | 92 | 0.978 | 0.918 | 0.947 |
| 0.55 | 92 | 0.978 | 0.918 | 0.947 |
| 0.60 | 92 | 0.978 | 0.918 | 0.947 |
| 0.65 | 92 | 0.978 | 0.918 | 0.947 |
| 0.70 | 92 | 0.978 | 0.918 | 0.947 |
| 0.75 | 92 | 0.978 | 0.918 | 0.947 |
| 0.80 | 92 | 0.978 | 0.918 | 0.947 |
| 0.85 | 92 | 0.978 | 0.918 | 0.947 |
| 0.90 | 92 | 0.978 | 0.918 | 0.947 |
| 0.95 | 0 | 0.000 | 0.000 | 0.000 |
| 1.00 | 0 | 0.000 | 0.000 | 0.000 |

## Breakdowns

### noise_level

| value | scenarios | precision | recall | F1 |
|---|---|---|---|---|
| high | 15 | 0.962 | 0.926 | 0.943 |
| low | 24 | 0.975 | 0.907 | 0.940 |
| med | 16 | 1.000 | 0.929 | 0.963 |

### family

| value | scenarios | precision | recall | F1 |
|---|---|---|---|---|
| cross_market | 4 | 1.000 | 1.000 | 1.000 |
| dark | 4 | 1.000 | 1.000 | 1.000 |
| edge | 10 | 0.900 | 0.643 | 0.750 |
| holdout | 4 | 1.000 | 1.000 | 1.000 |
| launch | 4 | 1.000 | 1.000 | 1.000 |
| mixed | 10 | 0.975 | 0.929 | 0.951 |
| null | 5 | n/a | n/a | n/a |
| pulse | 4 | 1.000 | 1.000 | 1.000 |
| single_channel | 4 | 1.000 | 1.000 | 1.000 |
| step | 6 | 1.000 | 1.000 | 1.000 |

### n_countries

| value | scenarios | precision | recall | F1 |
|---|---|---|---|---|
| 1 | 8 | 1.000 | 1.000 | 1.000 |
| 2 | 17 | 0.952 | 0.800 | 0.870 |
| 3 | 13 | 0.960 | 0.923 | 0.941 |
| 5 | 9 | 1.000 | 0.962 | 0.980 |
| 8 | 8 | 1.000 | 1.000 | 1.000 |

### n_channels

| value | scenarios | precision | recall | F1 |
|---|---|---|---|---|
| 2 | 7 | 1.000 | 0.429 | 0.600 |
| 4 | 17 | 0.971 | 0.944 | 0.958 |
| 6 | 11 | 1.000 | 1.000 | 1.000 |
| 9 | 10 | 1.000 | 0.944 | 0.971 |
| 12 | 10 | 0.947 | 0.947 | 0.947 |

### years

| value | scenarios | precision | recall | F1 |
|---|---|---|---|---|
| 1 | 8 | 1.000 | 1.000 | 1.000 |
| 2 | 47 | 0.974 | 0.904 | 0.938 |

### trend_p ⚠️

> **CONFOUNDED WITH FAMILY on both splits -- most families sit at a single value of this axis, so differences here measure family difficulty, not the axis. Do not report as an axis effect. See BENCHMARK.md.**

| value | scenarios | precision | recall | F1 |
|---|---|---|---|---|
| 0.0 | 19 | 1.000 | 0.969 | 0.984 |
| 0.5 | 18 | 0.978 | 0.957 | 0.967 |
| 1.0 | 18 | 0.938 | 0.750 | 0.833 |

### market_spread ⚠️

> **CONFOUNDED WITH FAMILY on both splits -- most families sit at a single value of this axis, so differences here measure family difficulty, not the axis. Do not report as an axis effect. See BENCHMARK.md.**

| value | scenarios | precision | recall | F1 |
|---|---|---|---|---|
| extreme | 18 | 1.000 | 0.933 | 0.966 |
| moderate | 18 | 1.000 | 0.935 | 0.967 |
| tight | 19 | 0.943 | 0.892 | 0.917 |

## Event-level breakdowns

_Spec §9 item 10's duration, magnitude and near-zero-vs-exact-zero axes. `meta.json` carries none of the three, so these bucket individual TRUTH events rather than whole scenarios — which makes them **recall-only**: a false positive belongs to no truth bucket, so precision has no denominator here and is not reported._

### duration

| value | truth events | matched | recall |
|---|---|---|---|
| <14 days | 1 | 0 | 0.000 |
| 14-41 days | 23 | 23 | 1.000 |
| 42-89 days | 39 | 34 | 0.872 |
| 90+ days | 35 | 33 | 0.943 |

### magnitude

| value | truth events | matched | recall |
|---|---|---|---|
| exact zero | 72 | 68 | 0.944 |
| near zero (0 < m < 0.1) | 10 | 10 | 1.000 |
| reduced (0.1 <= m < 1) | 4 | 3 | 0.750 |
| amplified (m >= 1) | 12 | 9 | 0.750 |

### zero_kind

| value | truth events | matched | recall |
|---|---|---|---|
| exact zero | 72 | 68 | 0.944 |
| near zero | 10 | 10 | 1.000 |
