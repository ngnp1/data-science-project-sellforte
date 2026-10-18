"""Pluggable detectors: the interface the app and the benchmark harness use.

`detection/` is one implementation (see rule_based.py). Others register here
and are picked up by the viewer and by `benchmark.eval.adapter.for_detector`.
"""
