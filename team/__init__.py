"""Our teammates' algorithms, one module per person and period.

| Module            | Author  | Periods                          |
|-------------------|---------|----------------------------------|
| single_channel.py | Nam Phu | only one channel on              |
| holdout.py        | Thai    | natural holdout, staggered launch |
| pulse/            | Triet   | on-off pulsing, dark period      |
| step_change.py    | David   | budget step                      |
| staggered_launch.py | tvh-ds | staggered launch (with readiness checks); same findings as Thai's on the dev split |

detectors/team.py combines them into one Detector, taking each period from
its author's module.
"""
