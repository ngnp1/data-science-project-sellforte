"""Our teammates' algorithms, one module per person and period.

| Module            | Author  | Periods                          |
|-------------------|---------|----------------------------------|
| single_channel.py | Nam Phu | only one channel on              |
| holdout.py        | Thai    | natural holdout (also has a staggered-launch detector) |
| pulse/            | Triet   | on-off pulsing, dark period      |
| step_change.py    | David   | budget step                      |
| staggered_launch.py | tvh-ds | staggered launch, with readiness checks |

detectors/team.py combines them into one Detector. On the development split
Triet's detector was strongest on steps, holdouts, pulses and dark periods,
Nam Phu's on single-channel periods, and tvh-ds's launches matched the
others with readiness checks added, so those are the ones it uses.
"""
