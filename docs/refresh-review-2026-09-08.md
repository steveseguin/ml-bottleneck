# September 8 refresh review

The [failed run](https://github.com/steveseguin/ml-bottleneck/actions/runs/34288756256)
fetched data successfully but failed four of 89 tests. The publication step
was skipped, so the published snapshot remains the August 24 version.

The failures are numerical assertions in `tests/index-logic.test.mjs` that
load the changing snapshot while expecting tightly pinned historical values:

| Check | Expected | Refreshed snapshot |
| --- | --- | --- |
| Four-B70 DeepSeek optimized target | 117.0–117.5 tok/s | 116.041082 tok/s |
| Supplied-assumption optimized target | 75.0–75.4 tok/s | 74.390657 tok/s |
| Exported DeepSeek optimized target | 117.0–117.5 tok/s | Same changed target as the first check |
| R9700 Qwen3.6 35B projected rate | 77.5–77.9 tok/s | 75.599397 tok/s |

The refresh replaced 31 of 320 gold cases and changed 50 retained cases.
Loading the published and candidate snapshots into the same current engine
reproduced the B70 change, 117.220739 → 116.041082 tok/s. Its projected rate
remained 40.510066 tok/s and its physical ceiling remained 389.514510 tok/s.
This establishes a data-dependent regression-pin failure; it does not by
itself establish that every incoming benchmark is valid.

The candidate's gold audit reported median observed/predicted 1.01, 83%
within 1.5× and 93% within 2×. It also flagged two physical-ceiling outliers
for separate data/model review (Qwen3.6 27B / RTX 4070 and Qwen3 1.7B /
RTX 3060). Those warnings are not the four failing tests and were not
resolved or suppressed by this change.

## Disposition

The action had six successful weekly runs through August 24, followed by
three failures. Its weekly schedule is now removed. Manual runs retain the
refresh, validation and review-artifact steps, with publication disabled by
default. The explicit `publish` input can commit only after all checks pass.
No engine constants, test tolerances or published evidence were changed.

Before resuming unattended updates, separate fixed-corpus numerical
regressions from fresh-corpus calibration/invariant checks, then investigate
the physical-ceiling outliers. Do not repeatedly repin expected numbers or
widen ranges merely to make incoming evidence pass.

The original `refresh-review` artifact contains test logs, the gold audit,
row-level changes, the candidate snapshot and its patch. GitHub retains that
artifact for 14 days; this note preserves the diagnosis beyond its expiry.
