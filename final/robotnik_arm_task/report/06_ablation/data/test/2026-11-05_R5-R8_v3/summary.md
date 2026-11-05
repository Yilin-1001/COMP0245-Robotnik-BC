# Unified R1–R8 test results

Started: 2026-11-05T13:01:45.548390+00:00.
Status: COMPLETE; 3120/3120 requested rollouts.
Scene manifest SHA-256: `0a916445f6d87d3c667038d9ce29d5e5cdd43c43c695b1f4eb81e7b0797dda78`.
Run identity: `e84979ff0f4d3f4382a090162aa674134afa279fb560b36710f88b728d237f06`.

Only continuous error <10 mm counts as success (3 s; T5 20 s). All rollouts run to their fixed cap. Completion time includes the qualifying hold. Any exception overrides success with F. Terminal A–F precision uses the final inclusive second maximum and is independent of success.

T0 was previously viewed during development. This new scoring protocol is not numerically equivalent to the historical rollout-v1 results. New extension scenes were frozen before this run. T4 deliberately pairs translations with T0; it is not an independent relative-task sample.

Intervals: descriptive 95% scenario-cluster bootstrap (1000 resamples, fixed seed 1101031). Training seeds stay together when resampling a scene; intervals do not establish uncertainty over all possible training runs.

## Model coverage

| Row | Available training seeds | DAgger round |
|---|---|---|
| R5 DAgger | 0, 1, 2 | None |
| R6 DAgger + Gate | 0, 1, 2 | None |
| R7 f-DAgger | 0, 1, 2 | None |
| R8 f-DAgger + Gate | 0, 1, 2 | None |
| expert | None | None |

## All frozen scenes

| Row | Suite/band | Success | Rate [95% interval] | Completion median (successes only), s | Terminal median, mm |
|---|---|---|---|---|---|
| R5 DAgger | T0/all | 185/192 | 96.4% [92.7, 99.0] | 10.59 | 9.18 |
| R6 DAgger + Gate | T0/all | 176/192 | 91.7% [85.9, 96.4] | 10.38 | 11.69 |
| R7 f-DAgger | T0/all | 185/192 | 96.4% [91.7, 100.0] | 10.42 | 20.04 |
| R8 f-DAgger + Gate | T0/all | 188/192 | 97.9% [94.8, 100.0] | 10.30 | 4.76 |
| expert | T0/all | 64/64 | 100.0% [100.0, 100.0] | 10.20 | 0.65 |
| R5 DAgger | T1/10-15m | 72/72 | 100.0% [100.0, 100.0] | 53.01 | 7.39 |
| R6 DAgger + Gate | T1/10-15m | 70/72 | 97.2% [93.1, 100.0] | 53.00 | 12.65 |
| R7 f-DAgger | T1/10-15m | 7/72 | 9.7% [2.8, 18.1] | 48.17 | 3486.84 |
| R8 f-DAgger + Gate | T1/10-15m | 4/72 | 5.6% [1.4, 11.1] | 50.51 | 3271.04 |
| expert | T1/10-15m | 24/24 | 100.0% [100.0, 100.0] | 52.42 | 1.13 |
| R5 DAgger | T1/3-6m | 72/72 | 100.0% [100.0, 100.0] | 21.27 | 11.44 |
| R6 DAgger + Gate | T1/3-6m | 68/72 | 94.4% [87.5, 100.0] | 21.69 | 20.22 |
| R7 f-DAgger | T1/3-6m | 72/72 | 100.0% [100.0, 100.0] | 21.34 | 27.34 |
| R8 f-DAgger + Gate | T1/3-6m | 69/72 | 95.8% [90.3, 100.0] | 21.18 | 9.59 |
| expert | T1/3-6m | 24/24 | 100.0% [100.0, 100.0] | 21.34 | 0.82 |
| R5 DAgger | T1/6-10m | 72/72 | 100.0% [100.0, 100.0] | 35.43 | 7.95 |
| R6 DAgger + Gate | T1/6-10m | 71/72 | 98.6% [95.8, 100.0] | 35.65 | 14.45 |
| R7 f-DAgger | T1/6-10m | 65/72 | 90.3% [81.9, 97.3] | 34.51 | 32.49 |
| R8 f-DAgger + Gate | T1/6-10m | 26/72 | 36.1% [33.3, 40.3] | 37.26 | 1853.04 |
| expert | T1/6-10m | 24/24 | 100.0% [100.0, 100.0] | 34.64 | 0.68 |
| R5 DAgger | T1/all | 216/216 | 100.0% [100.0, 100.0] | 35.43 | 8.24 |
| R6 DAgger + Gate | T1/all | 209/216 | 96.8% [94.0, 99.1] | 35.71 | 15.09 |
| R7 f-DAgger | T1/all | 144/216 | 66.7% [55.1, 76.4] | 28.03 | 71.44 |
| R8 f-DAgger + Gate | T1/all | 99/216 | 45.8% [36.1, 54.6] | 22.44 | 962.60 |
| expert | T1/all | 72/72 | 100.0% [100.0, 100.0] | 34.64 | 0.83 |
| R5 DAgger | T2/all | 25/72 | 34.7% [22.2, 45.8] | 13.05 | 34.96 |
| R6 DAgger + Gate | T2/all | 36/72 | 50.0% [36.1, 65.3] | 8.98 | 21.38 |
| R7 f-DAgger | T2/all | 27/72 | 37.5% [25.0, 51.4] | 13.70 | 41.57 |
| R8 f-DAgger + Gate | T2/all | 41/72 | 56.9% [44.4, 69.4] | 8.69 | 29.88 |
| expert | T2/all | 24/24 | 100.0% [100.0, 100.0] | 6.29 | 1.10 |
| R5 DAgger | T3/all | 32/96 | 33.3% [22.9, 45.8] | 14.29 | 25.05 |
| R6 DAgger + Gate | T3/all | 26/96 | 27.1% [15.6, 39.6] | 13.95 | 43.88 |
| R7 f-DAgger | T3/all | 29/96 | 30.2% [17.7, 43.8] | 15.09 | 64.98 |
| R8 f-DAgger + Gate | T3/all | 51/96 | 53.1% [40.6, 66.7] | 13.60 | 27.61 |
| expert | T3/all | 26/32 | 81.2% [65.6, 93.8] | 12.31 | 1.19 |
| R5 DAgger | T3/high | 13/48 | 27.1% [14.6, 39.6] | 13.04 | 50.84 |
| R6 DAgger + Gate | T3/high | 17/48 | 35.4% [18.8, 52.1] | 12.85 | 46.85 |
| R7 f-DAgger | T3/high | 14/48 | 29.2% [14.6, 47.9] | 13.57 | 126.92 |
| R8 f-DAgger + Gate | T3/high | 36/48 | 75.0% [58.3, 87.5] | 12.18 | 17.75 |
| expert | T3/high | 16/16 | 100.0% [100.0, 100.0] | 11.90 | 1.05 |
| R5 DAgger | T3/low | 19/48 | 39.6% [20.8, 58.3] | 15.41 | 17.73 |
| R6 DAgger + Gate | T3/low | 9/48 | 18.8% [4.1, 37.5] | 16.81 | 37.61 |
| R7 f-DAgger | T3/low | 15/48 | 31.2% [12.5, 52.1] | 16.99 | 36.33 |
| R8 f-DAgger + Gate | T3/low | 15/48 | 31.2% [16.7, 45.8] | 18.23 | 29.91 |
| expert | T3/low | 10/16 | 62.5% [37.5, 81.2] | 15.15 | 4.84 |
| R5 DAgger | T4/all | 68/72 | 94.4% [88.9, 100.0] | 10.07 | 8.91 |
| R6 DAgger + Gate | T4/all | 61/72 | 84.7% [72.2, 94.4] | 9.60 | 13.86 |
| R7 f-DAgger | T4/all | 66/72 | 91.7% [81.9, 100.0] | 10.20 | 17.46 |
| R8 f-DAgger + Gate | T4/all | 68/72 | 94.4% [86.1, 100.0] | 9.94 | 7.13 |
| expert | T4/all | 24/24 | 100.0% [100.0, 100.0] | 9.65 | 0.62 |
| R5 DAgger | T5/all | 50/72 | 69.4% [56.9, 80.6] | 26.75 | 5.20 |
| R6 DAgger + Gate | T5/all | 44/72 | 61.1% [50.0, 72.2] | 28.98 | 18.70 |
| R7 f-DAgger | T5/all | 36/72 | 50.0% [38.9, 61.1] | 25.81 | 29.81 |
| R8 f-DAgger + Gate | T5/all | 57/72 | 79.2% [68.1, 88.9] | 28.26 | 14.67 |
| expert | T5/all | 24/24 | 100.0% [100.0, 100.0] | 26.39 | 0.74 |

## Expert-success subset

| Row | Suite/band | Success | Rate [95% interval] | Completion median (successes only), s | Terminal median, mm |
|---|---|---|---|---|---|
| R5 DAgger | T0/all | 185/192 | 96.4% [92.7, 99.0] | 10.59 | 9.18 |
| R6 DAgger + Gate | T0/all | 176/192 | 91.7% [85.9, 96.4] | 10.38 | 11.69 |
| R7 f-DAgger | T0/all | 185/192 | 96.4% [91.7, 100.0] | 10.42 | 20.04 |
| R8 f-DAgger + Gate | T0/all | 188/192 | 97.9% [94.8, 100.0] | 10.30 | 4.76 |
| expert | T0/all | 64/64 | 100.0% [100.0, 100.0] | 10.20 | 0.65 |
| R5 DAgger | T1/10-15m | 72/72 | 100.0% [100.0, 100.0] | 53.01 | 7.39 |
| R6 DAgger + Gate | T1/10-15m | 70/72 | 97.2% [93.1, 100.0] | 53.00 | 12.65 |
| R7 f-DAgger | T1/10-15m | 7/72 | 9.7% [2.8, 18.1] | 48.17 | 3486.84 |
| R8 f-DAgger + Gate | T1/10-15m | 4/72 | 5.6% [1.4, 11.1] | 50.51 | 3271.04 |
| expert | T1/10-15m | 24/24 | 100.0% [100.0, 100.0] | 52.42 | 1.13 |
| R5 DAgger | T1/3-6m | 72/72 | 100.0% [100.0, 100.0] | 21.27 | 11.44 |
| R6 DAgger + Gate | T1/3-6m | 68/72 | 94.4% [87.5, 100.0] | 21.69 | 20.22 |
| R7 f-DAgger | T1/3-6m | 72/72 | 100.0% [100.0, 100.0] | 21.34 | 27.34 |
| R8 f-DAgger + Gate | T1/3-6m | 69/72 | 95.8% [90.3, 100.0] | 21.18 | 9.59 |
| expert | T1/3-6m | 24/24 | 100.0% [100.0, 100.0] | 21.34 | 0.82 |
| R5 DAgger | T1/6-10m | 72/72 | 100.0% [100.0, 100.0] | 35.43 | 7.95 |
| R6 DAgger + Gate | T1/6-10m | 71/72 | 98.6% [95.8, 100.0] | 35.65 | 14.45 |
| R7 f-DAgger | T1/6-10m | 65/72 | 90.3% [81.9, 97.3] | 34.51 | 32.49 |
| R8 f-DAgger + Gate | T1/6-10m | 26/72 | 36.1% [33.3, 40.3] | 37.26 | 1853.04 |
| expert | T1/6-10m | 24/24 | 100.0% [100.0, 100.0] | 34.64 | 0.68 |
| R5 DAgger | T1/all | 216/216 | 100.0% [100.0, 100.0] | 35.43 | 8.24 |
| R6 DAgger + Gate | T1/all | 209/216 | 96.8% [94.0, 99.1] | 35.71 | 15.09 |
| R7 f-DAgger | T1/all | 144/216 | 66.7% [55.1, 76.4] | 28.03 | 71.44 |
| R8 f-DAgger + Gate | T1/all | 99/216 | 45.8% [36.1, 54.6] | 22.44 | 962.60 |
| expert | T1/all | 72/72 | 100.0% [100.0, 100.0] | 34.64 | 0.83 |
| R5 DAgger | T2/all | 25/72 | 34.7% [22.2, 45.8] | 13.05 | 34.96 |
| R6 DAgger + Gate | T2/all | 36/72 | 50.0% [36.1, 65.3] | 8.98 | 21.38 |
| R7 f-DAgger | T2/all | 27/72 | 37.5% [25.0, 51.4] | 13.70 | 41.57 |
| R8 f-DAgger + Gate | T2/all | 41/72 | 56.9% [44.4, 69.4] | 8.69 | 29.88 |
| expert | T2/all | 24/24 | 100.0% [100.0, 100.0] | 6.29 | 1.10 |
| R5 DAgger | T3/all | 30/78 | 38.5% [25.6, 52.6] | 14.29 | 24.23 |
| R6 DAgger + Gate | T3/all | 26/78 | 33.3% [19.2, 47.4] | 13.95 | 46.77 |
| R7 f-DAgger | T3/all | 29/78 | 37.2% [21.8, 52.6] | 15.09 | 58.45 |
| R8 f-DAgger + Gate | T3/all | 50/78 | 64.1% [50.0, 76.9] | 13.22 | 22.23 |
| expert | T3/all | 26/26 | 100.0% [100.0, 100.0] | 12.31 | 1.13 |
| R5 DAgger | T3/high | 13/48 | 27.1% [14.6, 39.6] | 13.04 | 50.84 |
| R6 DAgger + Gate | T3/high | 17/48 | 35.4% [18.8, 52.1] | 12.85 | 46.85 |
| R7 f-DAgger | T3/high | 14/48 | 29.2% [14.6, 47.9] | 13.57 | 126.92 |
| R8 f-DAgger + Gate | T3/high | 36/48 | 75.0% [58.3, 87.5] | 12.18 | 17.75 |
| expert | T3/high | 16/16 | 100.0% [100.0, 100.0] | 11.90 | 1.05 |
| R5 DAgger | T3/low | 17/30 | 56.7% [30.0, 80.0] | 15.41 | 11.13 |
| R6 DAgger + Gate | T3/low | 9/30 | 30.0% [6.7, 56.7] | 16.81 | 43.32 |
| R7 f-DAgger | T3/low | 15/30 | 50.0% [23.3, 76.7] | 16.99 | 26.13 |
| R8 f-DAgger + Gate | T3/low | 14/30 | 46.7% [26.7, 63.3] | 17.26 | 22.85 |
| expert | T3/low | 10/10 | 100.0% [100.0, 100.0] | 15.15 | 1.46 |
| R5 DAgger | T4/all | 68/72 | 94.4% [88.9, 100.0] | 10.07 | 8.91 |
| R6 DAgger + Gate | T4/all | 61/72 | 84.7% [72.2, 94.4] | 9.60 | 13.86 |
| R7 f-DAgger | T4/all | 66/72 | 91.7% [81.9, 100.0] | 10.20 | 17.46 |
| R8 f-DAgger + Gate | T4/all | 68/72 | 94.4% [86.1, 100.0] | 9.94 | 7.13 |
| expert | T4/all | 24/24 | 100.0% [100.0, 100.0] | 9.65 | 0.62 |
| R5 DAgger | T5/all | 50/72 | 69.4% [56.9, 80.6] | 26.75 | 5.20 |
| R6 DAgger + Gate | T5/all | 44/72 | 61.1% [50.0, 72.2] | 28.98 | 18.70 |
| R7 f-DAgger | T5/all | 36/72 | 50.0% [38.9, 61.1] | 25.81 | 29.81 |
| R8 f-DAgger + Gate | T5/all | 57/72 | 79.2% [68.1, 88.9] | 28.26 | 14.67 |
| expert | T5/all | 24/24 | 100.0% [100.0, 100.0] | 26.39 | 0.74 |

## Efficiency and runtime

| Model | Parameters | Warmed latency p50 / p95 / p99, µs |
|---|---:|---|
| R5-baseline-dagger-nodev-r3__MLP01-256x3__s0__v3__2026-11-05 | 141065 | 66.27 / 68.60 / 70.23 |
| R5-baseline-dagger-nodev-r3__MLP01-256x3__s1__v3__2026-11-05 | 141065 | 65.89 / 68.53 / 70.80 |
| R5-baseline-dagger-nodev-r3__MLP01-256x3__s2__v3__2026-11-05 | 141065 | 69.50 / 72.82 / 75.70 |
| R6-gate-dagger-nodev-r3__MLP01-256x3__s0__v3__2026-11-05 | 141321 | 75.32 / 78.81 / 83.60 |
| R6-gate-dagger-nodev-r3__MLP01-256x3__s1__v3__2026-11-05 | 141321 | 74.81 / 77.84 / 79.08 |
| R6-gate-dagger-nodev-r3__MLP01-256x3__s2__v3__2026-11-05 | 141321 | 74.81 / 77.74 / 78.91 |
| R7-dart-dagger-nodev-r3__MLP01-256x3__s0__v3__2026-11-05 | 141065 | 66.90 / 69.63 / 70.78 |
| R7-dart-dagger-nodev-r3__MLP01-256x3__s1__v3__2026-11-05 | 141065 | 67.99 / 70.47 / 72.28 |
| R7-dart-dagger-nodev-r3__MLP01-256x3__s2__v3__2026-11-05 | 141065 | 66.70 / 69.18 / 71.01 |
| R8-dart-gate-dagger-nodev-r3__MLP01-256x3__s0__v3__2026-11-05 | 141321 | 74.07 / 76.55 / 78.18 |
| R8-dart-gate-dagger-nodev-r3__MLP01-256x3__s1__v3__2026-11-05 | 141321 | 76.19 / 79.21 / 80.75 |
| R8-dart-gate-dagger-nodev-r3__MLP01-256x3__s2__v3__2026-11-05 | 141321 | 75.31 / 77.46 / 78.99 |
| Markovian-expert-A | None | 15.62 / 16.74 / 17.46 |

Coverage by model/suite: `summary.csv` includes expected/evaluated/missing outcomes and pending expert verdicts. Partial estimates omit confidence intervals; missing jobs are not counted as failures.

## Runtime failures

{}

Expert failures describe this controller under this protocol, not physical infeasibility. The expert-success denominator is identical across models with complete coverage; no scenes are removed or redrawn.

Full per-seed/per-model statistics are in `summary.csv`; every rollout is in `rollouts.csv`. Atomic `jobs/*.json` preserve histories and exception details. `run.json` pins model, config, scene, protocol, code and runtime provenance.
