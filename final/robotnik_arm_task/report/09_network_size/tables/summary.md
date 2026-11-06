# Network size: summary tables

Rebuilt by `code/make_tables.py`. Rates are % with 95% Wilson intervals and (successes/rollouts).

## Validation (32 scenes, seeds 0-2 pooled)

| size | parameters | ou_gate_per_seed | ou_gate_pooled | f_dagger_gate_round | f_dagger_gate_per_seed | f_dagger_gate_pooled | gain_pp | chain_round0_seed0 | chain_seed0_by_round |
|---|---|---|---|---|---|---|---|---|---|
| 32×3 | 3337 | 25 / 25 / 22 | 75.0 [65.5, 82.6] (72/96) | 3 | 28 / 28 / 30 | 89.6 [81.9, 94.2] (86/96) | 14.6 | 19/32 | 19/32 / 22/32 / 27/32 / 28/32 |
| 64×3 | 10761 | 25 / 25 / 26 | 79.2 [70.0, 86.1] (76/96) | 2 | 29 / 27 / 31 | 90.6 [83.1, 95.0] (87/96) | 11.5 | 24/32 | 24/32 / 26/32 / 29/32 / 28/32 |
| 128×3 | 37897 | 26 / 27 / 28 | 84.4 [75.8, 90.3] (81/96) | 3 | 29 / 29 / 29 | 90.6 [83.1, 95.0] (87/96) | 6.2 | 26/32 | 26/32 / 25/32 / 29/32 / 29/32 |
| 256×3 | 141321 | 29 / 27 / 27 | 86.5 [78.2, 91.9] (83/96) | 3 | 29 / 29 / 31 | 92.7 [85.7, 96.4] (89/96) | 6.2 |  |  |

## Unified test (240 scenes, seeds 0-2 pooled)

| method | size | in range (T0+T4+T5) | edges (T2+T3) | far (T1) | all 240 |
|---|---|---|---|---|---|
| f-DAgger + Gate | 32×3 | 89.0 [85.2, 91.9] (299/336) | 51.2 [43.7, 58.6] (86/168) | 70.4 [64.0, 76.1] (152/216) | 74.6 [71.3, 77.6] (537/720) |
| f-DAgger + Gate | 64×3 | 90.5 [86.9, 93.2] (304/336) | 53.0 [45.4, 60.4] (89/168) | 65.3 [58.7, 71.3] (141/216) | 74.2 [70.8, 77.2] (534/720) |
| f-DAgger + Gate | 128×3 | 89.9 [86.2, 92.7] (302/336) | 57.7 [50.2, 65.0] (97/168) | 70.4 [64.0, 76.1] (152/216) | 76.5 [73.3, 79.5] (551/720) |
| f-DAgger + Gate | 256×3 | 93.2 [89.9, 95.4] (313/336) | 54.8 [47.2, 62.1] (92/168) | 45.8 [39.3, 52.5] (99/216) | 70.0 [66.6, 73.2] (504/720) |
| OU + Gate (reference) | 256×3 | 90.2 [86.5, 92.9] (303/336) | 51.2 [43.7, 58.6] (86/168) | 90.7 [86.1, 93.9] (196/216) | 81.2 [78.2, 83.9] (585/720) |
| Expert | - | 100.0 [96.7, 100.0] (112/112) | 89.3 [78.5, 95.0] (50/56) | 100.0 [94.9, 100.0] (72/72) | 97.5 [94.7, 98.8] (234/240) |

Expert counts identical on both machines: **True**.

## Efficiency

| size | parameters | flops | params_vs_256x3 | latency_p50_us_this_machine | latency_p99_us_this_machine | npz_kb_this_machine | latency_p50_us_teammate | train_min_per_model_teammate | pipeline_min_all_rounds_teammate | pipeline_min_to_chosen_teammate |
|---|---|---|---|---|---|---|---|---|---|---|
| 32×3 | 3337 | 6080 | 42.3 |  |  |  | 33.6 | 7.87 | 54.8 | 54.8 |
| 64×3 | 10761 | 20352 | 13.1 |  |  |  | 33.8 | 7.71 | 47.2 | 33.9 |
| 128×3 | 37897 | 73472 | 3.7 |  |  |  | 34.4 | 7.97 | 50.1 | 50.1 |
| 256×3 | 141321 | 278016 | 1.0 |  |  |  |  |  |  |  |

## Naming

| report_name | size | row | round | seed | run_id | original_run_id | source |
|---|---|---|---|---|---|---|---|
| f-DAgger + Gate | 32×3 | R8 | 3 | 0 | R8-f-DAgger-Gate-r3__MLP02-32x3__s0__v3__2026-11-05 | R8-dart-gate-dagger-MLP02nodev-r3__MLP02-32x3__s0__v2__2026-11-05 | teammate, feature/small-net-dagger (his machine) |
| f-DAgger + Gate | 32×3 | R8 | 3 | 1 | R8-f-DAgger-Gate-r3__MLP02-32x3__s1__v3__2026-11-05 | R8-dart-gate-dagger-MLP02nodev-r3__MLP02-32x3__s1__v2__2026-11-05 | teammate, feature/small-net-dagger (his machine) |
| f-DAgger + Gate | 32×3 | R8 | 3 | 2 | R8-f-DAgger-Gate-r3__MLP02-32x3__s2__v3__2026-11-05 | R8-dart-gate-dagger-MLP02nodev-r3__MLP02-32x3__s2__v2__2026-11-05 | teammate, feature/small-net-dagger (his machine) |
| f-DAgger + Gate | 64×3 | R8 | 2 | 0 | R8-f-DAgger-Gate-r2__MLP04-64x3__s0__v3__2026-11-05 | R8-dart-gate-dagger-MLP04nodev-r2__MLP04-64x3__s0__v2__2026-11-05 | teammate, feature/small-net-dagger (his machine) |
| f-DAgger + Gate | 64×3 | R8 | 2 | 1 | R8-f-DAgger-Gate-r2__MLP04-64x3__s1__v3__2026-11-05 | R8-dart-gate-dagger-MLP04nodev-r2__MLP04-64x3__s1__v2__2026-11-05 | teammate, feature/small-net-dagger (his machine) |
| f-DAgger + Gate | 64×3 | R8 | 2 | 2 | R8-f-DAgger-Gate-r2__MLP04-64x3__s2__v3__2026-11-05 | R8-dart-gate-dagger-MLP04nodev-r2__MLP04-64x3__s2__v2__2026-11-05 | teammate, feature/small-net-dagger (his machine) |
| f-DAgger + Gate | 128×3 | R8 | 3 | 0 | R8-f-DAgger-Gate-r3__MLP06-128x3__s0__v3__2026-11-05 | R8-dart-gate-dagger-MLP06nodev-r3__MLP06-128x3__s0__v2__2026-11-05 | teammate, feature/small-net-dagger (his machine) |
| f-DAgger + Gate | 128×3 | R8 | 3 | 1 | R8-f-DAgger-Gate-r3__MLP06-128x3__s1__v3__2026-11-05 | R8-dart-gate-dagger-MLP06nodev-r3__MLP06-128x3__s1__v2__2026-11-05 | teammate, feature/small-net-dagger (his machine) |
| f-DAgger + Gate | 128×3 | R8 | 3 | 2 | R8-f-DAgger-Gate-r3__MLP06-128x3__s2__v3__2026-11-05 | R8-dart-gate-dagger-MLP06nodev-r3__MLP06-128x3__s2__v2__2026-11-05 | teammate, feature/small-net-dagger (his machine) |
| f-DAgger + Gate | 256×3 | R8 | 3 | 0 | R8-f-DAgger-Gate-r3__MLP01-256x3__s0__v3__2026-11-05 | R8-dart-gate-dagger-nodev-r3__MLP01-256x3__s0__v3__2026-11-05 | report v3 (this machine; checkpoints/2026-11-05final) |
| f-DAgger + Gate | 256×3 | R8 | 3 | 1 | R8-f-DAgger-Gate-r3__MLP01-256x3__s1__v3__2026-11-05 | R8-dart-gate-dagger-nodev-r3__MLP01-256x3__s1__v3__2026-11-05 | report v3 (this machine; checkpoints/2026-11-05final) |
| f-DAgger + Gate | 256×3 | R8 | 3 | 2 | R8-f-DAgger-Gate-r3__MLP01-256x3__s2__v3__2026-11-05 | R8-dart-gate-dagger-nodev-r3__MLP01-256x3__s2__v3__2026-11-05 | report v3 (this machine; checkpoints/2026-11-05final) |
| OU + Gate | 32×3 | R4 | 0 | 0 | R4-OU-Gate__MLP02-32x3__s0__v1__2026-11-02 | R4-dart-gate__MLP02-32x3__s0__v1__2026-11-02 | 2026-11-02 architecture ablation (this machine) |
| OU + Gate | 32×3 | R4 | 0 | 1 | R4-OU-Gate__MLP02-32x3__s1__v1__2026-11-02 | R4-dart-gate__MLP02-32x3__s1__v1__2026-11-02 | 2026-11-02 architecture ablation (this machine) |
| OU + Gate | 32×3 | R4 | 0 | 2 | R4-OU-Gate__MLP02-32x3__s2__v1__2026-11-02 | R4-dart-gate__MLP02-32x3__s2__v1__2026-11-02 | 2026-11-02 architecture ablation (this machine) |
| OU + Gate | 64×3 | R4 | 0 | 0 | R4-OU-Gate__MLP04-64x3__s0__v1__2026-11-02 | R4-dart-gate__MLP04-64x3__s0__v1__2026-11-02 | 2026-11-02 architecture ablation (this machine) |
| OU + Gate | 64×3 | R4 | 0 | 1 | R4-OU-Gate__MLP04-64x3__s1__v1__2026-11-02 | R4-dart-gate__MLP04-64x3__s1__v1__2026-11-02 | 2026-11-02 architecture ablation (this machine) |
| OU + Gate | 64×3 | R4 | 0 | 2 | R4-OU-Gate__MLP04-64x3__s2__v1__2026-11-02 | R4-dart-gate__MLP04-64x3__s2__v1__2026-11-02 | 2026-11-02 architecture ablation (this machine) |
| OU + Gate | 128×3 | R4 | 0 | 0 | R4-OU-Gate__MLP06-128x3__s0__v1__2026-11-02 | R4-dart-gate__MLP06-128x3__s0__v1__2026-11-02 | 2026-11-02 architecture ablation (this machine) |
| OU + Gate | 128×3 | R4 | 0 | 1 | R4-OU-Gate__MLP06-128x3__s1__v1__2026-11-02 | R4-dart-gate__MLP06-128x3__s1__v1__2026-11-02 | 2026-11-02 architecture ablation (this machine) |
| OU + Gate | 128×3 | R4 | 0 | 2 | R4-OU-Gate__MLP06-128x3__s2__v1__2026-11-02 | R4-dart-gate__MLP06-128x3__s2__v1__2026-11-02 | 2026-11-02 architecture ablation (this machine) |
| OU + Gate | 256×3 | R4 | 0 | 0 | R4-OU-Gate__MLP01-256x3__s0__v1__2026-10-28 | R4-dart-gate__MLP01-256x3__s0__v1__2026-10-28 | report v1 R4 (this machine) |
| OU + Gate | 256×3 | R4 | 0 | 1 | R4-OU-Gate__MLP01-256x3__s1__v1__2026-10-28 | R4-dart-gate__MLP01-256x3__s1__v1__2026-10-28 | report v1 R4 (this machine) |
| OU + Gate | 256×3 | R4 | 0 | 2 | R4-OU-Gate__MLP01-256x3__s2__v1__2026-10-28 | R4-dart-gate__MLP01-256x3__s2__v1__2026-10-28 | report v1 R4 (this machine) |
| OU + Gate (f-DAgger round 0) | 32×3 | R4 | 0 | 0 | R4-OU-Gate__MLP02-32x3__s0__v2__2026-11-04 | R4-dart-gate__MLP02-32x3__s0__v2__2026-11-04 | teammate (his machine), seed 0 only |
| OU + Gate (f-DAgger round 0) | 64×3 | R4 | 0 | 0 | R4-OU-Gate__MLP04-64x3__s0__v2__2026-11-04 | R4-dart-gate__MLP04-64x3__s0__v2__2026-11-04 | teammate (his machine), seed 0 only |
| OU + Gate (f-DAgger round 0) | 128×3 | R4 | 0 | 0 | R4-OU-Gate__MLP06-128x3__s0__v2__2026-11-04 | R4-dart-gate__MLP06-128x3__s0__v2__2026-11-04 | teammate (his machine), seed 0 only |
