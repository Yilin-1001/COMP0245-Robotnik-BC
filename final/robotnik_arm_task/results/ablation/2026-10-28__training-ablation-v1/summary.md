# 2026-10-28__training-ablation-v1

Manifest: `2026-10-28__training-ablation-v1.yaml`; architecture MLP01; seeds [0, 1, 2]; validation 32 scenarios per model; terminal checkpoint only.

| Row | Full student: hold per seed | Hold rate (Wilson 95%) | Reach per seed | Median final error | Arm only: hold per seed | Hold test (arm, 8 scenarios) per seed | Max stiffness eig (N·m/rad) |
|---|---|---|---|---|---|---|---|
| R1-baseline | 0/1/0 (of 32) | 0.01 (0.00-0.06) | 3/5/2 | 929.7 mm | 2/2/0 | 1/4/3 | 146.2 |
| R2-gate | 2/3/5 (of 32) | 0.10 (0.06-0.18) | 5/6/14 | 1050.5 mm | 1/2/6 | 0/2/1 | 139.7 |
| R3-dart | 0/1/0 (of 32) | 0.01 (0.00-0.06) | 0/1/0 | 293.4 mm | 1/3/3 | 8/8/8 | 1.3 |
| R4-dart-gate | 29/27/27 (of 32) | 0.86 (0.78-0.92) | 32/29/29 | 1.3 mm | 26/25/27 | 7/8/8 | 1.7 |
