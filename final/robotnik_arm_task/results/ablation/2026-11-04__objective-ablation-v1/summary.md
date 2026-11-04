# 2026-11-04__objective-ablation-v1

Manifest: `2026-11-04__objective-ablation-v1.yaml`; architectures MLP01; seeds [0, 1, 2]; validation 32 scenarios per model; terminal checkpoint only.

| Row | Architecture | Parameters | Full student: hold per seed | Hold rate (Wilson 95%) | Reach per seed | Median final error | Arm only: hold per seed | Hold test (arm, 8 scenarios) per seed | Max stiffness eig (N·m/rad) |
|---|---|---|---|---|---|---|---|---|---|
| R4-dart-gate-bc | MLP01-256x3 | 141321 | 24/28/24 (of 32) | 0.79 (0.70-0.86) | 29/30/26 | 2.8 mm | 26/27/24 | 7/7/7 | 2.2 |
