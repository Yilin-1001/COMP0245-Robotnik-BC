# Small-network DART + DAgger (R8 end to end)

Date 2026-11-05. Pipeline minutes: seed-0 chain (collect + build + train + validate), all rounds (the real cost: the chosen round is only known after the last round) and, for reference, up to the chosen round. Latency: isolated single-thread NumPy predict(), batch 1. Test: 240-scene unified suite, 3 seeds pooled.

| Network | Params | FLOPs | .npz KB | Latency p50 / p99 (µs) | Train min / model | Pipeline min, all rounds | Chosen round | Pipeline min to chosen round | Validation s0 by round | Validation chosen (s0/s1/s2) | Validation pooled | Test in range (T0+T4+T5) | Test edges (T2+T3) | Test far (T1) | Test all 240 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MLP02 | 3337 | 6080 | 42.0 | 33.6 / 99.3 | 7.87 | 54.8 | 3 | 54.8 | 19/32 / 22/32 / 27/32 / 28/32 | 28 / 28 / 30 | 86/96 | 89.0% (299/336) | 51.2% (86/168) | 70.4% (152/216) | 74.6% (537/720) |
| MLP04 | 10761 | 20352 | 99.9 | 33.8 / 100.8 | 7.71 | 47.2 | 2 | 33.9 | 24/32 / 26/32 / 29/32 / 28/32 | 29 / 27 / 31 | 87/96 | 90.5% (304/336) | 53.0% (89/168) | 65.3% (141/216) | 74.2% (534/720) |
| MLP06 | 37897 | 73472 | 312.0 | 34.4 / 58.6 | 7.97 | 50.1 | 3 | 50.1 | 26/32 / 25/32 / 29/32 / 29/32 | 29 / 29 / 29 | 87/96 | 89.9% (302/336) | 57.7% (97/168) | 70.4% (152/216) | 76.5% (551/720) |
| teacher A |  |  |  |  |  |  |  |  |  |  |  | 100.0% (112/112) | 89.3% (50/56) | 100.0% (72/72) | 97.5% (234/240) |

Training device: cuda:NVIDIA GeForce RTX 3070 Ti Laptop GPU. Inputs: `results/ablation/2026-11-05__2026-11-05__small-net-dagger-nodev-v2/<arch>/{summary,cost}.csv`, `results/test/2026-11-05__small-net-dagger-nodev`.
