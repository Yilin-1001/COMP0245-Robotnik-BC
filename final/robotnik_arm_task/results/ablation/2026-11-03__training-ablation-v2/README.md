# 2026-11-03__training-ablation-v2: DART-style noise + few DAgger rounds

Supplement to [2026-11-01__training-ablation-v2](../2026-11-01__training-ablation-v2/README.md). It asks whether DART-style noise (fixed envelope, no noise optimisation) plus 1-2 DAgger rounds can match DAgger-only training at a lower cost.

- **DAgger only:** R2-gate (r0), then R6-gate-dagger r1-r3.
- **DART + DAgger:** R4-dart-gate (r0), then R8-dart-gate-dagger r1-r3.

No new data was collected. Seeds 1-2 of rounds 1-2 were trained on the existing 2026-11-01 DAgger datasets with the same recipe (`experiments/dagger_extra_seeds.py`). Seed 0 of each round is the 2026-11-01 model that collected the next round's data.

```bash
python -m experiments.dagger_extra_seeds --date 2026-11-03 --rows R8-dart-gate-dagger R6-gate-dagger --rounds 1 2 --seeds 1 2 --workers 20
python -m evaluation.test_suite --models <12 step_42000.npz: R6/R8 r1-r2, seeds 0-2> --out results/test/2026-11-03_231207__dart-low-round-dagger --workers 20
python -m experiments.dagger_low_round_report --tests results/test/2026-11-01_215340__R1-R8 results/test/2026-11-03_231207__dart-low-round-dagger \
    --out results/ablation/2026-11-03__training-ablation-v2/low-round
```

The expert's 240 test rollouts are identical in both test runs, so the two runs can be merged.

## Cost

Cost is the wall-clock added beyond the shared clean data and the round-0 training (42k updates, about 4 min, the same for R2 and R4). It was measured on the reference machine with 20 workers.

- **DAgger rounds:** times come from the 2026-11-01 log (collect + build + train + validate, seed-0 chain).
- **DART:** the time is the collection's first-to-last episode write (442 episodes). It needs no student, so it can run alongside the clean collection.

| Round | DAgger only: min / expert labels | DART + DAgger: min / expert labels | Sequential train→collect cycles |
|---|---|---|---|
| r0 | 0 / 0 | 3.3 / 5.3 M | 0 |
| r1 | 12.0 / 10.1 M | 11.1 / 9.4 M | 1 |
| r2 | 20.4 / 14.7 M | 18.8 / 13.0 M | 2 |
| r3 | 28.4 / 18.6 M | 26.0 / 16.5 M | 3 |

## Unified test success (3 seeds, 240 scenes; 95% scene-cluster bootstrap)

| Group | DAgger r0 | r1 | r2 | r3 | DART r0 | +r1 | +r2 | +r3 |
|---|---|---|---|---|---|---|---|---|
| In range (T0+T4+T5) | 16.7 | 75.3 | 86.9 | 92.0 | 90.2 | 92.6 | **96.1** | 96.1 |
| T0 core | 20.8 | 88.0 | 93.2 | 97.9 | 96.9 | 96.9 | 99.0 | 99.0 |
| Range edges (T2+T3) | 2.4 | 22.6 | 23.8 | 33.9 | 51.2 | 47.0 | 64.3 | **73.2** |
| Far (T1, 3-15 m) | 0.0 | 27.3 | 95.4 | **100** | 90.7 | 51.9 | 56.9 | 44.4 |
| All 240 | 8.3 | 48.6 | 74.7 | 80.8 | **81.2** | 69.7 | 76.9 | 75.3 |

Full table, with every band, per-seed counts and errors: `low-round/success_by_round.csv`. Figure: `low-round/cost_vs_success.png`.

## Far-distance caveat (T1)

DART + DAgger failures on T1 are navigation failures: the base finishes 1-12 m from the target. The arm is not the cause.

DAgger-only's far success comes from its data. R2's broken round-0 student drove away from the targets (up to 798 m), so the expert's round-1 labels include 3.1 M rows with a base-target distance above 3 m. Those rows stay in every later round.

Every other dataset (clean, DART, R6 r2-r3, R8 r1-r3) stops at 3.10 m. T1 is therefore partly in distribution for R6 but pure extrapolation for R4 and R8. R4's 90.7% is genuine extrapolation. Retraining on the DAgger mixtures makes that extrapolation seed-dependent (e.g. R8-r1 6-10 m: 24 / 0 / 8 of 24 per seed).
