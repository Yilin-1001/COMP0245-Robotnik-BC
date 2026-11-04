# 2026-11-04__dart-full: full DART (Laskey et al., 2017) vs DART-style noise

This run compares full DART, with learner-fitted noise optimisation, against our DART-style OU noise and against DAgger.

All rows share the inputs (features-v1-gate), the recipe (BC+J, Huber, 42k updates, lr 1e-3, arm/wheel 0.5/0.5) and the clean data. Round 0 is R2-gate (BC on the clean data).

## Method

The method follows the paper. Per iteration k:

1. Estimate Sigma_hat_k = (1/T) E[sum_t e e^T], where e = pi_{k-1}(x) - pi*(x). The estimate uses 32 fresh held-out demonstrations collected with the previous noise; iteration 1 uses clean ones. These demonstrations are never trained on and are not the validation or test scenes.
2. Set Sigma_k = alpha / (T tr Sigma_hat) * Sigma_hat. The alpha = 1 run uses alpha = T tr Sigma_hat, the paper's MuJoCo choice. The alpha = 6 run uses alpha = 6 T tr Sigma_hat_1, close to the paper's grasping choice.
3. Collect 221 fitting scenarios x 2 draws (442 episodes, the same as R4's DART data). The noise is i.i.d. N(0, Sigma_k) at every 1 kHz step for the whole episode, with no phase gating. Labels are the clean expert actions.
4. Retrain on clean253 + D_1..D_k.

Code:

- Collector: `data_collection/collect_dart_full.py`
- Iteration driver: `experiments/dart_full.py`
- alpha variant: `experiments/dart_full_alpha.py`
- Fitted covariances: `covariance/sigma_k*.npz` (with `.json` provenance)
- Per-step wall time: `timing.csv`
- Validation results: `summary.csv`

```bash
python -m experiments.dart_full --date 2026-11-04 --iterations 3 --extra-seeds 1 2 --workers 20   # stopped during extra seeds (user request)
python -m experiments.dart_full_alpha --date 2026-11-04 --scale 6 --seeds 0 1 2 --workers 20
python <wrapper> --models <3 alpha=6 models> --out results/test/2026-11-04_134912__dart-full-a6 --workers 20
python report/07_training_time_comparison/code/make_comparison.py --tests results/test/2026-11-01_215340__R1-R8 \
    results/test/2026-11-03_231207__dart-low-round-dagger results/test/2026-11-04_134912__dart-full-a6 \
    --out results/ablation/2026-11-04__dart-full/comparison
```

The unified test wrapper uses the frozen `unified-test-v1` manifest (sha256 0a916445...). Rebuilding the manifest only adds this run's four DART inventories to `source_audit`. The scenes are identical, and the audit finds zero overlaps. The expert's 240 rollouts are identical to the 11-01 run.

The alpha = 1 run was stopped by user request during the extra seeds. Its r1 s2 directory is incomplete and unused. alpha = 1 was not run on the unified test.

## Validation (32 scenes, sustained hold)

| Row | Seeds 0 / 1 / 2 |
|---|---|
| Full DART alpha = 1, r1 | 7 / 10 / - |
| Full DART alpha = 1, r2 | 6 / - / - |
| Full DART alpha = 1, r3 | 12 / - / - |
| **Full DART alpha = 6, r1** | 18 / 21 / 17 (56/96) |
| R4 DART-style (same 442 episodes) | 29 / 28 / 26 (83/96) |
| DAgger only r1 | 27 / 23 / 21 (71/96) |

## Applied noise (measured from the collected actions)

| Noise | Arm per-step RMS (N·m) | Arm impulse RMS, 100 ms (N·m·s) | Arm impulse RMS, 500 ms (N·m·s) | Noisy steps |
|---|---|---|---|---|
| DART-style OU (tau 0.2 s) | 0.50 | 0.046 | 0.178 | 87% |
| Full DART alpha = 6 | 0.88 | 0.0088 | 0.020 | 100% |
| Full DART alpha = 1 | 0.36 | 0.0036 | 0.008 | 100% |

## Unified test (3 seeds, % success)

| Group | R4 DART-style r0 | Full DART alpha = 6 r1 | DAgger only r1 | DART + 1 DAgger | DART + 2 DAgger | DAgger only r3 |
|---|---|---|---|---|---|---|
| T0 core | 96.9 | 72.9 | 88.0 | 96.9 | 99.0 | 97.9 |
| T4 translated start | 94.4 | 68.1 | 87.5 | 95.8 | 97.2 | 95.8 |
| T5 long hold | 68.1 | 22.2 | 29.2 | 77.8 | 87.5 | 72.2 |
| T2 near | 47.2 | 11.1 | 26.4 | 54.2 | 65.3 | 50.0 |
| T3 low | 16.7 | 2.1 | 27.1 | 25.0 | 47.9 | 37.5 |
| T3 high | 91.7 | 54.2 | 12.5 | 58.3 | 79.2 | 6.2 |
| T1 far 3-6 m | 97.2 | 26.4 | 50.0 | 90.3 | 100 | 100 |
| T1 far 6-10 m | 97.2 | 0.0 | 19.4 | 44.4 | 61.1 | 100 |
| T1 far 10-15 m | 77.8 | 0.0 | 12.5 | 20.8 | 9.7 | 100 |
| **All 240** | **81.2** | **36.0** | 48.6 | 69.7 | 76.9 | 80.8 |

## Cost

Wall time is beyond the shared clean data and round-0 training, on 20 workers.

| Pipeline | Added wall time | Sequential train→collect cycles | Student executes? |
|---|---|---|---|
| R4 DART-style | 3.3 min | 0 | no |
| Full DART, 1 iteration | 8.7 min | 1 | no |
| Full DART, 3 iterations | 25.7 min | 3 | no |
| DAgger only, 1 / 3 rounds | 12.0 / 28.4 min | 1 / 3 | yes |
| DART + 1 / 2 DAgger rounds | 11.1 / 18.8 min | 1 / 2 | yes |

Full DART also needs the round-0 student, R2, trained before its first noisy collection. The DART-style collection needs no student and can run alongside the clean collection.

## Update: alpha = 6, three iterations (run 14:35-16:15)

Iterations 2 and 3 were run with `python -m experiments.dart_full --date 2026-11-04 --scale 6 --start-iteration 2 --iterations 3 --extra-seeds 1 2`. Each iteration re-estimates Sigma_hat_k from the previous alpha = 6 learner on held-out demonstrations collected with the previous alpha = 6 noise, then sets Sigma_k = 6 Sigma_hat_k.

The fitted arm std (N·m, after the 6x scale) was 0.37-1.11 at k = 1, 0.13-1.06 at k = 2 and 0.19-1.86 at k = 3.

The unified test is `results/test/2026-11-04_151122__dart-full-a6-r2r3`, run through the same frozen-manifest wrapper. The expert's 240 rollouts are identical to the 11-01 run.

| alpha = 6 | Validation, seeds 0/1/2 | All 240 | In range | Range edges | Far | Added wall time |
|---|---|---|---|---|---|---|
| r1 | 18/21/17 (56/96) | 36.0% | 61.0% | 20.8% | 8.8% | 8.7 min |
| r2 | 17/18/20 (55/96) | 37.1% | 62.8% | 19.0% | 11.1% | 17.2 min |
| r3 | 23/22/17 (62/96) | 39.4% [33.8, 44.8] | 66.4% | 19.0% | 13.4% | 25.7 min |
| R4 DART-style r0 | 29/28/26 (83/96) | 81.2% | 90.2% | 51.2% | 90.7% | 3.3 min |
| DAgger only r3 | 31/29/30 (90/96) | 80.8% | 92.0% | 33.9% | 100% | 28.4 min |
| DART + DAgger r2 | 28/29/28 (85/96) | 76.9% | 96.1% | 64.3% | 56.9% | 18.8 min |

Full DART improves slowly with iterations (36.0 → 37.1 → 39.4% on all 240 scenes). After three iterations and 25.7 min it reaches less than half of R4's success, and R4 needs 3.3 min and no learner. The figure and full table are in `comparison/`.

The comparison script, formerly `experiments/dagger_low_round_report.py`, now lives at `report/07_training_time_comparison/code/make_comparison.py`. The final figures, tables and data for the report are in [`report/07_training_time_comparison/`](../../../report/07_training_time_comparison/README.md).
