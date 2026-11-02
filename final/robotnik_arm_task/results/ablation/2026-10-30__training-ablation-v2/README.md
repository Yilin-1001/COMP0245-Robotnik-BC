# 2026-10-30__training-ablation-v2

DAgger rows R5–R8 and the supplementary R4 controls of the training-method ablation, version 2.
Write-up: [docs/reports/2026-10-31__report__dagger-ablation-and-final-model.md](../../../../../docs/reports/2026-10-31__report__dagger-ablation-and-final-model.md).

**Status.** Development results only, run on a teammate's machine (30–31 Oct, WSL2). Validation numbers below are from that machine.
The reference v2 run is [`2026-11-01__training-ablation-v2`](../2026-11-01__training-ablation-v2/README.md); use its numbers where the two differ.
The 64 test scenarios have **not** been run for v2 under the project protocol; that test is still to be done.
Checkpoints, datasets and per-rollout details are git-ignored. Only the summaries in this folder are tracked.

| Row | Base row (v1) | Added data | Chosen round | Validation hold, seeds 0/1/2 (of 32) |
|---|---|---|---|---|
| R5-baseline-dagger | R1-baseline | DAgger rounds 1..k | r3 | 30 / 29 / 30 = 89/96 |
| R6-gate-dagger | R2-gate | DAgger rounds 1..k | r3 | 31 / 30 / 28 = 89/96 |
| R7-dart-dagger | R3-dart | DAgger rounds 1..k | r3 | 31 / 27 / 30 = 88/96 |
| R8-dart-gate-dagger | R4-dart-gate | DAgger rounds 1..k | r2 | 31 / 31 / 31 = 93/96 |

The round is chosen on seed-0 validation, then seeds 1–2 are trained for that round (see the pipeline document, Section 6).
In this run, round 0 was the teammate's own retraining of R1–R4 (dates 2026-10-29/30), not the v1 models in
`checkpoints/ablation/2026-10-28__training-ablation-v1/`. From now on `experiments.dagger_rounds` starts from those v1 models by default.

## Layout

```
dagger/<row>/summary.csv          one line per trained model: round, seed, validation hold/reach, errors
probes/far/<models>/summary.json  far-target probe (evaluation.far_probe, development seeds 980000+)
probes/robust/<models>/summary.json  robustness probe (evaluation.robust_probe, development seeds 990000+)
```

Probe folders are named after the models they contain, for example `R4-R8r2-s0` = R4 and R8 round 2, seed 0.

## Legacy names

These files were produced before the rename on 2026-11-01. **Run IDs inside them keep the legacy names.**
Use this table to read them, and to rename the teammate's local checkpoints and datasets.

| Legacy name | Current name |
|---|---|
| `R1D-dagger-R1v1-r<k>` | `R5-baseline-dagger-r<k>` |
| `R2D-dagger-R2v1-r<k>` | `R6-gate-dagger-r<k>` |
| `R3D-dagger-R3v1-r<k>` | `R7-dart-dagger-r<k>` |
| `R6-dagger-v1-r<k>` | `R8-dart-gate-dagger-r<k>` |
| `R5-dart-gate-split` | `R4-dart-gate-split` |
| `R4H-dart-half-gate` | `R4-dart-gate-s0375` |
| `R4D-dart-double-gate` | `R4-dart-gate-s150` |
| `R4X4-dart4-gate` | `R4-dart-gate-x4` |
| datasets `daggerR1v1-r<k>`, `daggerR2v1-r<k>`, `daggerR3v1-r<k>`, `daggerv1-r<k>` | `dagger-R5-v2-r<k>`, `dagger-R6-v2-r<k>`, `dagger-R7-v2-r<k>`, `dagger-R8-v2-r<k>` |
| run suffix `__v1__2026-10-30`, `__v1__2026-10-31` | `__v2__2026-10-30` |
| `checkpoints/ablation/2026-10-30__training-ablation-v1/`, `2026-10-31__training-ablation-v1/` | `checkpoints/ablation/2026-10-30__training-ablation-v2/` |
| `results/dagger/{R1v1,R2v1,R3v1,v1}/` | `dagger/{R5-baseline-dagger,R6-gate-dagger,R7-dart-dagger,R8-dart-gate-dagger}/` |
| `results/far_probe/{v1,v1-s1,v1-s2}/` | `probes/far/R4-R8r2-s{0,1,2}/` |
| `results/far_probe/v1-R6-rounds/` | `probes/far/R8r1-R8r3-s0/` |
| `results/far_probe/v1-rows/` | `probes/far/R5r3-R7r3-s0/` |
| `results/far_probe/v1-R1D-s12/` | `probes/far/R5r3-s12/` |
| `results/far_probe/v1-R2D-s0/`, `v1-R2D-s12/` | `probes/far/R6r3-s0/`, `probes/far/R6r3-s12/` |
| `results/robust_probe/v1/` | `probes/robust/R4-R5r3-R6r3-R8r2-s0/` |
| `results/robust_probe/v1-s12/` | `probes/robust/R6r3-R8r2-s12/` |

The shell scripts that ran these experiments use the legacy names and the teammate's paths. They are kept as a record in
`Archive/final/robotnik_arm_task/experiments/scripts/`.
