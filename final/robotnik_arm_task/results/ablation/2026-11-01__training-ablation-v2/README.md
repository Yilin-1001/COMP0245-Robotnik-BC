# 2026-11-01__training-ablation-v2

DAgger rows R5–R8 of the training-method ablation, version 2. This is the reference v2 run.
Method: [project pipeline, Section 7.2](../../../../../docs/2026-10-30__guide__project-pipeline.md#72-version-v2-dagger-r5r8).

**Status.** Validation only. The 64 test scenarios have not been run for v2; they will be run once after the final model is chosen.
Run on the reference machine on 1 November 2026, 16:55–19:24, at commit `780c476`:

```bash
for ROW in R5-baseline-dagger R6-gate-dagger R7-dart-dagger R8-dart-gate-dagger; do
    python -m experiments.dagger_rounds --row $ROW --rounds 3 --date 2026-11-01 --final-seeds 1 2 --workers 20
done
```

Round 0 is the base row's v1 seed-0 model from `checkpoints/ablation/2026-10-28__training-ablation-v1/`.
The round is chosen on seed-0 validation (a tie goes to the later round), then seeds 1–2 are trained for that round.

| Row | Base row (v1) | Seed-0 hold, rounds 0 / 1 / 2 / 3 (of 32) | Chosen round | Hold, seeds 0 / 1 / 2 | Pooled |
|---|---|---|---|---|---|
| R5-baseline-dagger | R1-baseline | 0 / 15 / 21 / 28 | r3 | 28 / 29 / 29 | 87/96 |
| R6-gate-dagger | R2-gate | 2 / 27 / 29 / 31 | r3 | 31 / 29 / 30 | 90/96 |
| R7-dart-dagger | R3-dart | 0 / 24 / 28 / 30 | r3 | 30 / 31 / 30 | 91/96 |
| R8-dart-gate-dagger | R4-dart-gate | 29 / 28 / 28 / 31 | r3 | 31 / 31 / 31 | 93/96 |

| Path | Contents | In Git |
|---|---|---|
| `dagger/<row>/summary.csv` | One line per (round, seed): hold, reach, final errors, model path | Yes |
| `dagger/run.log` | Full log of the run above | No |
| `figures/training/` | Training curves per row and `dagger_rounds` (`python -m evaluation.plot_training_curves --experiments 2026-11-01__training-ablation-v2`) | No |

Models: `checkpoints/ablation/2026-11-01__training-ablation-v2/`. DAgger episodes: `data/Markovian/DAgger/2026-11-01__collection__r<k>-<digest>/`. Feature datasets: `data/datasets/A/dagger-R<n>-v2-r<k>__<features>/`.
A teammate's earlier development run is in [`2026-10-30__training-ablation-v2`](../2026-10-30__training-ablation-v2/README.md).
