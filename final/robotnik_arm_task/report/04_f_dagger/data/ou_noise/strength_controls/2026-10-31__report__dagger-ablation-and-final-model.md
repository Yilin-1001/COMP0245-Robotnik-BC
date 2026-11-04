# DAgger rows R5–R8, DART controls, generalisation, robustness and network size

Date: 31 October 2026, revised 1 November 2026. Status: development results on a teammate's machine (validation and development probes only).
Revision on 1 November: rows renamed to the R5–R8 scheme (pipeline Section 7; legacy names in `results/ablation/2026-10-30__training-ablation-v2/README.md`), and the informal test-set results of this branch were removed. The 64 test scenarios will be run once under the project protocol after the final model is chosen.
The reference v2 run was repeated on 1 November on the reference machine (`results/ablation/2026-11-01__training-ablation-v2/`, pipeline Section 7.2); where its numbers differ from this report, the reference run is used.

Originally written on branch `feature/dart-ablation`. All runs on one machine (i9-12950HX, RTX 3070 Ti Laptop, WSL2).
Teacher A (`markovian-lt0.50-v1`) throughout. Standard predicates unless stated: reach < 10 mm after 8 s, 3 s more,
all of the final 1 s < 10 mm, 40 s cap. Validation = 32 frozen scenes × 3 seeds (96).
Default network MLP01 (256×256×256), recipe of the training-method ablation v1 (BC+J, lr 1e-3, 42,000 updates, Huber).

**Proposal, not yet decided: R6-gate-dagger, round 3** (`R6-gate-dagger-r3`, seeds 0–2).

## 1. Reproduction on a second machine

| | This machine | Original (R4 report) |
|---|---|---|
| R4 validation | 29 / 28 / 26 = 83/96 | 29 / 27 / 27 = 83/96 |

Clean collection is **not bit-identical across machines**: here teacher B fails `train-0220`, so
`build_clean_subsets` refuses; teacher A succeeds on all 256 training scenes but its `training_manifest.csv` hash differs
from `experiments/splits/clean253.json`. `tools/build_a_clean253.py` builds `data/Markovian/Clean_253` from teacher A
only, using the frozen 253 scenario IDs. Pooled results agree. Also: `libgomp1` must be installed (`sudo apt install libgomp1`)
or Pinocchio fails to import.

## 2. DAgger method

`data_collection/collect_dagger.py` reuses `collect.run_episode` unchanged, through a proxy environment that executes the
**student's** command while teacher A labels the same pre-action observation (β = 0, no takeover, 1 kHz). Episodes keep
the standard stop rule (3 s after first reach, or 40 s); failed episodes are kept. Each round: the 221 fitting scenes plus
64 fresh development scenes (ids `dev-r<k>-<i>`, 70% target height 0.1–0.5 m, 50% distance 0.5–1.0 m, never validation
or test). The next student is trained **from scratch** on the row's own datasets plus all DAgger rounds so far, then
validated; it drives the next round. `experiments/dagger_rounds.py` runs K rounds unattended for any DAgger row R5–R8 (one per base row R1–R4).

## 3. Main results

Rows R1–R4 below are this machine's retraining of v1 (the reference machine's v1 results are in the pipeline document, Section 7.1).

| Row | clean | DART | gate | DAgger | Validation | Far 9–12 m | Robustness (perturbed) |
|---|---|---|---|---|---|---|---|
| Teacher A | | | | | 96/96 | 100% | 225/288 |
| R1-baseline | ✓ | | | | 0/96 | — | — |
| R2-gate | ✓ | | ✓ | | 10/96 | — | — |
| R3-dart | ✓ | ✓ | | | 2/96 | — | — |
| R4-dart-gate | ✓ | ✓ | ✓ | | 83/96 | 21/24 (s0,s2) | 62/96 (s0) |
| R5-baseline-dagger (r3) | ✓ | | | 3 | 89/96 | 36/36 | 55/96 (s0) |
| **R6-gate-dagger (r3)** | ✓ | | ✓ | 3 | 89/96 | **36/36** | **195/288** |
| R7-dart-dagger (r3) | ✓ | ✓ | | 3 | 88/96 | 11/12 (s0) | — |
| R8-dart-gate-dagger (r2) | ✓ | ✓ | ✓ | 2 | 93/96 | **0/24** (s0,s2) | 189/288 |

DAgger rounds, seed 0, validation /32 (round chosen by seed-0 validation, then seeds 1–2 trained):

| Row | start | r1 | r2 | r3 | chosen round, s0/s1/s2 |
|---|---|---|---|---|---|
| R5-baseline-dagger | 0 | 27 | 29 | 30 | r3: 30/29/30 |
| R6-gate-dagger | 1 | 24 | 27 | 31 | r3: 31/30/28 |
| R7-dart-dagger | 0 | 23 | 26 | 31 | r3: 31/27/30 |
| R8-dart-gate-dagger | 29 | 29 | 31 | 30 | r2: 31/31/31 |

## 4. DART controls (all with gate, no DAgger)

| Data | Arm σ (N·m) | Validation |
|---|---|---|
| clean only (R2-gate) | — | 10/96 |
| + 442 DART, σ halved (`R4-dart-gate-s0375`) | 0.375 | 69/96 |
| + 442 DART (`R4-dart-gate`) | 0.75 | 83/96 |
| + 442 DART, σ doubled (`R4-dart-gate-s150`) | 1.5 | 84/96 |
| + 884 DART (`R4-dart-gate-x4`, equal-data control) | 0.75 | 82/96 |
| + 855 DAgger (`R6-gate-dagger-r3`) | — | 89/96 |

Doubling the DART data gives nothing; a similar number of DAgger episodes does. The DAgger gain is not explained by data volume.

## 5. Generalisation beyond the training range (`evaluation/far_probe.py`)

Start-to-target distance bands, 12 fresh scenes each (seeds 980000+), episode cap raised to 80 s (teacher A needs
≈ 40 s for 10 m), teacher A as reference.

| Model | Seeds | 0.5–3 m | 3–6 m | 6–9 m | 9–12 m |
|---|---|---|---|---|---|
| Teacher A | | 12/12 | 12/12 | 12/12 | 12/12 |
| R4 | s0, s2 | 24/24 | 24/24 | 21/24 | 21/24 |
| R8-dart-gate-dagger r1 | s0 | 12/12 | 12/12 | 12/12 | 9/12 |
| R8-dart-gate-dagger r2 | s0, s2 | 24/24 | 22/24 | 6/24 | 0/24 |
| R8-dart-gate-dagger r3 | s0 | 12/12 | 12/12 | 7/12 | 0/12 |
| R5-baseline-dagger r3 | s0–s2 | 33/36 | 36/36 | 36/36 | 36/36 |
| R6-gate-dagger r3 | s0–s2 | 36/36 | 36/36 | 35/36 | 36/36 |
| R7-dart-dagger r3 | s0 | 10/12 | 12/12 | 12/12 | 11/12 |

Only DART + gate + DAgger loses far targets, and more so with each round; the cause is not yet identified.

## 6. Robustness (`evaluation/robust_probe.py`)

In-range scenes, 12 per condition (seeds 990000+). At-target pushes 1 s after first entering 10 mm, with a 10 s
post-reach window; driving push at t = 3 s with a 60 s cap.

| Condition | Teacher A (×3) | R6 r3 (3 seeds) | R8 r2 (3 seeds) | R4 (s0) | R5 r3 (s0) |
|---|---|---|---|---|---|
| nominal | 36/36 | 32/36 | 34/36 | 10/12 | 11/12 |
| friction ×0.5 | 36/36 | 35/36 | 32/36 | 10/12 | 12/12 |
| friction ×1.5 | 36/36 | 33/36 | 33/36 | 12/12 | 11/12 |
| mass ×0.9 | 12/36 | 15/36 | 17/36 | 7/12 | 1/12 |
| mass ×1.1 | 30/36 | 16/36 | 18/36 | 6/12 | 8/12 |
| sensor noise ×20 | 36/36 | 36/36 | 36/36 | 12/12 | 12/12 |
| push base, driving | 18/36 | 15/36 | 6/36 | 2/12 | 1/12 |
| push base, at target | 21/36 | 13/36 | 13/36 | 4/12 | 2/12 |
| push hand, at target | 36/36 | 32/36 | 34/36 | 9/12 | 8/12 |
| **perturbed total** | 225/288 | 195/288 | 189/288 | 62/96 | 55/96 |

Overall robustness is tied between R6 and R8; the consistent difference is the push while driving
(base navigation). Mass changes and base pushes are weak for every student and for teacher A itself (no integrator).

## 7. Network size (validation, 3 seeds)

| Hidden layers | Parameters | R8 r2 data | R6 r3 data | Inference p50 |
|---|---|---|---|---|
| 64×64 | ≈ 6.5 k | 82/96 | 73/96 | 0.27 ms |
| 128×128 | ≈ 21 k | 88/96 | 75/96 | 0.29 ms |
| 128×128×128 | ≈ 38 k | 87/96 | 85/96 | 0.33 ms |
| 256×256×256 | ≈ 141 k | 93/96 | 89/96 | ≈ 0.4 ms |
| 2 × 256×256×256, gate-switched (`R4-dart-gate-split`, R4 data) | ≈ 282 k | 85/96 | — | — |

All are well inside the 1 ms control period. R6 needs three layers. Latency here is the evaluator's p50 on this machine;
the pipeline document quotes 83 µs per control step (features + FK + MLP, one CPU thread) on the reference machine.

## 8. Conclusions

1. DAgger (β = 0, from any starting row) is the main gain: 0–86% → 92–97% on validation.
2. Without DAgger, DART and gate are both needed; with DAgger, neither is necessary.
3. The DAgger gain is not data volume (equal-data DART control: no gain).
4. DART σ = 0.75 N·m is adequate; half is clearly worse, double gives nothing.
5. R6 is close to R8 in range (89 vs 93/96 on validation) and ties it on overall robustness, and is clearly better on
   far targets (36/36 vs 0/24) and on pushes while driving; it also needs no DART. Proposed as the final model
   (not yet decided).

**Caveats.** The DAgger round was chosen on seed-0 validation. The test set had already been viewed for R4 before
DAgger (`2026-10-29__report__r4-test-diagnosis.md`). The far and robustness probes use development seeds only. Some probe
rows use one or two seeds (marked). Validation differences of a few scenes are within noise.

## 9. Reproduce (from `final/robotnik_arm_task/`)

Prerequisites: `clean253__features-v1{,-gate}` and `dart-d2s075x2__features-v1{,-gate}` built as in the root README, and
the v1 models in `checkpoints/ablation/2026-10-28__training-ablation-v1/` (round 0 of each DAgger row).
Runs are not bit-identical across machines (Section 1); compare pooled numbers. The shell scripts of the original run use
the legacy names and are kept in `Archive/final/robotnik_arm_task/experiments/scripts/`.

```bash
# DAgger, 3 rounds per row: round 0 = the base row's v1 seed-0 model; then rounds 1-3; then seeds 1-2 of the best round
for ROW in R5-baseline-dagger R6-gate-dagger R7-dart-dagger R8-dart-gate-dagger; do
  python -m experiments.dagger_rounds --row $ROW --rounds 3 --date 2026-10-30 --final-seeds 1 2 --dev-scenes 64 --workers 20
done
# models:  checkpoints/ablation/2026-10-30__training-ablation-v2/<row>-r<k>__MLP01-256x3__s<seed>__v2__2026-10-30/
# results: results/ablation/2026-10-30__training-ablation-v2/dagger/<row>/summary.csv

# development probes (fresh development seeds, never test scenes)
V2=results/ablation/2026-10-30__training-ablation-v2
python -m evaluation.far_probe --models <a.npz> <b.npz> ... --out $V2/probes/far/<models>
python -m evaluation.robust_probe --models <a.npz> <b.npz> ... --out $V2/probes/robust/<models>

# DART controls: protocols experiments/configs/dart/{d2-s0375-x2,d2-s150-x2,d2-s075-x4}.yaml, collected with
# data_collection.collect_dart and trained as variants R4-dart-gate-s0375, R4-dart-gate-s150 and R4-dart-gate-x4
python -m training.datasets.train --variant R4-dart-gate-s150 --set identity.version=v2 --set identity.date=2026-10-30 \
  --set identity.seed=0 --set identity.objective=bcj --set training.lr=1.0e-3 --set training.updates=42000 \
  --set loss.action=huber --set loss.arm_weight=0.5 --set loss.wheel_weight=0.5
```

The test run of the chosen v2 models follows the pipeline document, Section 8, once the final model is fixed on validation.
Summary files of this run are tracked in `results/ablation/2026-10-30__training-ablation-v2/`; raw data and checkpoints are not.
