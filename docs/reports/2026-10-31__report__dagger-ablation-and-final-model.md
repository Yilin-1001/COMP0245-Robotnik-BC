# DAgger on R1–R4, DART controls, generalisation, robustness and network size

Date: 31 October 2026. Branch `feature/dart-ablation`. All runs on one machine (i9-12950HX, RTX 3070 Ti Laptop, WSL2).
Teacher A (`markovian-lt0.50-v1`) throughout. Standard predicates unless stated: reach < 10 mm after 8 s, 3 s more,
all of the final 1 s < 10 mm, 40 s cap. Validation = 32 frozen scenes × 3 seeds (96). Test = 64 sealed scenes × 3 seeds (192).
Default network MLP01 (256×256×256), recipe of the training-method ablation v1 (BC+J, lr 1e-3, 42,000 updates, Huber).

**Proposed final model: R2 + DAgger, round 3** (`R2D-dagger-R2v1-r3__MLP01-256x3__s{0,1,2}__v1__2026-10-30`).

## 1. Reproduction on a second machine

| | This machine | Original (R4 report) |
|---|---|---|
| R4 validation | 29 / 28 / 26 = 83/96 | 29 / 27 / 27 = 83/96 |
| R4 test | 61 / 61 / 58 = 180/192 | 60 / 60 / 59 = 179/192 |

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
validated; it drives the next round. `experiments/dagger_rounds.py` runs K rounds unattended for any row R1–R4.

## 3. Main results

| Row | clean | DART | gate | DAgger | Validation | Test | Far 9–12 m | Robustness (perturbed) |
|---|---|---|---|---|---|---|---|---|
| Teacher A | | | | | 96/96 | 64/64 | 100% | 225/288 |
| R1 | ✓ | | | | 0/96 | — | — | — |
| R2 | ✓ | | ✓ | | 10/96 | — | — | — |
| R3 | ✓ | ✓ | | | 2/96 | — | — | — |
| R4 | ✓ | ✓ | ✓ | | 83/96 | 180/192 | 21/24 (s0,s2) | 62/96 (s0) |
| R1 + DAgger (r3) | ✓ | | | 3 | 89/96 | 183/192 | 36/36 | 55/96 (s0) |
| **R2 + DAgger (r3)** | ✓ | | ✓ | 3 | 89/96 | 185/192 | **36/36** | **195/288** |
| R3 + DAgger (r3) | ✓ | ✓ | | 3 | 88/96 | 178/192 | 11/12 (s0) | — |
| R4 + DAgger (r2) | ✓ | ✓ | ✓ | 2 | 93/96 | 187/192 | **0/24** (s0,s2) | 189/288 |

DAgger rounds, seed 0, validation /32 (round chosen by seed-0 validation, then seeds 1–2 trained):

| Row | start | r1 | r2 | r3 | chosen round, s0/s1/s2 |
|---|---|---|---|---|---|
| R1 | 0 | 27 | 29 | 30 | r3: 30/29/30 |
| R2 | 1 | 24 | 27 | 31 | r3: 31/30/28 |
| R3 | 0 | 23 | 26 | 31 | r3: 31/27/30 |
| R4 | 29 | 29 | 31 | 30 | r2: 31/31/31 |

Test failures: R4 failed 7 scenes (0001, 0022, 0029, 0031, 0032, 0037, 0060, mostly low targets); R4 + DAgger failed
3 (0024, 0031, 0048, all < 0.8 m away).

## 4. DART controls (all with gate, no DAgger)

| Data | Arm σ (N·m) | Validation |
|---|---|---|
| clean only (R2) | — | 10/96 |
| + 442 DART, σ halved (`d2-s0375-x2`) | 0.375 | 69/96 |
| + 442 DART (`d2-s075-x2`, R4) | 0.75 | 83/96 |
| + 442 DART, σ doubled (`d2-s150-x2`) | 1.5 | 84/96 |
| + 884 DART (`d2-s075-x4`, equal-data control) | 0.75 | 82/96 |
| + 855 DAgger (R2 + DAgger) | — | 89/96 |

Doubling the DART data gives nothing; a similar number of DAgger episodes does. The DAgger gain is not explained by data volume.

## 5. Generalisation beyond the training range (`evaluation/far_probe.py`)

Start-to-target distance bands, 12 fresh scenes each (seeds 980000+), episode cap raised to 80 s (teacher A needs
≈ 40 s for 10 m), teacher A as reference.

| Model | Seeds | 0.5–3 m | 3–6 m | 6–9 m | 9–12 m |
|---|---|---|---|---|---|
| Teacher A | | 12/12 | 12/12 | 12/12 | 12/12 |
| R4 | s0, s2 | 24/24 | 24/24 | 21/24 | 21/24 |
| R4 + DAgger r1 | s0 | 12/12 | 12/12 | 12/12 | 9/12 |
| R4 + DAgger r2 | s0, s2 | 24/24 | 22/24 | 6/24 | 0/24 |
| R4 + DAgger r3 | s0 | 12/12 | 12/12 | 7/12 | 0/12 |
| R1 + DAgger | s0–s2 | 33/36 | 36/36 | 36/36 | 36/36 |
| R2 + DAgger | s0–s2 | 36/36 | 36/36 | 35/36 | 36/36 |
| R3 + DAgger | s0 | 10/12 | 12/12 | 12/12 | 11/12 |

Only DART + gate + DAgger loses far targets, and more so with each round; the cause is not yet identified.

## 6. Robustness (`evaluation/robust_probe.py`)

In-range scenes, 12 per condition (seeds 990000+). At-target pushes 1 s after first entering 10 mm, with a 10 s
post-reach window; driving push at t = 3 s with a 60 s cap.

| Condition | Teacher A (×3) | R2 + DAgger (3 seeds) | R4 + DAgger (3 seeds) | R4 (s0) | R1 + DAgger (s0) |
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

Overall robustness is tied between R2 + DAgger and R4 + DAgger; the consistent difference is the push while driving
(base navigation). Mass changes and base pushes are weak for every student and for teacher A itself (no integrator).

## 7. Network size (validation, 3 seeds)

| Hidden layers | Parameters | R4 + DAgger data | R2 + DAgger data | Inference p50 |
|---|---|---|---|---|
| 64×64 | ≈ 6.5 k | 82/96 | 73/96 | 0.27 ms |
| 128×128 | ≈ 21 k | 88/96 | 75/96 | 0.29 ms |
| 128×128×128 | ≈ 38 k | 87/96 | 85/96 | 0.33 ms |
| 256×256×256 | ≈ 141 k | 93/96 | 89/96 | ≈ 0.4 ms |
| 2 × 256×256×256, gate-switched (R5, R4 data) | ≈ 282 k | 85/96 | — | — |

All are well inside the 1 ms control period. R2 + DAgger needs three layers.

## 8. Conclusions

1. DAgger (β = 0, from any starting row) is the main gain: 0–86% → 92–97% on test.
2. Without DAgger, DART and gate are both needed; with DAgger, neither is necessary.
3. The DAgger gain is not data volume (equal-data DART control: no gain).
4. DART σ = 0.75 N·m is adequate; half is clearly worse, double gives nothing.
5. R2 + DAgger ties R4 + DAgger in range (185 vs 187/192) and on overall robustness, and is clearly better on far
   targets (36/36 vs 0/24) and on pushes while driving; it also needs no DART. Proposed as the final model.

**Caveats.** The DAgger round was chosen on seed-0 validation. The sealed test set had already been viewed before
DAgger (`2026-10-29__report__r4-test-diagnosis.md`), and the final-model choice uses the far and robustness probes, which
were run after the test results. Some probe rows use one or two seeds (marked). Validation differences of a few scenes
are within noise.

## 9. Reproduce (from `final/robotnik_arm_task/`)

Prerequisites: `clean253__features-v1{,-gate}` and `dart-d2s075x2__features-v1{,-gate}` built as in the root README.
Runs are not bit-identical across machines (Section 1); compare pooled numbers. Commands used are also in `experiments/scripts/`.

```bash
# DAgger, 3 rounds, for each row (trains the row's own seeds 0-2 first, then rounds 1-3, then seeds 1-2 of the best round)
for ROW in R1-baseline R2-gate R3-dart; do
  python -m experiments.dagger_rounds --row $ROW --rounds 3 --tag $(echo $ROW | cut -d- -f1)v1 --date 2026-10-30 \
    --baseline-seeds 1 2 --final-seeds 1 2 --dev-scenes 64 --workers 20
done
python -m experiments.dagger_rounds --row R4-dart-gate --start <R4 seed-0 .npz> --rounds 3 --tag v1 \
  --date 2026-10-30 --final-seeds 1 2 --dev-scenes 64 --workers 20
# results: results/dagger/<tag>/summary.csv

# test (sealed set; models already fixed)
python -m evaluation.test_diagnosis --modes student --models <model.npz> --out results/test/<name>
# far targets and robustness
python -m evaluation.far_probe --models <a.npz> <b.npz> ... --out results/far_probe/<name>
python -m evaluation.robust_probe --models <a.npz> <b.npz> ... --out results/robust_probe/<name>
# DART controls: experiments/configs/dart/d2-s075-x4.yaml, d2-s0375-x2.yaml, d2-s150-x2.yaml (see experiments/scripts/overnight3.sh)
```

Summary files of this run are committed under `results/` (`summary.csv` / `summary.json`); raw data and checkpoints are not.
