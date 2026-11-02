# Network-architecture ablation for R4, R6 and R8: design

Date: 2026-11-02. Status: requirements agreed in conversation; written design awaiting the user's approval. Nothing is built yet.

## Intent and scope

Measure what the network architecture alone does to closed-loop performance, for the three training recipes that work (R4-dart-gate, R6-gate-dagger, R8-dart-gate-dagger). The current architecture is MLP01 = 256×256×256 (141,321 parameters, about 83 µs per step on one CPU thread). The main question is the lightweight direction: how much smaller the network can get before performance drops, and what it buys in parameters, FLOPs and latency. Larger networks are included to show whether more capacity helps at all.

The result is a section of the report ("lightweight direction"). It does **not** select or replace the submitted final model.

### Goals

- Train 9 new architectures × 3 rows × 3 seeds = 81 models on fixed data with a frozen recipe. The architecture is the only variable.
- Evaluate every model, plus the existing 256×3 references, on validation, the hold test and the frozen 240-scene unified test suite.
- Report success against size (parameters, FLOPs, latency) per row, and name the smallest architecture that keeps 256×3-level performance.

### Non-goals

- No new data collection. DAgger is **not** rerun per architecture (the user chose option A: retrain on the existing aggregated round-3 data).
- No λ_J ratio sweep, no loss, feature or optimiser changes, and no residual, recurrent or other model families.
- No change to the submitted final model or to existing R1–R8 checkpoints and results.

### Anti-goals (a failure even if everything runs)

- Using test results to choose anything: architecture, learning rate, checkpoint or the final model. Test numbers are only reported.
- Letting a second variable creep in. This includes a λ_J recalibrated per architecture, or a learning rate chosen on test or chosen differently per row.
- Overwriting or renaming existing v1/v2 checkpoints, datasets or `results/test/2026-11-01_215340__R1-R8`.

## Decisions made by the user

| Decision | Choice |
|---|---|
| Rows | R4, R6, R8 |
| Data for R6/R8 | Option A: the existing round-3 aggregated datasets, shared by all architectures and seeds |
| Training recipe | Frozen: BC+J, Huber, arm/wheel 0.5/0.5, AdamW, lr 1e-3, cosine schedule with 500 warm-up updates, batch 1024, 42,000 updates, seeds 0–2, terminal checkpoint only |
| Recorded efficiency metrics | Parameters, FLOPs, single-thread NumPy latency, plus hold-test stiffness/damping eigenvalues |
| Test | Every architecture variant also runs the frozen 240-scene unified test suite |
| Grid | The revised 9-architecture grid below, including the learning-rate check |

## Architecture grid

All rows use `features-v1-gate`: 22 inputs, 7 torque outputs and 2 wheel outputs. The block is Linear → LayerNorm → SiLU per hidden layer (unchanged `training/common/models.py`). New names follow the `MLP<NN>` registry in order of first use and are added to `model.architectures` in `experiments/configs/datasets/base.yaml`.

| Name | Hidden layers | Parameters | FLOPs per step (2·MACs) | Line |
|---|---|---:|---:|---|
| MLP02 | 32×3 | 3,337 | 6,080 | width |
| MLP03 | 64×2 | 6,473 | 12,160 | small-width depth |
| MLP04 | 64×3 | 10,761 | 20,352 | width + small-width depth |
| MLP05 | 64×4 | 15,049 | 28,544 | small-width depth |
| MLP06 | 128×3 | 37,897 | 73,472 | width |
| MLP07 | 256-128-64 | 48,521 | 94,336 | tapered (Guegan & Teo 2026) |
| MLP08 | 256×2 | 75,017 | 146,944 | depth |
| MLP01 (existing) | 256×3 | 141,321 | 278,016 | reference, not retrained |
| MLP09 | 256×4 | 207,625 | 409,088 | depth |
| MLP10 | 512×3 | 544,777 | 1,080,320 | width |

The FLOPs count linear layers only. LayerNorm and SiLU add O(width) work, which is reported separately as negligible.

## Data and λ_J

| Row | Variant file | Training datasets (fitting scenarios only) | Fixed λ_J (from the 256×3 seed-0 calibration) |
|---|---|---|---|
| R4 | `R4-dart-gate` | `clean253` + `dart-d2s075x2` (`__features-v1-gate`) | 0.07439793449003185 |
| R6 | `R6-gate-dagger-r3` | `clean253` + `dagger-R6-v2-r1..r3` | 0.00028908631878552324 |
| R8 | `R8-dart-gate-dagger-r3` | `clean253` + `dart-d2s075x2` + `dagger-R8-v2-r1..r3` | 0.12621421349827852 |

**λ_J is fixed per row, not recalibrated per architecture.** The ratio rule (λ_J = 0.1·L_action/L_J at seed-0 initialisation) gave 0.75–1.05× the 256×3 value across architectures in a probe (session scratchpad `arch/calib_probe.py`). The spread came from initialisation noise in L_action and had no trend with size. A fixed value makes the objective identical across architectures, and it reproduces the 256×3 references exactly, because their calibrated value is that same number.

Implementation requirement: `training/datasets/train.py:prepare()` currently always recalibrates for `bcj`. Add an explicit opt-in, for example `loss.jacobian.calibration: fixed` with `loss.jacobian.lambda_j` given, that skips calibration. It must record in provenance the source run whose calibration it copies, and it must fail if the source config's λ_J differs from the value given. Read the value from the source checkpoint's `config.yaml`, not from a hand-typed number. The default behaviour and existing config hashes must not change.

## Learning-rate check (before the grid)

lr 1e-3 was screened for 256×3. Under standard parametrisation, the optimal learning rate drifts with width (Yang et al. 2021). Henderson et al. (2018) note that architecture changes may need retuning. The check uses validation only:

- R4, seed 0, MLP02 (32×3) and MLP10 (512×3), each at lr 3e-4 and 3e-3: 4 extra runs. Their lr 1e-3 runs are the grid's own R4 seed-0 runs.
- Metric: validation sustained hold of the full student over 32 scenes.
- Rule (fixed now): if lr 1e-3 is within 2/32 of the best of the three learning rates at **both** extremes, keep 1e-3 for every architecture and row. Otherwise, run the same three-lr screen on R4 seed 0 for all 9 new architectures, choose each architecture's lr by validation (ties go to 1e-3), and use that lr for all rows and seeds of that architecture.
- Report the check, whichever way it goes.
- The run name does not contain the learning rate, so the lr-check runs go under their own experiment, `architecture-lr-check` v1. Otherwise the 3e-4 and 3e-3 runs would collide with the grid's directories.

## Evaluation per model

1. **Validation.** 32 frozen validation scenes in mode `student` (primary) and mode `arm`, with the same predicates as ablation v1, written through the existing `experiments/ablation/run.py` stages.
2. **Hold test.** `evaluation/hold_test.py` on the 8 v1 hold-test scenarios. Record arm/full stayed, median final error, max real eigenvalue, stiffness eigenvalue max and damping diagonal.
3. **Unified test suite.** `python -m evaluation.test_suite --models <step_42000.npz …>` on the frozen `unified-test-v1` manifest (T0–T5, 240 scenes, continuous <10 mm for 3 s, 20 s for T5). The run must report the same `scene_manifest_sha256` (`0a916445…`) and protocol as the reference run `results/test/2026-11-01_215340__R1-R8`. Known gap: `suite_manifest._model_record` labels a model by its row only (`R4`), and `suite_report` pools per row. Different architectures of one row would therefore be merged into one row-level aggregate. The architecture must become part of the group label for these models (for example `R4/MLP02-32x3`), or the ablation's own analysis must aggregate from `rollouts.csv` by (row, architecture) and ignore the suite's row-level tables. Either way, outputs for the existing MLP01 rows must not change. Do not rerun expert A if the suite allows skipping it; its results already exist in the reference run.
4. **Isolated latency.** A dedicated micro-benchmark, not the in-rollout latency, which runs 20 workers in parallel. Use the NumPy policy forward pass the rollouts use, including normalisation, with batch 1, `OMP/OPENBLAS/MKL_NUM_THREADS=1`, 2,000 warm-up calls and 20,000 timed calls, repeated 3 times. Report median and p99 µs per architecture, on one seed's model, with nothing else running. Also record the `.npz` size in bytes and the parameter count.

**References (256×3):** reuse the existing terminal checkpoints (R4 v1 s0–2; R6/R8 `-r3` v2 2026-11-01 s0–2) and their test results from `2026-11-01_215340__R1-R8`; they are not retrained or retested. Run validation and the hold test for them inside the new experiment, so that every row of the summary comes from the same evaluator version. The R6/R8 references have no hold-test eigenvalues yet. If a rerun validation differs from the recorded value, report both and investigate before analysis.

## Analysis and reporting

- One table per row with one line per architecture: parameters, FLOPs, latency, validation hold (per seed and pooled with Wilson 95%), test success per suite (T0…T5) and over all 240 scenes (pooled over 3 seeds, Wilson 95%), the test's A–F precision tiers and completion time, and hold-test stiffness/damping.
- Figures: success vs parameters (log x), one line per row with seed spread; the same against latency; separate panels for the width line (×3) and the depth lines (256 and 64); and offline validation arm RMSE vs closed-loop success, contrasting with Tuyls et al. (2024), where offline loss predicts return.
- Lightweight recommendation, descriptive (my call): per row, the smallest architecture whose all-suite test success pooled over seeds is within 3 percentage points of 256×3 on that row. Name the speed-up and the parameter reduction. Report the T0 ceiling (R4/R6/R8 are already at 97–99%) and point out that T1–T3/T5 carry most of the signal.
- Limitations to state in the report: the DAgger data reflect the 256×3 student's state distribution; only 3 seeds; a single learning rate (or the outcome of the check); terminal checkpoint only.

## Output layout and naming

- Experiment `architecture-ablation`, version `v1`, regime `ablation`, date fixed in the manifest at launch.
- Manifest(s) under `experiments/ablation/<date>__architecture-ablation-v1*.yaml`. The current runner takes one `architecture` per manifest. Either loop over architectures in the runner, or write one manifest per architecture with a shared experiment directory. This is the builder's choice, but run names and directories must follow the existing pattern.
- Checkpoints: `checkpoints/ablation/<date>__architecture-ablation-v1/<row-variant>__MLP<NN>-<shape>__s<seed>__v1__<date>/`.
- Results: `results/ablation/<date>__architecture-ablation-v1/` (validation, hold_test, lr-check, latency, runs.csv, summary.csv/.md, figures) and `results/test/<datetime>__arch-ablation/` from the test suite.
- Everything resumable and skip-if-complete, as the existing runner and test suite are.

## Budget and order

Measured: one 256×3 R8-r3 training takes about 3.7 min on the RTX 5070 Ti, and R4 is faster. One model on the unified suite takes about 2.9 CPU-hours, roughly 9 min of wall time at 20 workers. Plan:

1. Code changes and tests (fixed λ_J, registry, manifest/runner support, latency benchmark).
2. LR check: 4 runs plus validation, under 30 min. Apply the rule.
3. Grid training: 81 runs, about 4.5 h on the GPU. Validation, the hold test and the test suite run on CPU as rows finish.
4. Test suite: 81 models, about 12 h of wall time. Latency benchmark at the end, on an idle machine.
5. Summary, figures and the report paragraph.

The target is to finish by the end of 2026-11-03, leaving 2026-11-04 to 2026-11-05 for the report (deadline 2026-11-06).

## Left to the builder

The manifest structure (one or many), how training runs share the GPU, test-suite batching, figure styling, and the exact form of the fixed-λ_J config key, all within the constraints above.

## References

- Andrychowicz et al., What Matters in On-Policy RL, ICLR 2021, arXiv 2006.05990
- Guegan & Teo, Behavior Cloning of MPC for 3-DOF Robotic Manipulators, arXiv 2606.00383
- Mandlekar et al., What Matters in Learning from Offline Human Demonstrations, CoRL 2021
- Rusu et al., Policy Distillation, ICLR 2016, arXiv 1511.06295
- Yang et al., Tensor Programs V, NeurIPS 2021, arXiv 2203.03466
- Henderson et al., Deep RL that Matters, AAAI 2018, arXiv 1709.06560
- Chen et al., Lipschitz-Constrained Policies, arXiv 2410.11825
- Tuyls et al., Scaling Laws for Imitation Learning in Single-Agent Games, TMLR 2024, arXiv 2307.09423
