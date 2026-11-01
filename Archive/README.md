# Archive

Programs, models, data and documents that are no longer used. They were superseded by the training-method ablation, versions v1 (R1–R4) and v2 (R5–R8, DAgger), in `checkpoints/ablation/`.
The current pipeline is described in the [project pipeline](../docs/2026-10-30__guide__project-pipeline.md).

**Every archived file keeps its original relative path:** `Archive/<original path>`. To restore a file, move it back unchanged, using `git mv` for code.

- Code and documents are tracked by Git, with full history.
- Models, data and results are git-ignored (`Archive/final/robotnik_arm_task/{checkpoints,data,results}/`).
- Tests: `pytest.ini` sets `norecursedirs = Archive`, so archived tests are not collected.

Files that feed digests or hashes are never edited, not even comments. This covers the collection sources listed in `SOURCE_FILES` in `data_collection/collect.py` (including `settings.yaml`), `core/features.py` and `core/feature_sets.py` (dataset `builder_id`), and `experiments/splits/clean253.json`. Training code and configs (`training/`, `experiments/configs/`) may change: each run records their digest in `provenance.json`, and no check compares it with older runs.

## Effect on old records

- Old model paths in `results/results.csv`, `docs/reports/` and `outputs/2026-10-27/organization/` were **not rewritten**. To find a file, prefix its original path with `Archive/` (relative to the repository root). File contents and SHA-256 hashes are unchanged.
- `results/results.csv` is an append-only index and still holds rows for archived models.

---

## First archive: 2026-10-28

Done before the training-method ablation.

### Programs (tracked)

| Original path (relative to repository root) | Reason |
|---|---|
| `final/robotnik_arm_task/training/dataset.py`, `training/policies.py` | Early placeholders, replaced by `training/common/` |
| `final/robotnik_arm_task/training/train_act.py` | ACT placeholder; ACT was dropped |
| `final/robotnik_arm_task/training/train_mlp.py`; `tests/training/test_train_mlp_wrapper.py` (formerly in `test_baseline_cli.py`) | Deprecated compatibility entry point and its test |
| `final/robotnik_arm_task/experiments/capacity.py`, `experiments/capacity/`; `tests/training/test_capacity.py` | Finished D8-W256 capacity experiment |
| `final/robotnik_arm_task/experiments/reweight/` | Finished arm 0.8 / wheel 0.2 paired retraining |
| `final/robotnik_arm_task/training/dagger/`, `data_collection/dagger.py`, `dagger_preflight.py`, `dagger_rollout.py`, `dagger_scenarios.py`, `evaluation/dagger_audit.py`, `evaluation/dagger_report.py`; `tests/dagger/` | DAgger A round 1 did not improve reaching |
| `final/robotnik_arm_task/experiments/protocols/2026-10-27__protocol__dagger-a-r1.json` | Frozen DAgger R1 protocol |
| `final/robotnik_arm_task/experiments/demo_models/` (all files at the time) | Demo manifests pointing to archived models |

### Models (git-ignored)

| Original path `final/robotnik_arm_task/checkpoints/…` | Contents | Size |
|---|---|---|
| `clean/A/`, `clean/B/` | Old clean pipeline: BC screening and seeds for A and B, M6/M1 BC+J | 797 MB |
| `reweighted/` | 7 A models retrained with arm 0.8 / wheel 0.2 | 105 MB |
| `capacity/` | D8-W256 | 77 MB |
| `dagger/` | DAgger R1 | 34 MB |
| `sanity/`, `calibration/` | Single-trajectory overfit checks, old λ_J calibration records | <2 MB |

---

## Second archive: 2026-10-30

Goal: the main folders hold only the current version (ablation v1, rows R1–R4) and the collection, dataset and test tools it uses.

### Programs and tests (tracked)

| Original path | Reason |
|---|---|
| `final/robotnik_arm_task/experiments/run.py` | Stage runner of the clean pipeline (M1–M6); replaced by `experiments/ablation/run.py` |
| `final/robotnik_arm_task/experiments/protocols/p-*.yaml`, `README.md` | Frozen clean-pipeline protocols; they accept only archived models |
| `final/robotnik_arm_task/evaluation/residual_noise.py` | Residual-noise diagnostic for the clean models |
| `tests/training/test_experiments.py`, `tests/evaluation/test_residual_noise.py` | Tests of the above |
| `test_frozen_legacy_experiment_keeps_its_original_action_weights` from `tests/training/test_action_weights.py` | Moved to `Archive/tests/training/test_action_weights_legacy.py`. The rest of that file stays and now calls `training.baseline.train.objective_from` directly (equivalent to the old `experiments.run._action_objective`). |

The single-model CLIs in `training/baseline/train.py` and `training/jacobian/train.py` stayed in place in this pass, because the current entry point imported their `objective_from` (archived in the third pass). The CLIs themselves belong to the clean pipeline (they need `Clean_223` and `data/cache/`) and are no longer used.

### Documents (tracked)

| Original path `docs/…` | Reason |
|---|---|
| `plans/2026-10-26__implementation-plan__mlp-baseline-and-jacobian.md` | Completed implementation plan of the clean pipeline |
| `plans/2026-10-27__plan__arm08-wheel02-retraining-a.md`, `reports/2026-10-27__report__arm08-wheel02-retraining-a.md` | 0.8/0.2 retraining |
| `plans/2026-10-27__plan__d08w0256-single-run.md`, `reports/2026-10-27__report__d08w0256-single-run.md`, `specs/2026-10-27__design__network-capacity-a.md` | D8-W256 capacity experiment |
| `reports/2026-10-27__report__dagger-a-r1.md`, `reports/2026-10-27__review__dagger-a-plan.md`, `specs/2026-10-27__design__dagger-a.md` | DAgger R1 |
| `reports/2026-10-27__report__reaching-failure-and-training-rollouts.md`, `reports/2026-10-27__report__shared-loss-gradient-diagnostics.md` | Diagnoses of old models. They motivated R4 and can be cited in the report. |
| `indexes/2026-10-27__index__workspace-structure.md` | Outdated workspace index, replaced by `docs/README.md` and the pipeline document |
| Clean-pipeline sections of `2026-10-29__guide__reproduction-details.md` | Split out to `Archive/docs/2026-10-29__guide__clean-pipeline-legacy.md` |
| `COMP0245-Assignment-2026-2.pdf` | Byte-identical to `docs/references/2026-10-24__assignment__comp0245-final-project.pdf` (git-ignored) |

13 compatibility symlinks were removed: the old `docs/*.md` names and `docs/superpowers/`. Each was only a second name for one file. The table of old names is in `docs/README.md`. Relative links inside archived documents were updated to the new locations.

### Data and results (git-ignored)

| Original path `final/robotnik_arm_task/…` | Contents | Size |
|---|---|---|
| `data/cache/` | Feature caches of the clean pipeline (the current pipeline uses `data/datasets/`) | 2.8 GB |
| `data/Markovian/DAgger/` | DAgger R1 data | 580 MB |
| `data/clean_subsets/build_clean_subsets.py` | Old one-off copy with a hard-coded absolute path; the maintained version is in `data_collection/` | <1 MB |
| `results/clean/`, `results/capacity/`, `results/reweighted/`, `results/dagger/` | Evaluations and diagnostics of old training versions | 51 MB |
| `results/diagnostics/2026-10-27__*`, `results/diagnostics/expert_residual_noise_*` | Diagnostics of old models | 39 MB |
| `results/figures/2026-10-27/` and the 6 compatibility links pointing into it | Clean baseline figures | 2 MB |
| `results/live_compare/2026-10-27/` and the `comparison.json` link | Old demo record. Through that link, a demo run on 2026-10-30 overwrote `2026-10-27__summary__live-comparison.json`, so the file now holds that run's output. Later demos write a new regular file, `results/live_compare/comparison.json`. | <1 MB |

### Deliberately kept in the main folders

- `data/{Markovian,Stateful}/Clean_223/`: not used for training, but the frozen split `experiments/splits/clean253.json` records its file hashes, the `-m realdata` tests re-check them, and `build_clean_subsets` produces it.
- `data/Stateful/`, `template_teacher.py` and other teacher-B files: the paired subsets require both teachers to succeed, and the expert comparison is part of the report.
- `results/results.csv`: the append-only evaluation index.
- R1–R3 in `checkpoints/ablation/`: they are the controls of the current ablation, not old versions.

Verification after this archive: `pytest` passes 189 tests (205 before, minus 16 archived). The R4 training `--dry-run`, the imports of every stage's entry point, and loading the 32 validation and 64 test scenarios all work.

## Third archive: 2026-11-01

Goal: after merging the DAgger branch (`feature/dart-ablation`), keep one current document per topic and give the ablation rows one naming scheme (R1–R8).

### Documents (tracked)

| Original path `docs/…` | Reason |
|---|---|
| `plans/2026-10-24__roadmap__project-milestones.md`, `plans/2026-10-26__plan__project-overview.md` | 24 October plan exports (T1–T3 test sets, 23 inputs, one coder); the live plan is the Claude Doc |
| `plans/2026-10-26__plan__ablation-strategy.md` | Its own header marked it as the archived 24 October proposal |
| `plans/2026-10-26__plan__mlp-baseline-training.md` | Clean M1–M6 baseline plan; the split and success predicate now live in the pipeline document |
| `research/2026-10-26__research__mlp-training-recipe.md` | Recipe research for M1–M6; the adopted recipe is in pipeline Section 6 |
| `specs/2026-10-26__design__mlp-training-structure.md` | Pre-implementation design; the implemented layout is in the reproduction reference |
| Sections "Data (clean collection details)", "Live comparison demo" and the build commands of "Gate and DART datasets" in `2026-10-29__guide__reproduction-details.md` | Duplicates of other sections, the root README and pipeline Section 9 |

### Programs (tracked)

| Original path | Reason |
|---|---|
| `final/robotnik_arm_task/experiments/scripts/*.sh` | Shell scripts that ran v2 on a teammate's machine (home-directory paths, legacy row names). Kept as a record; `experiments.dagger_rounds` replaces them |
| `final/robotnik_arm_task/training/baseline/train.py`, `training/jacobian/train.py` | Single-model CLIs of the clean M1–M6 pipeline (they need `Clean_223` and `data/cache/`). Their `objective_from` moved to `training/baseline/loss.py` and `training/jacobian/loss.py`, where `training/datasets/train.py` now imports it |
| `final/robotnik_arm_task/evaluation/plots.py` | Figures from the clean pipeline's result index; nothing else used it |
| Functions cut from live modules, kept as `*_legacy.py` beside the original path: `prepare_run` (`training/common/engine_legacy.py`); `select_refs`, `cache_id`, `subset_cache` (`training/common/dataset_legacy.py`); frozen-protocol IDs and checks (`training/common/config_legacy.py`); `validate_final_protocol`, `evaluate_final` (`evaluation/evaluate_legacy.py`) | Only the clean launchers or the archived `experiments/run.py` called them. `build_cache`/`load_cache` stay, because the tests use them to check raw-episode validation |
| `tests/training/test_baseline_cli.py`, `test_jacobian_cli.py`, `tests/evaluation/test_plots.py`, `test_final_protocol.py`; two tests from `test_refit.py` (`Archive/tests/training/test_refit_legacy.py`) and one from `test_action_weights.py` (appended to `test_action_weights_legacy.py`) | Tests of the code above |

`experiments/configs/baseline.yaml` and `teachers/*.yaml` stay: `evaluation.evaluate --policy expert` resolves the teacher configuration through them. `jacobian.yaml` stays for the configuration tests, and `common.yaml` keeps its `cache_root` key: removing it would change the resolved configuration, and with it the configuration hash of every new run.

### Data and results (git-ignored)

| Original path `final/robotnik_arm_task/…` | Archived as | Reason |
|---|---|---|
| `data/cache/` (750 MB) | `data/cache-2026-11-01/` | Caches rebuilt on 2026-10-30 by the real-data test of `prepare_run`; nothing current reads them |
| `results/tables/2026-10-24`, `2026-10-25`, `2026-10-26` and `results/figures/2026-10-24/` | Same paths | Byte-identical dated copies of `results/controller_timing.csv`, `expert_t3_dev.csv`, `expert_noise_robustness.csv` and `figures/controller_timing.png`, which stay |

### Renamed in place (v2)

The DAgger and control variants, the DAgger runner's output paths and the v2 result summaries were renamed to the R5–R8 scheme.
The legacy-to-current table is in `final/robotnik_arm_task/results/ablation/2026-10-30__training-ablation-v2/README.md`.
`results/dagger/`, `results/far_probe/` and `results/robust_probe/` moved to `results/ablation/2026-10-30__training-ablation-v2/{dagger,probes/far,probes/robust}/`. The `results/dagger/` archived on 2026-10-28 is the unrelated 27 October DAgger round 1.

## Restore examples

```bash
# Restore the DAgger code (from the repository root)
git mv Archive/final/robotnik_arm_task/training/dagger final/robotnik_arm_task/training/dagger
# Restore old models
mv Archive/final/robotnik_arm_task/checkpoints/clean final/robotnik_arm_task/checkpoints/clean
```
