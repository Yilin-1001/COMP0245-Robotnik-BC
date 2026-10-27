# MLP Training Code Structure Design

Date: 26 October 2026. Status: unified implementation target; documentation only. The directories, commands and interfaces below describe future implementation, not working training code.

Companion documents: [training research and recipe](../research/2026-10-26__research__mlp-training-recipe.md), [baseline training plan](../plans/2026-10-26__plan__mlp-baseline-training.md), and [dataset contract](2026-10-26__spec__expert-datasets.md).

Implementation tasks for the current clean BC/BC+J stage are in the [implementation plan](../plans/2026-10-26__implementation-plan__mlp-baseline-and-jacobian.md). Later DART/DAgger collection stays outside that plan.

Authority by subject: this document defines code boundaries, configuration resolution, run identity and artifacts; the research recipe defines the current numerical training settings and BC+J protocol; the baseline plan defines the exact clean scenario split and evaluation predicate; dataset manifests define the recorded teacher, action order and execution limits. The older total plan and ablation matrix are historical where these documents replace them. An inconsistency between a manifest, checkpoint and requested runtime interface must fail validation rather than be resolved by a silent override.

## 1. Intent and scope

Keep the clean-data MLP baseline, Jacobian-supervised MLP, later DART collection/training, and later DAgger iteration in clearly separate code locations. Their configurations, checkpoints and metrics must identify both the training objective and the data-collection regime. Reuse low-level components so comparisons do not accidentally change the network, normalisation or evaluator.

The current task is structural design. It does not implement training, move existing scripts, install PyTorch or launch experiments. Offline perturbation relabelling remains excluded. The existing clean recordings and their directories remain authoritative.

## 2. Approaches considered

| Approach | Tradeoff | Decision |
| --- | --- | --- |
| One large training script with switches for BC, Jacobian, DART and DAgger | Few files, but collection, objectives and iteration become mixed together | Does not satisfy the requested separation |
| Four completely independent implementations | Clear folders, but duplicates networks, normalisation and checkpoint handling; fixes and comparisons can diverge | Avoid |
| Separate method entry points and losses, shared low-level training components | Clear ownership with one implementation of common behaviour | Implementation target |

The user requested separation; the directory names and shared-module boundaries below define the unified implementation target.

## 3. Target source tree

All paths below are relative to `final/robotnik_arm_task/`. Python package directories include `__init__.py` when implemented; they are omitted from this diagram.

```text
core/                           # existing deployment/environment contracts
  interfaces.py                 # Observation, Action, Policy
  features.py                   # shared A/B features and deployable FK
  env.py                        # simulator adapter

data_collection/                # existing expert and clean-data collection
  expert.py                     # Markovian teacher A
  template_teacher.py           # Stateful teacher B
  scenarios.py                  # scenario construction
  collect.py                    # existing clean collection
  qa_clean.py                   # clean-data quality checks
  ...                           # existing collection/verification utilities

training/
  common/
    config.py                   # shared config resolution, validation and run identity
    dataset.py                  # manifests, feature cache, episode sampling
    normalisation.py            # fit/transform/inverse; fitting-only statistics
    models.py                   # MLP M1–M6; 7 torque + 2 mirrored wheel outputs
    policies.py                 # Policy wrapper and deployment inference
    engine.py                   # optimiser, schedule, update loop, validation hooks
    checkpoint.py               # save/load, metadata, optional NumPy export
  baseline/
    loss.py                     # action loss only
    train.py                    # clean-data plain-BC entry point
  jacobian/
    loss.py                     # action loss plus damping-Jacobian loss
    train.py                    # clean-data BC+J entry point

dart/                           # later stage: disturbed expert execution
  noise.py                      # perturbation process and its recorded parameters
  collect.py                    # apply noise; save clean labels and executed actions
  train.py                      # train on the declared clean/DART data recipe

dagger/                         # later stage: learner-driven aggregation
  collect.py                    # learner/expert mixture; expert labels visited states
  aggregate.py                  # round manifests, provenance and sampling mixture
  run.py                        # collect -> aggregate -> retrain -> evaluate per round

evaluation/
  offline.py                    # physical-unit errors, hold bias, damping diagnostics
  evaluate.py                   # shared closed-loop evaluator for all variants
  test_sets.py                  # frozen validation/test scenario definitions
  plots.py                      # reporting
  ...                           # existing expert/controller diagnostics

experiments/
  configs/
    common.yaml                 # shared training defaults
    baseline.yaml               # clean BC workflow, pinned objective=bc
    jacobian.yaml               # clean BC+J workflow, pinned objective=bcj
    dart.yaml                   # DART collection and retraining settings
    dagger.yaml                 # iteration, beta and aggregation settings
    objectives/
      bc.yaml                   # action-only loss definition
      bcj.yaml                  # action loss plus Jacobian settings
    teachers/
      A.yaml                    # teacher A, 21 inputs, source data/version
      B.yaml                    # teacher B, 26 inputs, source data/version
  splits/                       # versioned scenario split manifests
  protocols/                    # frozen decisions and evidence hashes, not config defaults
  run.py                        # launch existing entry points for a declared matrix
```

`training/baseline/` and `training/jacobian/` represent different objectives. `dart/` and `dagger/` represent different collection/training workflows. They do not contain copies of the MLP architecture or normaliser.

## 4. Responsibility and dependency boundaries

| Component | Owns | Must not own |
| --- | --- | --- |
| `training/common/config.py` | Resolve templates, validate identity/interface, create run ID | Collection or training side effects |
| `training/common/models.py` | Configurable PyTorch MLP and wheel mirroring | Losses, expert queries, collection |
| `training/common/policies.py` | Features, saved normalisation, inference and execution limits | Training-time derivative penalties or online expert assistance |
| `training/common/engine.py` | Execute updates using a supplied loss object and callbacks | Select BC versus BC+J itself; run DART or DAgger collection |
| `training/baseline/loss.py` | `0.5 * arm_loss + 0.5 * wheel_loss` | Jacobian-loss imports or second-order training gradients |
| `training/jacobian/loss.py` | Reuse baseline action loss; add valid-entry damping-Jacobian loss | Noise injection, new states or DAgger round management |
| `dart/` | Noise selection/configuration, disturbed collection, data recipe and retraining launch | Separate network, normaliser or evaluation definitions |
| `dagger/` | Current learner, beta schedule, teacher state, aggregation, retraining and round selection | Changes to the selected MLP architecture or loss implementation |
| `evaluation/` | Identical metrics and success predicates across methods | Selecting actions with expert assistance during student evaluation |

Dependency direction: baseline and Jacobian modules use `training/common`; Jacobian loss may reuse baseline action loss. DART and DAgger workflows may choose either loss implementation and call the common engine. Common code and the clean-training entry points never import the DART or DAgger workflows. Imports perform no collection, training or simulator construction.

Baseline and BC+J share the exact same `MLP` and `MLPPolicy` definitions. Their distinction is the objective, entry point, configuration and saved experiment identity. Jacobian-specific imports should be local to the Jacobian branch so baseline training does not execute the derivative-loss path. Offline damping diagnostics are available for either model and do not contribute gradients to baseline training.

NumPy deployment loads an explicit `.npz` inference export; it does not deserialize a PyTorch `.pt` checkpoint without torch. The torch backend loads `.pt`. Save the source checkpoint step/hash in every NumPy export, and record both artifact paths in evaluations. A selected `inference.npz` must correspond to the selected `best.pt` or terminal `final.pt`; candidate step exports have distinct step-specific names. Reject backend/file-format mismatches rather than silently loading an unrelated sibling artifact.

## 5. Entry points and method identity

These are proposed commands, not currently available commands. Run them from `final/robotnik_arm_task/` after implementation:

```bash
# Clean demonstrations: ordinary MLP baseline
python -m training.baseline.train --config experiments/configs/baseline.yaml --teacher A

# Same clean demonstrations: Jacobian-supervised MLP
python -m training.jacobian.train --config experiments/configs/jacobian.yaml --teacher A

# Later DART stage: collect first, then train on the resulting manifest
python -m dart.collect --config experiments/configs/dart.yaml --teacher A
python -m dart.train --config experiments/configs/dart.yaml --teacher A --objective bc --manifest <dart_manifest>

# Later DAgger stage: use a declared initial checkpoint and aggregate by round
python -m dagger.run --config experiments/configs/dagger.yaml --teacher A --objective bc --initial-checkpoint <checkpoint>
```

Track two independent identifiers in every run:

- `regime`: `clean`, `dart`, or `dagger` (the workflow that produced the run).
- `objective`: `bc` or `bcj` (the training loss).

Also record `teacher` (`A`/`B`), architecture ID, seed, dataset/split/feature hashes and source revision. A DAgger run's manifest records whether its initial checkpoint/data came from clean BC or DART; its regime alone does not describe all data sources.

`fit_stage` is `selection` (independent validation available) or `refit` (final all-training-data fit). The stage is separate from `regime` and `objective` and is recorded in every config and result row.

The baseline launcher accepts clean fitting data and `objective=bc` only; the Jacobian launcher accepts the same clean fitting data and `objective=bcj` only. They reject incompatible configuration or manifest provenance. Later DART/DAgger launchers select the corresponding loss object directly and call the common engine; they do not bypass these clean-launcher guards or invoke clean-training CLIs as subprocesses. Their initial no-J controls use `objective=bc`; `bcj` combinations are separate experiments.

### Configuration resolution

Versioned source templates live under `experiments/configs/`. `training/common/config.py` resolves them in this exact order, with later permitted values replacing earlier ones:

```text
common.yaml
  -> selected workflow (baseline.yaml / jacobian.yaml / dart.yaml / dagger.yaml)
  -> selected objectives/bc.yaml or objectives/bcj.yaml
  -> selected teachers/A.yaml or teachers/B.yaml
  -> explicit permitted CLI overrides
```

Merge mappings recursively; replace scalar and list values rather than concatenating lists. Reject unknown keys, missing required values and invalid combinations. Workflow files own workflow settings, objective files own loss settings, and teacher files own teacher/data/feature settings. Defining the same method-specific field in an unrelated layer is an error, not an extra source of defaults. Shared tunable training fields such as LR and update budget can be overridden explicitly and their effective values are recorded.

Versioned `experiments/protocols/` records frozen pilot, loss/budget, finalist and final-evaluation decisions with evidence hashes. It selects explicit template/CLI values; it is not another implicit override layer. Initial numerical implementation defaults are input scale floor 1e-6, arm/wheel scale floors 0.1 N·m/0.05 rad/s, torque-boundary tolerance `epsilon_tau=1e-4 N·m`, and LayerNorm epsilon 1e-5. Record and freeze inspected floors/tolerance before paired comparisons.

Ownership of frozen protocols: only `experiments/run.py` writes `experiments/protocols/<protocol_id>.yaml`, and never edits an existing file. A changed decision is a new protocol file with a new ID. Launchers receive frozen values as explicit permitted overrides (for example `training.updates`, `training.lr`, `loss.action`, `loss.jacobian.lambda_j`), so the resolved run config, not the protocol, is what a checkpoint records. The expected execution interface (`action_order`, ±30 N·m, ±6 rad/s, 1 kHz) is declared once in `common.yaml` and must equal the source manifests and episode metadata; it is not a CLI override.

The clean launchers pin `regime=clean` and their respective objective; teacher input dimensions are fixed by the selected feature definition. CLI/config overrides cannot defeat these invariants. DART and DAgger launchers permit `--objective bc` or `--objective bcj` and validate the matching objective template. An objective change creates a new experiment rather than an exact resume.

Only the fully resolved snapshot belongs in `checkpoints/<regime>/<teacher>/<objective>/<run_id>/config.yaml`; never write resolved per-run configs back into the source template directory. Save source template paths/hashes and a snapshot of relevant simulation/interface settings alongside it.

Keep robot/simulation/clean-collection settings in the existing `settings.yaml`; it is not a fallback source for MLP hyperparameters. Its current `mlp`, `train_stride`, `dagger` and DART-related collection defaults are legacy placeholders until migrated to their owning templates; the new launchers must not consume them implicitly. The current training recipe supersedes stride 10 and the initial 3×256 default. This documentation update does not alter the simulator or existing collection configuration.

### Execution interface and derivative-mask contract

Read action order and execution limits from the source dataset manifest and freeze them in the run config/checkpoint. Current A and B Clean_253 manifests record arm limits **±30 N·m** and wheel limits **±6 rad/s** in `collection_configuration.exec_torque_limit` and `collection_configuration.exec_wheel_limit`. Use those limits in policy wrapping and closed-loop evaluation; do not hard-code ±3 rad/s. Reject incompatible interfaces when aggregating datasets or loading a checkpoint. A deliberate interface change needs a separately identified experiment.

The recorded `clip_flags` describes execution-interface clipping (`a_executed != a_expert`). It does not describe the arm controller's internal torque saturation and must never be used as the Jacobian saturation mask. If teacher pre-clip torques or explicit internal saturation metadata are available, use them with a documented boundary tolerance. Otherwise supervise only clearly interior clean torque labels (`abs(tau_expert) < 30 - epsilon_tau`, epsilon_tau > 0 in N·m), omit near-limit rows, and record epsilon_tau and valid-entry coverage. Reuse that mask for the Jacobian loss and teacher-relative damping diagnostics; distinguish omitted rows from genuinely saturated zero-derivative targets. Never infer the mask from the noisy executed action.

## 6. Data, checkpoints and results

Keep the existing `data/Markovian/Clean_253` and `data/Stateful/Clean_253` recordings in place. Store the versioned internal split under `experiments/splits/`. Feature caches are disposable, keyed by source and feature hashes, and stored under `data/cache/<teacher>/<cache_id>/`; the plain-BC and BC+J pair reuses the same cache, split and fitting-only normalisers.

Use `A`/`B` for the `teacher` identity in configs, checkpoints and results. The raw-data `<Teacher>` directory is `Markovian` for A and `Stateful` for B; these are aliases of the same recorded teacher identity, not extra datasets.

New collection uses distinct locations:

```text
data/<Teacher>/DART/<collection_id>/
data/<Teacher>/DAgger/<collection_id>/round_001/
data/<Teacher>/DAgger/<collection_id>/round_002/
```

Each manifest identifies clean versus executed actions, source episodes, teacher version and collection parameters. DAgger additionally identifies the learner checkpoint, beta and round. Never overwrite clean demonstrations or use validation/test episodes as training data. B's teacher memory must evolve along the actual collection trajectory.

Separate training outputs by both identifiers:

```text
checkpoints/<regime>/<teacher>/<objective>/<run_id>/
  config.yaml                   # fully resolved immutable run configuration
  provenance.json               # dataset, split, features, source revision, parent checkpoint
  normaliser.npz                # actual saved input/output scales
  latest.pt                     # model plus optimiser/scheduler/RNG state for resumption
  best.pt                       # selection runs only: chosen using validation
  final.pt                      # refit runs only: frozen terminal update, no selection
  inference.npz                 # optional NumPy inference export
  metrics.csv                   # per-run training and validation metrics

results/<regime>/<teacher>/<objective>/<run_id>/
  offline.csv
  rollouts.csv                  # per-scenario outcomes
  figures/
```

### Unique run identity

Use the same globally unique `run_id` in the directory, checkpoint metadata, per-run metrics and shared result index:

```text
<regime>-<teacher>-<objective>-<cohort>-<architecture>-lr<lr>-s<seed>-<fit_stage>[-r<round>]-<config_id>
```

`cohort` identifies the declared data recipe, such as `clean253`, `clean223` or `aggregate`. Write learning rates with an explicit exponent sign, e.g. `3e-4`. `config_id` is the first 12 hex characters of a canonical effective-config hash covering data/split/feature hashes, model, loss and λ_J, normalisation policy, update budget, fit stage, source digest and parent/round identity; exclude the generated run ID, output directory and transient timestamps. Save the full hash and reject a short-hash collision. Within an exact configuration, an existing output directory requires explicit resume with a compatible `latest.pt`; it is never overwritten. A separate rerun requires a new recorded seed or replicate identifier included in the identity hash.

Finalise identity only after manifest/split validation, fitting-only normaliser construction and any fitting-only λ_J calibration. Save normaliser and calibration hashes in that identity. Device selection, worker count and destination root are operational fields: record them but exclude them from the scientific identity, so CPU fallback or moving artifacts does not create a new experiment. Exact resume restores the saved configuration and RNG states and records operational changes; it does not silently refit scales or recalibrate λ_J.

Verify bitwise resume equivalence only on the same device/backend and pinned software stack. Moving between CPU/GPU or library versions is recorded as a resumed continuation, not a promise of bitwise-identical training; PyTorch explicitly does not guarantee reproducibility across platforms ([official reproducibility notes](https://docs.pytorch.org/docs/2.7/notes/randomness.html)). Preserve scientific identity and record each continuation's environment/device separately.

Examples below use illustrative 12-character hashes:

```text
checkpoints/clean/A/bc/clean-A-bc-clean253-m2-lr3e-4-s0-selection-a1b2c3d4e5f6/
checkpoints/clean/A/bcj/clean-A-bcj-clean253-m2-lr3e-4-s0-selection-b2c3d4e5f6a1/
checkpoints/clean/A/bc/clean-A-bc-clean253-m2-lr3e-4-s0-refit-c3d4e5f6a1b2/
```

`results/results.csv` is the versioned compact index. Include `run_id`, `regime`, `teacher`, `objective`, `cohort`, `fit_stage`, seed, config/split hashes, checkpoint step/path, evaluation split and metric fields. The evaluation row key is `(run_id, checkpoint_step, evaluation_split, evaluation_id)`, since one run can have multiple checkpoint evaluations. Models with different datasets, objectives or fit stages never share an output directory. Checkpoints validate feature order, input/output dimensions, execution limits and normaliser compatibility on load.

### Selection versus final refit

Selection runs use the frozen 221/32 clean split (196/32 for the reduced fitting comparison), and save `best.pt` by the declared closed-loop validation ranking, plus `latest.pt` for resumption. A BC/BC+J pair uses identical fitting-only scales and all other paired settings.

Final refit is optional, as in the baseline plan. Before refitting, freeze the architecture, objective, LR schedule, update-count rule, normalisation scale floors, Jacobian weight and final seed set using validation results. Create a new `fit_stage=refit` run, retain the selected run as provenance, initialise afresh from the declared seed and fit on all 253 primary training episodes (or all 223 for a declared reduced-data finalist). Refit normalisers on that cohort and recompute standardised Jacobian targets using the new scales. Keep the frozen λ_J; no new validation-driven tuning or checkpoint selection is allowed. Save the fixed terminal checkpoint as `final.pt`; `best.pt` is not produced in refit mode, and prior validation episodes are not still treated as held out. The untouched 64 test scenarios remain evaluation-only. If refit is skipped, freeze and test the previously selected `best.pt` instead. DART/DAgger final-refit data recipes require a separately declared manifest before using this mode.

### Repository policy for results

Commit only the compact shared `results/results.csv` summary. Keep per-run detailed results, figures, logs, raw datasets and checkpoints ignored. `.gitignore` must contain a narrow exception for `final/robotnik_arm_task/results/results.csv`; this does not automatically stage any existing result file. Figures are reproducible local artifacts, not committed outputs.

## 7. Migration from the existing skeleton

This is an implementation sequence, not a migration performed by this design task:

1. Keep the implemented `core/` and clean collection code in place. Move/implement the placeholder dataset and normaliser responsibilities under `training/common/`; implement common models, policy wrapping, persistence and engine.
2. Implement `training/baseline/train.py` and its action-only loss first, including one verified closed-loop M2-A baseline. Existing `training/train_mlp.py` can remain a thin deprecated wrapper for this baseline entry point only; document its configuration changes rather than silently reinterpreting old arguments.
3. Implement `training/jacobian/train.py` and its loss using the same engine/model and the paired-run rules in research §4.4. Keep separate configurations and checkpoints; use the controller's clipping-aware, standardised derivative targets.
4. Add DART under `dart/` when disturbed collection starts. Add DAgger under `dagger/` when learner-driven collection starts. The existing placeholder `data_collection/dagger.py` may become a compatibility wrapper for `dagger.run`; it must not become a second DAgger implementation.
5. Retire duplicate implementations of `training/dataset.py` and `training/policies.py`; retain re-exports only if existing callers need them. Keep the optional ACT placeholders outside this MLP work. Implement `experiments/run.py` as orchestration rather than another trainer.
6. Update the README and older plan references to implemented entry points, pin actual training dependencies and remove superseded MLP settings when the new loaders land. The narrow `.gitignore` exception already permits the agreed shared summary CSV; keep detailed results, figures, data and checkpoints local.

## 8. Verification required when implemented

- Baseline and BC+J use identical initial weights for paired seeds, identical splits/scales/features/architecture, and the same update budget. Only the added loss differs.
- Baseline gradients come from action loss only. BC+J gradients flow through the student velocity Jacobian, with boundary masking and standardisation checked against an analytical PD example.
- Saving and loading either model preserves actions; inference needs neither the expert nor the Jacobian-loss module. NumPy and PyTorch outputs agree within a documented tolerance if NumPy export is used.
- Clean entry points reject DART/DAgger training manifests. DART saves noisy executed actions separately from clean labels. DAgger records and uses its actual current learner and round data.
- Split, normaliser and checkpoint compatibility checks prevent fitting/evaluation leakage and accidental A/B interchange. Every method is scored by the same offline and closed-loop evaluation definitions.
- Config composition obeys the documented order and pinned identities. Baseline/BC+J, Clean_223/Clean_253 and selection/refit runs have distinct global IDs; exact resume rejects incompatible configs.
- Refit mode produces `final.pt` at its frozen update count, without held-out-validation selection; selection mode alone can produce `best.pt`.
- Internal torque saturation masks are never read from execution `clip_flags`; missing pre-clip metadata causes conservative near-limit masking. Policy/evaluation limits match the source manifests and checkpoint.

No additional architecture or normalisation decisions are implied by this separation. Implementers should follow the latest numerical recipe and settle routine helper names locally without creating parallel training implementations.
