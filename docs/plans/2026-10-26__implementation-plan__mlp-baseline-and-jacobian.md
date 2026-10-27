# MLP Baseline and Jacobian Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a reproducible clean-data BC-MLP pipeline and a separately callable BC+J pipeline, evaluate both in closed loop, and preserve independent future DART/DAgger workflows.

**Architecture:** Shared data, normalisation, network, persistence and update engine under `training/common/`; action-only training under `training/baseline/`; derivative-supervised training under `training/jacobian/`. One evaluator serves all variants. This plan does not implement DART/DAgger collection or create their placeholder packages; the shared interfaces leave those later workflows independent.

**Tech Stack:** Python 3.10, NumPy, PyYAML, PyTorch ≥ 2.7 with CUDA 12.8/`sm_120` verification and CPU fallback, existing Pinocchio/PyBullet/RoboEnv, pytest for targeted verification. No new orchestration framework or hyperparameter optimisation service.

**Spec:** [Unified structure design](../specs/2026-10-26__design__mlp-training-structure.md), [numerical recipe](../research/2026-10-26__research__mlp-training-recipe.md), [clean split/evaluation plan](2026-10-26__plan__mlp-baseline-training.md), [dataset contract](../specs/2026-10-26__spec__expert-datasets.md), [subset inventories](../reports/2026-10-26__report__paired-clean-dataset-subsets.md). Read the spec and recipe before executing any task.

## Global Constraints

- Preserve `core/interfaces.py`, the recorded experts, simulator behaviour and raw datasets; share `core/features.py` for fitting and inference. Add only deployable task-context helpers there if needed.
- A/B have **21/26 inputs**. M1–M6 use `[Linear → LayerNorm → SiLU] × depth`, linear **7 torque + 2 wheel** heads, and expansion `[r, l, l, r]` to the 11-command interface.
- Clean_253 uses **221 fitting / 32 validation** scenarios and **64 untouched test** scenarios per teacher; Clean_223 comparison uses **196 fitting**, with the same 32 validation scenarios loaded from Clean_253.
- Raw `a_expert[t]` corresponds to observation `t` from the first T observations of a T+1 record. Train at full recorded rate, sampling episode-uniform then timestep-uniform.
- Fit normalisers on fitting episodes only. Action loss is `0.5·arm + 0.5·wheel`; Huber δ=1 versus MSE is chosen once by matched M2-A pilots. AdamW, weight decay **1e-4**, batch **1024**; LR candidates **{1e-3, 3e-4, 1e-4}** per architecture/teacher; budget from constant-LR pilots.
- Execution is **1 kHz**, arm **±30 N·m**, wheels **±6 rad/s**, as verified in both Clean_253 manifests. Checkpoint/runtime/data interfaces must agree.
- Pilot and architecture screen use ordinary BC. BC+J is a separate paired experiment using the same initial weights, action batches, scales, schedule, budget and seeds **0–2**; only its loss differs. Offline perturbation relabelling, residual control, ACT and recurrent training are excluded.
- Teacher internal memory, phase/done and simulator truth never enter student features. B may use its known fixed navigation goal and elapsed time; teacher memory remains unavailable.
- `clip_flags` means execution clipping and is never a teacher torque-saturation mask. BC+J uses valid internal-torque rows and scale-correct derivative targets, before execution-time clipping.
- Set `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1` before numerical imports in rollout processes and `torch.set_num_threads(1)` in those processes. Use separate training and rollout RNG streams.
- Preserve independent `regime` (`clean` now; `dart`/`dagger` later), `objective` (`bc`/`bcj`), `teacher` (`A`/`B`) and `fit_stage` (`selection`/`refit`). Use the structure design's global run ID and artifact paths.
- Selection produces validation-selected `best.pt`; optional full-training-data refit produces terminal `final.pt`, without validation selection. All test evaluation follows a frozen protocol.
- Version compact `results/results.csv`, configs, split manifests and code. Keep data, cache, model weights, detailed results and figures local. Never stage those generated artifacts.
- Commands below run from the repository root unless a task explicitly changes directory. The existing interpreter is `../.venv/bin/python`; pytest imports the task package via `tests/conftest.py`. Future task modules run from `final/robotnik_arm_task/`.

## Review Focus

1. Subset metadata retains original train IDs and acquisition settings: reject wrong inventory, split leakage, T/T+1 alignment or stale feature caches (Tasks 3–4).
2. Near-limit torques and standardisation can silently corrupt Jacobian targets: verify valid rows, cross-joint zeros and gradients against an analytical PD map (Tasks 6 and 9).
3. Identical seeds do not imply identical action batches if Jacobian sampling consumes the same RNG; same-device resume must restore all streams (Tasks 5, 8 and 9).
4. B's elapsed time/goal context and `env.step().done` can invalidate deployment/evaluation: enforce causal context and the independent 8-second/3-second/1-second predicate (Tasks 5 and 7).
5. Legacy CSV headers, concurrent evaluators and final refit can mix or overwrite runs: parent-only result indexing, collision checks, frozen test protocol and no refit validation selection (Tasks 2, 7 and 11).

---

## Scope, evidence and file ownership

Read-only inspection on 26 October found a clean worktree at `87cced1`, Python 3.10.20/NumPy 2.2.6 in the existing simulation venv, and missing torch/pytest. Both real source manifests record 30/6 execution limits. Recomputing the split with `default_rng(20261026)` exactly reproduced all 32 IDs in the baseline plan and the 221/32 and 196/32 counts. One sample per teacher confirmed T+1 observations/T actions and B's metadata task context. No test episode arrays were opened.

The plan repairs these documentation conflicts before implementation: residual-noise diagnostics require an already fitted student; expert preflight uses raw observations rather than decoding compressed student features; Huber/MSE comparisons require matching budgets and schedules; LR-screen rollouts follow offline LR selection; run identity follows data/normaliser/λ_J finalisation; cross-device continuation is not bitwise resume.

| Files relative to `final/robotnik_arm_task/` | Responsibility | First task |
| --- | --- | --- |
| `training/common/{__init__,config}.py` | Runtime checks, config resolution, validation, identities | 1–2 |
| `training/common/dataset.py` | Source validation, split/cache, episode/timestep sampling | 3 |
| `training/common/normalisation.py` | Fitting-only statistics and transforms | 4 |
| `training/common/models.py` | Pure configurable MLP | 4 |
| `training/baseline/{__init__,loss}.py` | Action objective only | 4 |
| `training/common/{checkpoint,policies}.py` | Resume state, NumPy export, deployment wrapper | 5 |
| `core/features.py` | Existing builders plus pure B task-context helper | 5 |
| `evaluation/offline.py` | Errors, hold bias, valid-row damping diagnostics | 6 |
| `evaluation/{test_sets,evaluate}.py` | Frozen scenarios, rollout state machine, result index | 7 |
| `training/common/engine.py`, `training/baseline/train.py` | Objective/result contracts first; later shared updates, callbacks and clean BC launcher | 4, 8 |
| `training/jacobian/{__init__,loss,train}.py` | BC+J objective, calibration and launcher | 9 |
| `experiments/{__init__,run}.py`, `experiments/configs/`, `experiments/protocols/`, `experiments/splits/` | Pilot/search/paired orchestration and frozen decisions | 2, 3, 10–11 |
| `training/{dataset,policies,train_mlp}.py` | Compatibility re-exports/wrapper only after replacement works | 8 |

Tests live at repository-root `tests/training/` and `tests/evaluation/`, with tiny generated fixtures and explicit real-data/simulator markers. Add package `__init__.py` files with their first implemented consumer. Do not create `dart/`, `dagger/`, DART configs or DAgger configs in this implementation stage. Existing `data_collection/dagger.py` remains a clearly documented future placeholder.

### Public contracts used by later tasks

Use these names consistently; define these small records in the owning module rather than introducing another framework:

- `common.dataset.EpisodeRef`: `path: Path`, `scenario_id: str`, `teacher: str`, `sha256: str`, `metadata: dict`.
- `build_split` returns `validation_ids`, `screening_ids`, `fitting_ids` keyed by `clean253`/`clean223`, source hashes and the frozen rule/version; IDs are scenario IDs shared across teachers.
- `common.dataset.EpisodeCache`: float32 `x[N,D]`, float32 raw `y[N,9]`, int64 `offsets[E+1]`, ordered episode IDs, diagnostic arrays and `metadata: dict`. The 9 labels are arm 0:7 plus right/left wheels; source recordings remain 11-dimensional.
- `common.normalisation.Normaliser`: serialisable `x_mean`, `x_scale`, `y_mean`, `y_scale`, feature names and scale floors; `transform_x(x)`, `transform_y(y)`, `inverse_y(y)` preserve NumPy/Tensor backend and gradients for Tensor inputs.
- `common.engine.LossResult`: `total: Tensor`, `metrics: dict[str, float]`; `Objective(model, x, y, normaliser, generator) -> LossResult`, where x/y are standardised batches and generator is the independent Jacobian sub-batch RNG. Baseline ignores that generator.
- `common.engine.FitResult`: `run_dir: Path`, `terminal_update: int`, `checkpoints: dict[str, Path]`, `status: str`. `fit(config: dict, train: EpisodeCache, validation: EpisodeCache | None, model: MLP, normaliser: Normaliser, objective: Objective, on_checkpoint: Callable | None = None, resume: Path | None = None) -> FitResult`.
- Config schema groups: `identity`, `model`, `training`, `loss`, `data`, `evaluation`, `interface`, `outputs`, `provenance`. CLI `--set key=value` addresses dotted keys. Identity includes regime/objective/teacher/cohort/seed/fit_stage; unresolved required training fields are allowed only during preparation, never at the first optimiser update.

## Task 1: Make the existing environment capable of verified training

**Files:** Modify root `requirements.txt`, `README.md`; create root `requirements-dev.txt`, `tests/conftest.py`, `tests/training/test_runtime.py`; create task `training/common/__init__.py` and `training/common/config.py`.

**Interfaces:** Produce `configure_runtime(device: str, rollout: bool = False) -> dict`, returning selected device, torch/Python/CUDA versions, CUDA architecture list and thread settings. No training or simulator construction at import.

- [ ] **Step 1: Inspect and install dependencies in the existing venv.** Install a stable Python-3.10-compatible torch ≥2.7 from the CUDA-12.8 wheel index, then pytest. Verify an actual CUDA tensor operation and `sm_120` architecture support; listing a GPU alone is insufficient. If CUDA setup exceeds approximately 30 minutes or kernels fail, record the reason and use CPU. Pin actual installed versions rather than inventing a version now; document the wheel index and fallback. Do not upgrade the simulator dependencies opportunistically.
- [ ] **Step 2: Write failing runtime/import tests.** `test_import_has_no_simulator_or_training_side_effects`, `test_cpu_device_is_explicit`, `test_unavailable_requested_cuda_is_reported`, and `test_rollout_thread_limits_are_one` assert no `RobotEnv` construction, a real CPU tensor result, descriptive device failure/fallback metadata and one torch thread. `tests/conftest.py` inserts only the task directory into `sys.path` and registers `realdata`/`sim` markers.
- [ ] **Step 3: Run the failing tests.** `../.venv/bin/python -m pytest tests/training/test_runtime.py -q`; expect missing runtime helper failures after pytest is installed.
- [ ] **Step 4: Implement the runtime contract.** Keep entry-point environment-variable assignment before NumPy/torch imports; use lazy runtime imports and record precision/backend settings. Torch uses float32 model/data; no automatic AMP or reduced-precision experiment is introduced.
- [ ] **Step 5: Verify and commit.** Repeat the test command; expect PASS. Run `git diff --check`, stage only the listed files, and commit `build: prepare verified MLP training runtime`.

## Task 2: Resolve clean workflow configuration and immutable identities

**Files:** Extend `training/common/config.py`; create `experiments/configs/{common,baseline,jacobian}.yaml`, `experiments/configs/objectives/{bc,bcj}.yaml`, `experiments/configs/teachers/{A,B}.yaml`, `tests/training/test_config.py`; document frozen-protocol ownership in the structure spec.

**Interfaces:** `resolve_config(path: Path, teacher: str, overrides: dict) -> dict`; `validate_config(config: dict, ready_to_train: bool = False) -> None`; `finalise_identity(config: dict, provenance: dict) -> dict`; `create_run_dir(config: dict, resume: bool = False) -> Path`.

- [ ] **Step 1: Write failing configuration tests.** Assert common → workflow → objective → teacher → permitted CLI ordering; list replacement; rejection of unknown keys, null update budget at training, baseline+bcj overrides, wrong 21/26 dimension, DART provenance in clean entry points and a 3-versus-6 wheel-limit mismatch. Assert Clean_223/Clean_253, BC/BC+J and selection/refit yield different global IDs; identical scientific config with a different device/root retains identity; existing directories refuse overwrite without explicit compatible resume.
- [ ] **Step 2: Run** `../.venv/bin/python -m pytest tests/training/test_config.py -q`; expect missing resolver/template failures.
- [ ] **Step 3: Implement the templates and resolver.** Pin clean baseline to `bc`, clean Jacobian to `bcj`; A/B source versions are `markovian-lt0.50-v1`/`template-origin-goal-v1`. Supply the agreed M1–M6 definitions, AdamW 1e-4 weight decay, batch 1024, 500-update warm-up, LR grid and initial action-loss candidates. Leave main-screen update budget and BC+J λ_J unresolved until fitting-only pilot/calibration evidence exists. Start numerical defaults with LayerNorm epsilon 1e-5, input scale floor 1e-6, arm/wheel scale floors 0.1/0.05 and Jacobian boundary tolerance `epsilon_tau=1e-4 N·m`; these floors/tolerance are recorded implementation defaults, frozen before paired comparisons.
- [ ] **Step 4: Finalise identities in two phases.** Resolve a preparation config; load source/split hashes, fit scales and, for BC+J, load calibration before creating the final run ID. Canonicalise JSON, include full normaliser/calibration/source hashes and save the full config hash; exclude generated ID, output roots, timestamps and operational device/workers. Reject short-hash collisions. Runtime interface comes from source manifests and is immutable. Do not consume legacy `settings.yaml` MLP/stride/DART/DAgger placeholders.
- [ ] **Step 5: Verify and commit.** Run runtime/config tests and `git diff --check`; stage listed files and commit `feat: define isolated MLP workflow configurations`.

## Task 3: Load manifests, materialise the split and build raw feature caches

**Files:** Create `training/common/dataset.py`, `experiments/splits/clean253.json`, `tests/training/test_dataset.py`, `tests/training/test_split_realdata.py`; fixtures belong to `tests/conftest.py`.

**Interfaces:** `load_episode_refs(root: Path, teacher: str, split: str) -> list[EpisodeRef]`; `build_split(primary_roots: dict[str, Path], reduced_roots: dict[str, Path]) -> dict`; `build_cache(refs: list[EpisodeRef], feature_builder: Callable, destination: Path) -> EpisodeCache`; `sample_indices(cache: EpisodeCache, size: int, generator: torch.Generator, device: torch.device) -> Tensor`.

- [ ] **Step 1: Write failing data tests.** Use two tiny unequal-length episodes to assert T+1/T alignment, 11→9 label reduction and same-side wheel equality. Reject NaN/Inf, duplicate IDs, wrong teacher version, wrong file hashes, non-clean provenance, unsafe manifest paths, wrong action order and tampered feature-cache metadata. Spy on file opens to prove training never opens `test/*.npz`. Assert timestep sampling gives each episode equal probability rather than each global timestep; use fixed RNG draws and exact offset assertions instead of a flaky significance test.
- [ ] **Step 2: Run** `../.venv/bin/python -m pytest tests/training/test_dataset.py -q`; expect missing loader/cache failures.
- [ ] **Step 3: Implement source/split/cache contracts.** Read derived `counts`, `scenarios_train.csv`, `training_manifest.csv` and `episode_provenance.csv`; do not use the inherited original 256-episode acquisition count. Generate the 32 IDs by the exact baseline rule, assert equality to its listed IDs, save source hashes and 221/32 and 196/32 memberships. Validation for reduced-data comparison comes from Clean_253. Freeze the eight screening IDs by taking the lexicographically smallest validation scenario ID in each of eight equal `u_rel_azimuth` sectors, saving them in this manifest; fail if a sector is empty. Read only allowed fitting/validation episodes, use first T observations and metadata task context with the existing feature builders, and cache float32 unnormalised arrays plus episode offsets/diagnostics once. Publish cache files atomically under hash-keyed `data/cache/`; a partial cache without a completion manifest is invalid.
- [ ] **Step 4: Verify real-data contracts without training.** `../.venv/bin/python -m pytest tests/training/test_dataset.py tests/training/test_split_realdata.py -q`; require exact 32-ID agreement, 221/32/196 counts, source A/B pairing, eight unique screening IDs and no test-array reads. Do not regenerate source data or QA collections.
- [ ] **Step 5: Commit.** Stage source/tests and the small split JSON only; commit `feat: add leakage-safe MLP data and split pipeline`. Cache/raw `.npz` files remain ignored.

## Task 4: Implement fitting-only scales, the shared MLP and action objective

**Files:** Create `training/common/{normalisation,models,engine}.py`, `training/baseline/{__init__,loss}.py`, `tests/training/test_normalisation.py`, `tests/training/test_model.py`, `tests/training/test_action_loss.py`. At this task `engine.py` defines only `LossResult`/`Objective`; the update loop is implemented in Task 8.

**Interfaces:** `fit_normaliser(cache: EpisodeCache, floors: dict) -> Normaliser`; `MLP(input_dim: int, hidden: list[int], layernorm_eps: float = 1e-5)` returns standardised `[B,9]`; `action_loss(pred: Tensor, target: Tensor, kind: str) -> Tensor`; `ActionObjective(kind: str)` implements the Objective call contract without derivative calculations.

- [ ] **Step 1: Write failing numerical tests.** Put an extreme outlier exclusively in validation and assert it cannot alter fitting means/scales. Check population statistics (`ddof=0`), scale floors, inverse round trip and Tensor gradient preservation. Assert output shape 9 and exact A/B parameter counts from the baseline matrix, including M2 21,001/21,641. For one unit arm residual and zero wheel residual, assert MSE group loss is 0.5; for all-zero residuals loss is zero. Assert baseline never calls a Jacobian helper.
- [ ] **Step 2: Run** `../.venv/bin/python -m pytest tests/training/test_normalisation.py tests/training/test_model.py tests/training/test_action_loss.py -q`; expect missing implementation failures.
- [ ] **Step 3: Implement the contracts.** Accumulate step-weighted fitting population moments in float64, save actual floored scales, and transform cached data once for training. Use last-dimension affine LayerNorm and SiLU, no dropout/extra residual paths. Match physical raw labels to standardised 7+2 outputs; expand wheel commands only after inverse scaling. Huber uses standardised δ=1; MSE and Huber share equal action-group weighting.
- [ ] **Step 4: Verify and commit.** Repeat the three test files; require all six architecture counts and float64 round trips within 1e-10. Stage listed files and commit `feat: implement shared MLP and action-only objective`.

## Task 5: Checkpoint round trips and deployable policy inference

**Files:** Create `training/common/{checkpoint,policies}.py`, extend `core/features.py` only for a pure task-context helper, create `tests/training/test_checkpoint.py`, `tests/training/test_policy.py`.

**Interfaces:** `save_checkpoint(path: Path, payload: dict) -> None`; `load_checkpoint(path: Path, expected: dict | None = None) -> dict`; `export_numpy(path: Path, model: MLP, normaliser: Normaliser, metadata: dict) -> None`; `MLPPolicy.load(ckpt_path: str, backend: str = "numpy") -> MLPPolicy`, `reset()`, `act(obs: Observation) -> Action`; `stateful_goal(target: ndarray, rule: dict) -> tuple[ndarray, float]` in `core/features.py`. The NumPy backend requires an explicit `.npz` export; the torch backend requires `.pt`.

- [ ] **Step 1: Write failing persistence/policy tests.** Save/reload a seeded M2 and assert physical actions match. Reject mismatched features, normalisers, teachers and limits. Assert NumPy and PyTorch predictions agree at rtol 1e-4/atol 1e-5 before clipping, finite bounded 7+4 actions after clipping, and exact same-side wheel equality. For B, reset twice and assert its first elapsed feature is 0 both times and its second is 0.001/25; its goal matches known dataset metadata without importing an expert. Assert loading NumPy inference neither imports torch/Jacobian-loss modules nor constructs an environment.
- [ ] **Step 2: Run** `../.venv/bin/python -m pytest tests/training/test_checkpoint.py tests/training/test_policy.py -q`; expect missing persistence/wrapper failures.
- [ ] **Step 3: Implement the contracts.** Save model/optimizer/scheduler state, independent RNG states, config/normaliser/provenance and source/environment hashes atomically. Export Linear/LayerNorm/SiLU weights, affine terms and epsilon explicitly; NumPy LayerNorm uses population variance. Include source checkpoint step/hash in each export. Candidate checkpoints have step-specific exports; selected `inference.npz` corresponds to selected `best.pt` or `final.pt`. Reject `.pt` with the NumPy backend rather than importing torch or guessing a sibling file. `MLPPolicy` lazily constructs deployable FK, computes B's fixed goal from the saved pure geometry rule on first observation, advances elapsed time once per 1 kHz action, and clears context/clock on reset. Never invoke teacher feedback to generate context. Test the helper against existing metadata; leave teacher code untouched.
- [ ] **Step 4: Verify and commit.** Repeat tests and record same-device round trips. Cross-device continuation records its changed environment and is not a bitwise reproducibility claim. Commit `feat: persist MLP policies and export deployable inference` with explicit paths.

## Task 6: Offline control diagnostics, independent of the training objective

**Files:** Create `evaluation/offline.py`, `tests/evaluation/test_offline.py`.

**Interfaces:** `torque_derivative_targets(raw_torque: Tensor, normaliser: Normaliser, epsilon_tau: float, preclip_torque: Tensor | None = None) -> tuple[Tensor, Tensor]`; `velocity_jacobian(model: MLP, x: Tensor) -> Tensor` yields `[B,7,7]`; `offline_metrics(model: MLP, cache: EpisodeCache, normaliser: Normaliser, config: dict) -> dict`.

- [ ] **Step 1: Write failing diagnostic tests.** Analytical PD labels give `-30*I` in physical units, with zero cross-joint terms; non-unit input/output scales give the exact `sigma_qd_j/sigma_tau_i` factor. Missing pre-clip metadata omits near-limit rows even if execution `clip_flags` is false. Verified strictly saturated rows target zero; exact boundary rows are omitted. Episode biases +0.8 and −0.8 both fail the approximate 0.5 N·m screen; opposite episodes cannot cancel. Partition synthetic navigation/approach/hold rows and check physical-unit errors.
- [ ] **Step 2: Run** `../.venv/bin/python -m pytest tests/evaluation/test_offline.py -q`; expect missing metric/derivative helpers.
- [ ] **Step 3: Implement the contracts.** Build targets from raw expert torque and the saved positive tolerance. Use differentiable `torch.func` Jacobians of standardised torque versus feature columns 7:14, holding all other inputs fixed. The helper supports training gradients; diagnostic callers detach results locally rather than disabling graph construction globally. Report bounded, reproducibly sampled hold Jacobians, valid/omitted/saturated coverage, diagonals, cross-joint error and fraction below −33 on valid unclipped rows. Raw teacher/truth fields can define offline diagnostic phases, but never inputs. Save signed per-episode last-second torque bias and its absolute summaries, not a signed maximum.
- [ ] **Step 4: Verify and commit.** Repeat tests, including an all-invalid-mask case without NaN; commit `feat: add offline control and damping diagnostics`.

## Task 7: Shared closed-loop evaluation and versioned result indexing

**Files:** Implement `evaluation/{test_sets,evaluate}.py`; create `tests/evaluation/test_rollout.py`, `tests/evaluation/test_results.py`, `tests/evaluation/test_expert_preflight.py`.

**Interfaces:** `get_split(name: str, split_manifest: Path, teacher: str, allow_test: bool = False) -> list[Scenario]`; `rollout(policy: Policy, scenario: Scenario, config: dict, env_factory: Callable) -> dict`; `evaluate_policy(policy_factory: Callable, scenarios: list[Scenario], config: dict) -> list[dict]`; `append_results(path: Path, rows: list[dict]) -> None`.

- [ ] **Step 1: Write failing rollout/index tests with fake environments.** `info['done']=True` before 8 s must not terminate; a reach at t>8 s starts exactly 3 s of post-reach execution; sustained success requires the final inclusive 1-second observation window below 10 mm; the episode caps at 40 s. Assert true EE information only reaches metric code, action application order is wheels then arm at `RobotEnv.step`, reset is per scenario and `close()` runs on exceptions. Assert a duplicate evaluation key is rejected and the header-only legacy CSV upgrades without inventing results; a nonempty incompatible CSV requires explicit migration and is never silently rewritten.
- [ ] **Step 2: Run** `../.venv/bin/python -m pytest tests/evaluation/test_rollout.py tests/evaluation/test_results.py -q`; expect missing evaluator/index writer failures.
- [ ] **Step 3: Implement selection scenarios and the evaluator.** Use saved 8/32 scenario lists rather than archived T1–T3 generators; reject test access unless an explicit frozen final protocol permits it. The evaluator owns termination, ignores `env.step().done` for stopping and applies checkpoint-frozen 30/6 limits consistently. Workers return outcomes; only the parent writes the compact index, keyed by `(run_id, checkpoint_step, evaluation_split, evaluation_id)`. Save per-scenario detail and latency median/p95/p99 locally. Generic expert adapters accept raw observations; B uses its genuine teacher-side state only for expert reference runs.
- [ ] **Step 4: Run preflight without a trained student.** `../.venv/bin/python -m pytest tests/evaluation/test_expert_preflight.py -m sim -q`; exercise an explicit validation scenario with the frozen A teacher at lead_time=0.50, the common action round trip and evaluator. Check feature dimensions in parallel, not feature-to-expert reconstruction. Then score teacher A/B on the declared validation set to record the reference under the same predicate. Do not require byte-identical outcomes to an older predicate or use test scenarios.
- [ ] **Step 5: Verify and commit.** Run evaluator tests, `git diff --check`, and commit `feat: implement unified validation rollouts and result indexing`. CSV schema migration is included with the writer that owns it, not as a fabricated training result.

## Task 8: Shared update engine and clean baseline entry point

**Files:** Extend `training/common/engine.py`; create `training/baseline/train.py`, `tests/training/test_engine.py`, `tests/training/test_baseline_cli.py`; replace old `training/{dataset,policies,train_mlp}.py` with compatibility adapters where needed; update README and retire legacy MLP settings only when no live consumer depends on them.

**Interfaces:** Preserve Task 4's `LossResult`/`Objective` and implement `FitResult`/`fit(...)` from the public contracts. `training.baseline.train.main(argv: list[str] | None = None) -> int` resolves config, validates clean data, prepares cache/scales/model, finalises identity and supplies `ActionObjective` to `fit`.

- [ ] **Step 1: Write failing engine/CLI tests.** A tiny deterministic fitting fixture must reduce action error. Assert exactly the requested update count, AdamW/schedule values, episodic sampling, no derivative-loss imports and no teacher calls. Assert same-device interrupted/resumed updates exactly match uninterrupted weights and LR after restoring model, optimiser, scheduler, sampling RNG and global RNG. A callback must not alter subsequent action batches; use separate model-init, batch, evaluation and Jacobian streams. Refuse DART/DAgger manifests and invalid overrides at the clean CLI.
- [ ] **Step 2: Run** `../.venv/bin/python -m pytest tests/training/test_engine.py tests/training/test_baseline_cli.py -q`; expect missing engine/launcher failures.
- [ ] **Step 3: Implement training and persistence hooks.** Transfer normalised fitting arrays to the selected device when memory permits; sample indices there without a DataLoader. Fall back to CPU before starting a run if needed, and record the actual device. The common loop accepts an objective, never imports method packages or collection workflows, and rejects nonfinite batches/losses. Derive model-init and action-sampler streams from the paired experiment seed, not the objective-specific global run ID; keep Jacobian and evaluation streams independent. Log offline validation every 500 updates, save `step_<update>.pt` plus resumable `latest.pt` every 2k and terminal update; pilot logging is every 1k. Use warm-up then constant or cosine according to the declared mode. Call validation hooks with isolated RNG/mode and preserve model state on return.
- [ ] **Step 4: Overfit a fitting episode and verify the full path.** Add a separately identified sanity run with one actual fitting ID, a short declared budget and action-loss-only objective; verify cached/offline replay improves, saved inference is bounded, then record a diagnostic student rollout. This is not evidence of full-task success or a substitute for the later validated baseline.
- [ ] **Step 5: Verify and commit.** Run engine/CLI tests and previously passed training tests once, check wrapper `--help`/deprecation behaviour, and commit `feat: train clean MLP baselines through a shared engine`. Do not create DART/DAgger implementations or alter the existing collectors.

## Task 9: Independent Jacobian objective, calibration and launcher

**Files:** Create `training/jacobian/{__init__,loss,train}.py`, `tests/training/test_jacobian.py`, `tests/training/test_jacobian_cli.py`; reuse Task 6 derivatives and Task 8 engine.

**Interfaces:** `JacobianObjective(kind: str, lambda_j: float, epsilon_tau: float, subbatch_size: int = 256)` implements Objective; `calibrate_lambda(model: MLP, x: Tensor, y: Tensor, normaliser: Normaliser, config: dict) -> dict` returns value, reference seed/batch/scales hashes; `training.jacobian.train.main(argv: list[str] | None = None) -> int`.

- [ ] **Step 1: Write failing derivative-training tests.** On an analytical 7-joint linear PD model, loss is zero at the exact target slope and positive at a wrong slope; one gradient step moves a wrong slope toward −30. Assert a nonzero gradient reaches model weights through the Jacobian. Check scale conversion, cross-joint entries, verified saturation, conservative missing-metadata masking and all-invalid batches. λ_J=0 must reduce exactly to baseline action loss and avoid derivative computation. With the same seeded initial weights, baseline/BC+J consume identical action-batch indices despite different Jacobian sub-batches.
- [ ] **Step 2: Run** `../.venv/bin/python -m pytest tests/training/test_jacobian.py tests/training/test_jacobian_cli.py -q`; expect missing objective/calibration failures.
- [ ] **Step 3: Implement the objective and preparation phase.** Reuse action loss from `training/baseline/loss.py`; compute the mean squared error over valid entries of the 7×7 Jacobian on at most 256 samples, with the independent sub-batch generator. Skip only the derivative term when no entries are valid. No wheel derivative penalty or synthetic examples. Calibrate at the selected architecture's seed-0 initialisation on a saved fitting-only batch, `lambda_j = 0.1 * action_loss / jacobian_loss`; reject zero/nonfinite calibration denominators for diagnosis. Save calibration once, restore initial weights/RNG, then freeze its value across seeds and refit. Complete scientific identity only after calibration is fixed.
- [ ] **Step 4: Implement the isolated clean BC+J launcher.** Pin objective `bcj`, require the selected baseline config/scales and frozen calibration, invoke the common engine with the Jacobian objective, and save under the separate global identity. It must not import DART/DAgger or continue the already-trained baseline weights. Loading an inference artifact needs no Jacobian module.
- [ ] **Step 5: Verify and commit.** Run analytical gradient, CLI and baseline-isolation tests and commit `feat: add separately callable damping-supervised MLP training`.

## Task 10: Pilot, LR screening and paired experiment orchestration

**Files:** Create `experiments/{__init__,run}.py`, `experiments/protocols/README.md`, `tests/training/test_experiments.py`; extend config/protocol validation in `training/common/config.py` and README examples.

**Interfaces:** `build_jobs(stage: str, protocol: dict) -> list[dict]`; `rank_offline(metrics: dict) -> tuple`; `rank_rollout(metrics: dict) -> tuple`; `write_frozen_protocol(path: Path, decisions: dict, evidence: list[Path]) -> None`; `experiments.run.main(argv: list[str] | None = None) -> int`.

- [ ] **Step 1: Write failing orchestration tests.** Job lists contain two 50k Huber constant-LR M2-A pilot runs, two matched Huber/MSE comparison fits after budget selection, 36 seed-0 screen runs, eight extra finalist-seed fits, and three BC+J seeds for each selected teacher architecture. Assert no automatic test access, no DART/DAgger jobs and no reuse when source/scales/loss/schedule/config hashes differ. Frozen decisions cannot change in place; updating them creates a new protocol and identity.
- [ ] **Step 2: Run** `../.venv/bin/python -m pytest tests/training/test_experiments.py -q`; expect missing stage/job builder failures.
- [ ] **Step 3: Implement explicit stages.** `prepare`, `pilot`, `loss`, `baseline`, `screen`, `seeds`, `jacobian`, optional `compare223`, `refit`, `test`; `--dry-run` prints jobs and paths without starting training. The runner only resolves configurations and calls existing launchers/evaluators. Store reviewed numerical decisions and evidence hashes under versioned `experiments/protocols/<protocol_id>.yaml`; raw curves/checkpoints remain local. Protocols link the frozen fitting split, source versions, loss/budget/LR decisions and, later, final artifacts; they are not a second configuration-default layer.
- [ ] **Step 4: Define deterministic selection and missing-evidence behaviour.** Rank offline LR proxies lexicographically by hold-bias gate failure, maximum absolute hold bias, arm RMSE and wheel RMSE; record all components. For each architecture/teacher, only the chosen LR's saved checkpoints get eight-scenario rollouts; select its checkpoint by sustained-hold success, reach success, lower hold/final error and lower instability/saturation, with earlier update as the final tie break. Evaluate that checkpoint on all 32 scenarios; compare the 12 architecture/teacher candidates, then confirm top two per teacher on seeds 1–2. Use paired per-scenario results; effectively tied repeated-seed performance favours the smaller model. Do not promote a one-scenario difference as a proven gain. Missing/no plateau evidence after 50k blocks the grid until a documented budget decision, rather than silently choosing 10k or an arbitrary extra budget.
- [ ] **Step 5: Verify and commit.** Assert the dry-run matrix counts and no-test guards, run experiment/config tests, then commit `feat: orchestrate reproducible MLP pilot and paired comparisons`.

## Task 11: Freeze, optional refit and final report artifacts

**Files:** Extend `training/common/engine.py`, `experiments/run.py`, `evaluation/{evaluate,plots}.py`; create `tests/training/test_refit.py`, `tests/evaluation/test_final_protocol.py`; update root README and compact CSV migration with the implemented schema.

**Interfaces:** `validate_final_protocol(protocol: dict) -> None`; `refit` remains a mode of the existing clean objective launcher/common engine, not another trainer. Final evaluation consumes a protocol naming exact checkpoint hashes and final seeds.

- [ ] **Step 1: Write failing freeze/refit tests.** All-training-data refit may open former internal-validation episodes as fitting data, but may not treat them as independent validation or produce `best.pt`; it saves `final.pt` at the fixed update count. Assert its normaliser is refitted, standardised Jacobian targets use new scales and λ_J is unchanged. No test arrays may be opened by fit/refit or training-time plots. A final protocol cannot alter architecture/loss/LR/budget/seed after observing test outcomes; duplicate test evaluation keys refuse rerun.
- [ ] **Step 2: Run** `../.venv/bin/python -m pytest tests/training/test_refit.py tests/evaluation/test_final_protocol.py -q`; expect missing refit/freeze behaviour.
- [ ] **Step 3: Implement the terminal mode and reporting.** Freeze selected objective/architecture/LR, common duration rule, scale floors, tolerance, λ_J/calibration, split/data versions and final seeds before any test evaluation. For clean refit, start anew from declared seeds on all 253 (or declared 223), rebuild scales and save `final.pt` without validation hooks. If skipped, the protocol explicitly selects the frozen `best.pt`. The protocol identifies A/B and BC/BC+J artifacts being compared, not a model chosen after testing. Evaluate the untouched 64 scenarios once per declared artifact/seed; derive summaries and local figures from recorded rows, retaining failed scenarios. No DART/DAgger refit recipe is invented here.
- [ ] **Step 4: Verify and commit implementation before experimental claims.** Run the targeted freeze/refit/report tests and `../.venv/bin/python -m pytest tests/training tests/evaluation -q -m 'not sim and not realdata'`; require PASS, run the explicit realdata/sim checks once, then commit `feat: freeze and report final MLP evaluation artifacts`. Document actual installed versions, commands, outputs and limitations. Staging remains explicit; do not commit weights or generated figures.

## Experiment execution after implementation

Implementation verification and experimental success are different deliverables. Tasks above build the pipeline; the following ordered work uses it. No run is started by merely writing this plan.

1. Prepare split/cache/scales and run raw-expert interface preflight. Complete the separate fitting-episode sanity run once the trainer exists.
2. Run the two constant-LR M2-A Huber pilots at 1e-3 and 3e-4, 50k updates each. Inspect physical-unit RMSE/hold-bias curves every 1k; record plateau evidence and choose approximately 1.5× plateau updates. If no plateau exists, resolve budget/extension before proceeding.
3. At the chosen budget/LR, compare matched warm-up/cosine Huber and MSE runs on identical data, seed and initial weights, including eight-scenario rollouts; freeze the loss. Record this as a separate pilot stage.
4. Obtain the verified M2-A baseline, then M2-B, with offline diagnostics and 8/32 closed-loop results. Once residuals exist, run the separately labelled expert-noise diagnostic; differences from the student failure are information, not proof of a broken harness. Reuse an existing run only on exact full-config identity.
5. Screen M1–M6 × A/B × three LRs offline; roll out only each architecture/teacher's selected LR checkpoints; add the eight finalist-seed runs. This is 36+8 fits, plus separately identified pilots/first-baseline work unless identical runs are reused.
6. Run BC+J for one selected architecture per teacher on seeds 0–2, pairing against those plain-BC fits. Calibrate λ_J once per teacher/selected architecture and freeze it. Optional 223 comparison remains a separately declared data-selection experiment.
7. Freeze the final method/artifact list and seed set using validation, optionally refit clean finalists, and evaluate declared artifacts on the 64 test scenarios once. Save report numbers in the compact CSV and figures locally. Report failures honestly; never infer successful robot control from a low offline loss.

Future DART/DAgger require their own implementation plans under the reserved directories, initialising from explicitly chosen BC or BC+J checkpoints and keeping no-J controls. Their collection budgets/noise calibration/aggregation policies are not prerequisites for this clean-data implementation.

## Completion evidence and self-review

- Scope map: spec configuration/identity/artifacts → Tasks 1–2, 5, 7, 11; source/data/scales → Tasks 3–4; baseline pipeline → Tasks 5–8; Jacobian extension → Tasks 6 and 9; controlled experiment order → Tasks 10–11. Later collection workflows are intentionally deferred, not partially implemented.
- Recheck every signature against the public contracts before implementation. In particular, raw labels have 11 source dimensions but nine trained channels; physical-unit 7×7 Jacobians use the qd feature slice 7:14.
- Every task has a specific failing check, implementing owner, passing command and focused commit. Run already-passing broad checks again only after a relevant change or before the final integrated verification.
- Planning-time checks are document/source inspections, not torch training tests: torch and pytest are absent at writing. The runtime installation and all future test PASS claims must be demonstrated during execution.
- No decision requiring new collection or test-driven tuning is delegated to an implementer as an implicit default. If fitting pilots cannot establish a feasible budget, CUDA fails without a viable CPU throughput, or a required manifest is incompatible, report the evidence and resolve that experimental constraint before the affected stage.

## Assertion anchors for each task's failing-test step

These assertions belong in the owning task's listed test file, in addition to its other named failure cases. Build fixtures in that file using the public contracts: `unit_normaliser` has zero means/unit scales and the correct feature names; the PD fixture has seven diagonal velocity slopes of −20 and zero intercept/cross terms; training fixtures use tiny unequal-length episodes, a four-update config and separate fitting/validation caches; the result-row fixture is complete for the target index schema. These are future test requirements, not tests executed during planning.

```python
# Task 1 — tests/training/test_runtime.py
def test_rollout_threads_are_one():
    report = configure_runtime("cpu", rollout=True)
    assert report["device"] == "cpu"
    assert report["torch_threads"] == 1

# Task 2 — tests/training/test_config.py
def test_baseline_cannot_be_changed_to_bcj():
    with pytest.raises(ValueError):
        resolve_config(Path("final/robotnik_arm_task/experiments/configs/baseline.yaml"),
                       "A", {"identity.objective": "bcj"})

# Task 3 — tests/training/test_split_realdata.py (mark realdata)
def test_real_split_contract(primary_roots, reduced_roots):
    split = build_split(primary_roots, reduced_roots)
    assert len(split["validation_ids"]) == 32
    assert len(split["screening_ids"]) == 8
    assert len(split["fitting_ids"]["clean253"]) == 221
    assert len(split["fitting_ids"]["clean223"]) == 196

# Task 4 — tests/training/test_action_loss.py
def test_action_groups_have_equal_weight():
    pred = torch.cat((torch.ones(2, 7), torch.zeros(2, 2)), dim=1)
    assert action_loss(pred, torch.zeros_like(pred), "mse").item() == 0.5

# Task 5 — tests/training/test_policy.py
def test_numpy_backend_refuses_torch_format(tmp_path):
    path = tmp_path / "best.pt"
    path.write_bytes(b"not-a-numpy-export")
    with pytest.raises(ValueError):
        MLPPolicy.load(str(path), backend="numpy")

# Task 6 — tests/evaluation/test_offline.py
def test_execution_clipping_is_not_internal_saturation(unit_normaliser):
    torque = torch.tensor([[0.0] * 7, [30.0] * 7])
    target, valid = torque_derivative_targets(torque, unit_normaliser, 1e-4)
    assert valid.shape == (2, 7)
    assert valid[0].all() and not valid[1].any()
    torch.testing.assert_close(target[0], -30 * torch.eye(7))

# Task 7 — tests/evaluation/test_results.py
def test_duplicate_evaluation_is_rejected(tmp_path, complete_row):
    path = tmp_path / "results.csv"
    append_results(path, [complete_row])
    with pytest.raises(ValueError):
        append_results(path, [complete_row])

# Task 8 — tests/training/test_engine.py
def test_engine_honours_update_count(config, train_cache, validation_cache,
                                    model, normaliser, action_objective):
    result = fit(config, train_cache, validation_cache, model, normaliser, action_objective)
    assert result.terminal_update == config["training"]["updates"] == 4

# Task 9 — tests/training/test_jacobian.py
def test_derivative_loss_reaches_parameters(linear_pd_model, unit_normaliser):
    objective = JacobianObjective("mse", lambda_j=1.0, epsilon_tau=1e-4)
    result = objective(linear_pd_model, torch.zeros(2, 21), torch.zeros(2, 9),
                       unit_normaliser, torch.Generator().manual_seed(1))
    assert result.total.item() > 0
    result.total.backward()
    assert linear_pd_model.weight.grad[:, 7:14].abs().sum().item() > 0

# Task 10 — tests/training/test_experiments.py
def test_architecture_grid_is_offline_only(frozen_protocol):
    jobs = build_jobs("screen", frozen_protocol)
    assert len(jobs) == 6 * 2 * 3
    assert all(j["identity"]["regime"] == "clean" for j in jobs)
    assert all(j["identity"]["objective"] == "bc" for j in jobs)

# Task 11 — tests/training/test_refit.py
def test_refit_has_no_validation_selected_artifact(refit_result):
    assert (refit_result.run_dir / "final.pt").is_file()
    assert not (refit_result.run_dir / "best.pt").exists()
```

Primary API references: [PyTorch LayerNorm epsilon/statistics](https://docs.pytorch.org/docs/2.7/generated/torch.nn.LayerNorm.html), [differentiable Jacobian transform](https://docs.pytorch.org/docs/2.14/generated/torch.func.jacrev.html), [reproducibility limits and RNG control](https://docs.pytorch.org/docs/2.7/notes/randomness.html). Verify the installed stable version's matching APIs in Task 1; these references do not require upgrading to the documentation site's current release.
