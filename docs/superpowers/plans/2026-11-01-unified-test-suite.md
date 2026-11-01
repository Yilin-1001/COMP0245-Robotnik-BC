# Unified Test Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Implement and execute the approved 240-scene, 24-model plus expert evaluation and publish complete result artifacts.

**Architecture:** Add standalone scorer/rollout, frozen-scene/model preparation, resumable runner, and report/plot modules. Reuse current policy and environment adapters; preserve existing evaluation behavior.

**Tech Stack:** Python 3.10, NumPy, SciPy, matplotlib, multiprocessing, pytest; existing Pinocchio/PyBullet runtime.

**Spec:** `docs/superpowers/specs/2026-11-01-unified-test-suite-design.md` (approved by user).

## Global Constraints

- Only <10 mm with continuous 3 s / T5 20 s counts as success; no 8 s gate, no secondary 20 mm success.
- Run the fixed full duration; later normal drift retains earlier success, any exception/nonfinite value overrides it with F/failure.
- Grade last inclusive 1 s maximum with boundaries 5,10,20,50 mm; precision independent of success.
- Preserve T0 CSV/hash; 240 total scenes, T4 24 paired translations, T1 includes 10–15 m at 120 s; no T6/extreme group.
- Use v1 2026-10-28 and v2 2026-11-01 validation-selected round, all seeds, step_42000; expert lead_time=.50.
- Freeze before formal evaluation; use separate development fixture for smoke checks.
- 6000 jobs, atomic persistence, identity-checked resume, plot-only without physics, report all scenes and expert-success subset.
- Preserve pre-existing unrelated workspace changes. Commit explicit task paths only.

## Review Focus

- Failure after initial success must override success; an incomplete final window must never earn A–E.
- No teacher successes / no student successes must produce N/A times and valid plots.
- Multiple seeds share scenes; bootstrap resamples scene clusters and coverage distinguishes missing models.
- Partial/corrupted job results must not count as compatible completed jobs; a changed model/protocol/code must reject resume.
- T4 intentional pairing must not be rejected as duplicate geometry or misrepresented as independent relative tasks.

### Task 1: Dedicated continuous-hold rollout

**Files:** Create `evaluation/suite_rollout.py`; test `tests/evaluation/test_suite_rollout.py`.
**Interfaces:** `score_errors(errors, control_hz, hold_s, failure=None)->dict`; `rollout(policy, scenario, duration_s, hold_s, interface, env_factory=None)->dict`.

- [ ] Write scripted-environment tests for strict 10 mm, streak reset, first completion including t=0, long hold, fixed duration, late drift, exception override, inclusive terminal second, limits/observation interface.
- [ ] Run tests RED (new module unavailable), implement scorer and dedicated rollout, run GREEN and existing rollout checks.
- [ ] Commit verified task batch with this plan.

### Task 2: Frozen scenarios and terminal-model selection

**Files:** Create `evaluation/suite_manifest.py`; test `tests/evaluation/test_suite_manifest.py`.
**Interfaces:** `build_manifest(task_dir=TASK_DIR)->dict`; `freeze_manifest(path, manifest)->dict`; `load_manifest(path)->dict`; `select_models(rows, seeds, paths=None, task_dir=TASK_DIR)->(models,missing)`; `scene_from_record(record)`.

- [ ] Write tests for count/ranges, original T0 hash failure, T4 exact paired transform, freeze immutability, selected DAgger round/ties, filtering/missing models, checkpoint step mismatch.
- [ ] Watch RED; implement LHS new scenarios and source-overlap audit, canonical hashes and model metadata resolution; watch GREEN.
- [ ] Verify actual default selection resolves 24 models without running formal scenes, commit.

### Task 3: Resumable entry point and statistics/figures

**Files:** Create `evaluation/test_suite.py`, `evaluation/suite_report.py`, `evaluation/suite_plots.py`; tests `tests/evaluation/test_test_suite.py`, `tests/evaluation/test_suite_report.py`.
**Interfaces:** runner CLI in spec; per-job JSON contains identity/result/history; `generate_report(result_dir)->dict`, reads run/scenes/results only; plotting `plot_results(result_dir, rows, summary)->None`.

- [ ] RED tests: CLI filters, atomic job identities and mismatch/corruption, resumption without rerun, aggregate success/grades/time/coverage, expert-zero subset, paired T4, plot-only artifact generation.
- [ ] Implement preflight/pinned jobs, worker processes with one-thread numerical libraries, atomic JSON checkpoints, summaries and seven figure families, metadata/latency benchmark and detailed failure records.
- [ ] GREEN targeted tests; run project suite once; fix relevant regressions with RED/GREEN. Commit.

### Task 4: Review, development smoke, freeze and full execution

**Files:** New usage guide `docs/2026-11-01__guide__unified-test-suite.md`; frozen JSON `experiments/splits/unified-test-v1.json`; generated results stay ignored.

- [ ] Fresh independent whole-change review as required by executing-plans. Address important issues with regression tests before formal runs.
- [ ] Execute smoke scenarios from an independent development fixture for expert and feature-v1/gate models, confirm actual physics works and measure throughput. Do not use formal scenes for tuning.
- [ ] Freeze 240 formal scenes and commit verified source/manifest/guide changes before full run.
- [ ] Start all 6000 formal rollouts; persist incremental progress and provide updates; resume compatible jobs after interruptions.
- [ ] Generate and visually inspect plots; verify 6000 unique outcomes and 25 policies x 240 scenes, hashes, F categories, summary tables, and report both all-scene and teacher-success subsets.
- [ ] Record real runtime/results and final verification in ledger and final response; retain all user changes and report source commit hashes and artifact paths.
