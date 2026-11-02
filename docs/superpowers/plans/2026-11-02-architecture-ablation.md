# Architecture Ablation (R4/R6/R8) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and evaluate 9 new MLP architectures × R4/R6/R8 × 3 seeds on fixed data with a frozen recipe. Compare them with the existing 256×3 models on validation, the hold test, the 240-scene unified test suite and isolated latency.

**Architecture:** Extend the existing pipeline; do not build a parallel one. `training.datasets.train` gains an opt-in "copy λ_J from a source run". `experiments/ablation/run.py` learns lists of architectures, reference models that are evaluated but never trained, and a `--rows` filter. The unified test suite is run **unchanged** through `--models`, so its code hash matches the reference run. A new report module groups the rollouts by (row, architecture). A small latency benchmark measures the NumPy forward pass in isolation.

**Tech Stack:** Python 3.10, PyTorch (training), NumPy (inference), PyBullet/Pinocchio (simulation), pytest, matplotlib, YAML manifests.

**Spec:** `docs/superpowers/specs/2026-11-02-architecture-ablation-design.md`

## Global Constraints

- All paths are relative to `final/robotnik_arm_task/` unless they start with `tests/` or `docs/` (repo root). Run code from `final/robotnik_arm_task/` with `../../../.venv/bin/python`. Run tests from the repo root with `../.venv/bin/python -m pytest`.
- Frozen recipe (copy of the v1 manifest's `recipe`): `identity.objective: bcj`, `training.lr: 1.0e-3`, `training.updates: 42000`, `loss.action: huber`, `loss.arm_weight: 0.5`, `loss.wheel_weight: 0.5`. Seeds `[0, 1, 2]`. Terminal checkpoint only.
- Rows: `R4-dart-gate`, `R6-gate-dagger-r3`, `R8-dart-gate-dagger-r3`.
- λ_J source configs (seed-0 256×3):
  - R4 `checkpoints/ablation/2026-10-28__training-ablation-v1/R4-dart-gate__MLP01-256x3__s0__v1__2026-10-28/config.yaml` (0.07439793449003185)
  - R6 `checkpoints/ablation/2026-11-01__training-ablation-v2/R6-gate-dagger-r3__MLP01-256x3__s0__v2__2026-11-01/config.yaml` (0.00028908631878552324)
  - R8 `checkpoints/ablation/2026-11-01__training-ablation-v2/R8-dart-gate-dagger-r3__MLP01-256x3__s0__v2__2026-11-01/config.yaml` (0.12621421349827852)
- Registry: MLP02 [32,32,32], MLP03 [64,64], MLP04 [64,64,64], MLP05 [64,64,64,64], MLP06 [128,128,128], MLP07 [256,128,64], MLP08 [256,256], MLP09 [256,256,256,256], MLP10 [512,512,512].
- Experiment `architecture-ablation` v1. `<LAUNCH>` = the YYYY-MM-DD on which Task 6 starts, used identically in every new manifest's file name and `date:`.
- Never overwrite, retrain or rename existing checkpoints, datasets, `results/results.csv` rows or `results/test/2026-11-01_215340__R1-R8`. Never edit `core/features.py`, `data_collection/collect.py`, `settings.yaml`, or any file listed in `evaluation/test_suite.py:code_hashes()`.
- Test results never choose anything. The lr rule uses validation only.
- Work on a new branch `feature/architecture-ablation` from `develop` in the main checkout (data and checkpoints are git-ignored and exist only there). The user has unrelated uncommitted edits, so every commit stages **only** the files named in its task.

## Review Focus

1. **Re-validating a reference model:** its key is already in `results/results.csv`, which refuses duplicates. Validation of a reference must write its detail CSV and skip the index append, or the run crashes after hours of rollouts (test in Task 2).
2. **Interrupting and rerunning the runner** mid-grid: finished runs are skipped, a `latest.pt` resumes, and references are never trained. A missing reference model is an error, not a training job (test in Task 2).
3. **Comparing test runs made with different code:** if any `code_sha256` entry, the scene manifest hash or the protocol differs from the reference run, the report must refuse to pool them (test in Task 4).
4. **Tapered architecture labels** (`MLP07-256-128-64`) contain hyphens. Parsing the row and architecture out of a model id must split on `__`, not `-` (test in Task 4).
5. **A λ_J source that does not match the run:** a different variant, different datasets or a different normaliser must fail before training, not train silently with the wrong weight (test in Task 1).

---

### Task 1: Architecture registry and copied λ_J calibration

**Files:**
- Modify: `experiments/configs/datasets/base.yaml` (`model.architectures`)
- Modify: `training/datasets/train.py` (`prepare()`, new `copy_calibration`)
- Modify: `tests/datasets/test_datasets.py:69-75` (registry assertion)
- Create: `tests/datasets/test_fixed_calibration.py`

**Interfaces:**
- Produces: `copy_calibration(config: dict, normaliser_sha256: str, source: Path) -> tuple[dict, str]`, which returns `(record, record_sha256)`. When `config["loss"]["jacobian"]["calibration"]` is a path string, `prepare()` calls it instead of `calibrate()`. The config key `loss.jacobian.calibration` already exists (`objectives/bcj.yaml`, null = auto) and is CLI-permitted.

- [ ] **Step 1: Write failing tests** in `tests/datasets/test_fixed_calibration.py`. Build the config with `resolve("R4-dart-gate", {...})`. Write a source `config.yaml` to `tmp_path` containing `variant`, `identity.objective: bcj`, `data.datasets`, `loss.jacobian.lambda_j: 0.0744`, `provenance.calibration: {...}`, `provenance.calibration_sha256: "abc"` and `provenance.normaliser_sha256: "n1"`.
  - `test_copies_lambda_and_source_hash`: `record["lambda_j"] == 0.0744`, the returned sha equals `"abc"`, and `record["copied_from"]` ends with `config.yaml`.
  - `test_rejects_other_variant` / `test_rejects_other_datasets` / `test_rejects_other_normaliser`: `pytest.raises(ValueError, match="variant"|"datasets"|"normaliser")`.
  - `test_rejects_disagreeing_given_lambda`: the config already has `lambda_j: 0.07` → `match="lambda_j"`.
  - `test_source_must_be_bcj`: a source with objective `bc` → `match="bcj"`.
- [ ] **Step 2: Run** `../.venv/bin/python -m pytest tests/datasets/test_fixed_calibration.py -v`. Expected: FAIL (ImportError: copy_calibration).
- [ ] **Step 3: Implement `copy_calibration` in `training/datasets/train.py`.** `record = {**source provenance.calibration, "copied_from": <path relative to TASK_DIR>, "source_config_sha256": sha256(file)}`; the return sha is the source's `calibration_sha256`. In `prepare()`, inside the `bcj` branch: if `calibration` is a str, set `lambda_j`, set `calibration_record = {"path": f"fixed from {rel}", "lambda_j": λ, "sha256": sha}` and set `provenance.calibration` and `provenance.calibration_sha256`. Otherwise keep today's code path byte-for-byte.
- [ ] **Step 4: Add MLP02–MLP10 to `base.yaml`** (values in Global Constraints). In `test_variants_resolve`, change the registry assertion to `config["model"]["architectures"]["MLP01"] == [256, 256, 256]` and assert that `set(config["model"]["architectures"]) == {f"MLP{i:02d}" for i in range(1, 11)}`.
- [ ] **Step 5: Run** `../.venv/bin/python -m pytest tests/datasets tests/training -q`. Expected: all pass.
- [ ] **Step 6: Real-data check.** From `final/robotnik_arm_task/`, run `../../../.venv/bin/python -m training.datasets.train --variant R4-dart-gate --dry-run --set identity.version=v1 --set identity.date=2026-11-02 --set identity.architecture=MLP02 --set training.lr=1.0e-3 --set training.updates=42000 --set loss.action=huber --set loss.arm_weight=0.5 --set loss.wheel_weight=0.5 --set loss.jacobian.calibration=<R4 source path>`. Expected: `"lambda_j": 0.07439793449003185` and a run id `R4-dart-gate__MLP02-32x3__s0__v1__2026-11-02`. Repeat for `R8-dart-gate-dagger-r3` with its source (expected 0.12621421349827852).
- [ ] **Step 7: Commit** `git add final/robotnik_arm_task/experiments/configs/datasets/base.yaml final/robotnik_arm_task/training/datasets/train.py tests/datasets/test_datasets.py tests/datasets/test_fixed_calibration.py && git commit -m "feat: MLP02-MLP10 registry and copied lambda_J calibration"`

### Task 2: Runner support for architecture grids and references

**Files:**
- Modify: `experiments/ablation/run.py` (`overrides`, `runs`, `train`, `validate`, `summary`, `main`)
- Modify: `tests/datasets/test_ablation_runner.py`

**Interfaces:**
- Consumes: Task 1's `loss.jacobian.calibration` override.
- Manifest keys, all optional, so the v1 manifest keeps resolving to the same 12 runs: `architectures: [MLP..]` (instead of `architecture`), `lambda_j_source: {row: config path}`, `references: {row: path template with {seed}}`.
- Produces: `overrides(m: dict, seed: int, architecture: str, row: str) -> dict`. `runs(m)` entries gain `"architecture"` (label, e.g. `MLP07-256-128-64`), `"parameters"` (int) and `"reference"` (bool). `main` gains `--rows ROW [ROW ...]` (must be a subset of the manifest rows, else `SystemExit` naming the bad row). `runs.csv` gains the columns `architecture` and `parameters`. `summary.csv` and `summary.md` have one line per (row, architecture).

- [ ] **Step 1: Write failing tests** in `tests/datasets/test_ablation_runner.py`. Keep the three existing tests unchanged; they pin v1 compatibility. Add the following, using a manifest dict written to `tmp_path`:
  - `test_grid_manifest_resolves_rows_x_architectures_x_seeds_plus_references`: with 2 architectures, 1 row and 2 seeds, plus a reference template pointing at an existing R4 v1 model, `len(runs(m)) == 4 + 2`. A run named `R4-dart-gate__MLP07-256-128-64__s1__v1__2026-11-02` exists. Reference entries have `reference is True`, `architecture == "MLP01-256x3"` and `parameters == 141321`.
  - `test_overrides_carry_architecture_and_lambda_source`: `overrides(m, 0, "MLP02", "R4-dart-gate")["loss.jacobian.calibration"]` equals the manifest's R4 source, and `["identity.architecture"] == "MLP02"`.
  - `test_missing_reference_is_an_error_not_a_training_job`: a reference template pointing at a nonexistent file → `train(m, 1)` raises `FileNotFoundError` before starting any process (monkeypatch `subprocess.Popen` to fail the test if it is called).
  - `test_reference_validation_skips_results_index`: monkeypatch the rollout pool and `append_results`. Validating a reference writes `validation/<name>__validation-student.csv` and passes **no** reference row to `append_results`.
  - `test_rows_filter_rejects_unknown_row`: `main(["--manifest", p, "--rows", "R9", "--stage", "summary"])` → `SystemExit`.
- [ ] **Step 2: Run** `../.venv/bin/python -m pytest tests/datasets/test_ablation_runner.py -v`. Expected: the new tests FAIL and the old ones PASS.
- [ ] **Step 3: Implement** in `run.py`. For references, the name is the model's stem, the dir is its parent, and the architecture label comes from that dir's `config.yaml` via `architecture_label`. `parameters` is the sum of the numeric array sizes in the model `.npz` (as `evaluation/suite_manifest._artifact_metadata` counts them), or `None` while a run is untrained. `train()` excludes references and raises on missing ones. `validate()` skips `append_results` for reference rows. `summary()` groups by `(row, architecture)` and orders by row and then by parameter count. `hold()` is unchanged.
- [ ] **Step 4: Run** `../.venv/bin/python -m pytest tests/datasets -q`. Expected: all pass.
- [ ] **Step 5: Commit** `git add final/robotnik_arm_task/experiments/ablation/run.py tests/datasets/test_ablation_runner.py && git commit -m "feat: ablation runner handles architecture grids, references and a rows filter"`

### Task 3: Cost model and isolated latency benchmark

**Files:**
- Create: `evaluation/latency_bench.py`
- Create: `tests/evaluation/test_latency_bench.py`

**Interfaces:**
- Produces: `mlp_cost(input_dim: int, hidden: list[int], outputs: int = 9) -> dict`, which returns `{"parameters": int, "flops": int}`. Parameters include the LayerNorm γ/β (2 per hidden unit). FLOPs = 2·Σ(in·out) over the linear layers.
- Produces: `time_predict(model: str, calls: int = 20000, warmup: int = 2000, repeats: int = 3) -> dict`, which returns `{"median_us", "p99_us", "repeat_medians_us": list}`. It uses `MLPPolicy.load(model).predict(x)` with `x = policy.x_mean` (one feature row; normalisation included, FK excluded).
- CLI: `python -m evaluation.latency_bench --models <npz ...> --out <csv>` writes one row per model: `model, architecture, parameters, flops, npz_bytes, median_us, p99_us`. Pin `OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=1` at the top of the module, before the NumPy import, as `evaluation/test_suite.py` does.

- [ ] **Step 1: Write failing tests:**
  - `test_cost_matches_known_counts`: `mlp_cost(22, [256]*3) == {"parameters": 141321, "flops": 278016}`, `mlp_cost(22, [32]*3) == {"parameters": 3337, "flops": 6080}`, `mlp_cost(22, [256, 128, 64]) == {"parameters": 48521, "flops": 94336}`.
  - `test_cost_agrees_with_exported_model` (`@pytest.mark.realdata`): for the R4 v1 s0 npz, `mlp_cost(...)["parameters"]` equals the sum of the npz numeric array sizes.
  - `test_time_predict_returns_positive_stats` (`@pytest.mark.realdata`): with `calls=200, warmup=20, repeats=2` on the R4 v1 s0 npz, `0 < median_us <= p99_us` and `len(repeat_medians_us) == 2`.
- [ ] **Step 2: Run** `../.venv/bin/python -m pytest tests/evaluation/test_latency_bench.py -v`. Expected: FAIL (module missing).
- [ ] **Step 3: Implement** `evaluation/latency_bench.py`. Time each call with `time.perf_counter_ns()`. `median_us` is the median of the per-repeat medians, and `p99_us` comes from the pooled calls.
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit** `git add final/robotnik_arm_task/evaluation/latency_bench.py tests/evaluation/test_latency_bench.py && git commit -m "feat: MLP cost model and isolated NumPy latency benchmark"`

### Task 4: Architecture report (grouping, comparability, lr rule, recommendation, figures)

**Files:**
- Create: `experiments/ablation/architecture_report.py`
- Create: `tests/datasets/test_architecture_report.py`

**Interfaces:**
- Consumes: `results/test/<run>/rollouts.csv` and `run.json`, the runner's `runs.csv`, Task 3's latency CSV and each run's `metrics.csv` (last row `val_arm_rmse_nm`).
- Produces:
  - `split_model_id(model_id: str) -> tuple[str, str]`, which returns `("R4", "MLP07-256-128-64")` from `R4-dart-gate__MLP07-256-128-64__s0__v1__2026-11-03`. The row is the part before the first `-` of the first `__` field.
  - `check_comparable(reference_run: dict, new_run: dict) -> None` raises `ValueError` unless `scene_manifest_sha256`, `identity.protocol`, `identity.interface` and the whole `identity.code_sha256` are equal.
  - `aggregate(rollouts: list[dict], rows: set[str]) -> list[dict]` returns one entry per (row, architecture, suite) and per (row, architecture, "all"). Each entry has `k, n, rate, ci_lo, ci_hi` (Wilson from `experiments.ablation.run.wilson`), grade counts A–F and the median `completion_time_s` over successes. The expert and rows outside `rows` are dropped.
  - `lr_decision(holds: dict[str, dict[float, int]], tolerance: int = 2) -> str` returns `"keep"` if, for every architecture, `holds[a][1e-3] >= max(holds[a].values()) - tolerance`, else `"screen"`.
  - `recommend(table: list[dict], margin: float = 0.03) -> dict[str, str]` returns, per row, the smallest-parameter architecture whose "all" rate is `>= rate(MLP01-256x3) - margin`.
  - CLI: `python -m experiments.ablation.architecture_report --experiment results/ablation/<LAUNCH>__architecture-ablation-v1 --reference-test results/test/2026-11-01_215340__R1-R8 --tests <run dirs ...> --latency <csv>` writes `architecture_summary.csv`, `architecture_summary.md`, a `README.md` note in each new test-run dir saying that its row-level tables pool architectures and pointing to this summary, and figures in `figures/architecture/`. The figures are: success vs parameters (log x, one line per row, seed range as error bars), success vs median latency, the width line (×3) and the two depth lines (256 and 64) as separate panels, and validation arm RMSE vs all-suite success. Before writing figure code, load the `dataviz` skill.

- [ ] **Step 1: Write failing tests** (synthetic rows, no simulator):
  - `test_split_model_id_handles_tapered_labels`: covers the example above and `R6-gate-dagger-r3__MLP01-256x3__s2__v2__2026-11-01` → `("R6", "MLP01-256x3")`.
  - `test_check_comparable_rejects_code_or_manifest_change`: equal dicts pass. Changing one `code_sha256` value raises `match="code"`. Changing `scene_manifest_sha256` raises `match="manifest"`.
  - `test_aggregate_pools_seeds_per_architecture_and_drops_expert`: rows for R4/MLP02 (2 seeds × 2 scenes, 3 successes) and R4/MLP01 plus an expert row give `k == 3, n == 4` for `("R4", "MLP02-32x3", "all")`, and no entry with row `expert`.
  - `test_lr_decision`: `{"MLP02-32x3": {3e-4: 20, 1e-3: 25, 3e-3: 26}, "MLP10-512x3": {3e-4: 30, 1e-3: 29, 3e-3: 28}}` → `"keep"`. Setting `MLP02-32x3`'s `1e-3` to 23 → `"screen"`.
  - `test_recommend_picks_smallest_within_margin`: with reference rate 0.95, MLP02 0.90, MLP04 0.93 and MLP06 0.96 → `"MLP04-64x3"`.
- [ ] **Step 2: Run** `../.venv/bin/python -m pytest tests/datasets/test_architecture_report.py -v`. Expected: FAIL (module missing).
- [ ] **Step 3: Implement** `experiments/ablation/architecture_report.py`. The markdown table has one line per (row, architecture): parameters, FLOPs, median/p99 latency, validation hold per seed and pooled with CI, test "all" rate with CI, per-suite T0–T5 rates, hold-test stiffness max, damping diagonal and max real eigenvalue. Below the table, put the recommendation per row, with its speed-up and parameter ratio against MLP01. Then add a line stating the T0 ceiling and the expert determinism check: the expert's per-scene success in each new run compared with the reference run, given as a count of identical scenes.
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit** `git add final/robotnik_arm_task/experiments/ablation/architecture_report.py tests/datasets/test_architecture_report.py && git commit -m "feat: architecture ablation report with comparability checks"`

### Task 5: Manifests and preflight

**Files:**
- Create: `experiments/ablation/<LAUNCH>__architecture-ablation-v1.yaml`
- Create: `experiments/ablation/<LAUNCH>__architecture-lr-check-{3e-4,1e-3,3e-3}-v1.yaml` (experiment names `architecture-lr-check-3e-4` etc., so directories never collide)

**Interfaces:** Consumes Tasks 1–2 (manifest keys). The grid manifest has `architectures` MLP02–MLP10, the three rows, seeds 0–2, the frozen recipe, the `lambda_j_source` map, and `references`. The references map R4 to the v1 template and R6/R8 to `checkpoints/ablation/2026-11-01__training-ablation-v2/<row>__MLP01-256x3__s{seed}__v2__2026-11-01/<same>.npz`. `evaluation` is copied from the v1 manifest (modes student + arm, the same 8 hold-test scenarios). Each lr-check manifest has `architectures: [MLP02, MLP10]`, rows `[R4-dart-gate]`, seeds `[0]`, its own `training.lr`, the R4 λ source, no references, and `validation_modes: [student]`.

- [ ] **Step 1: Write the four manifests**, each with a header comment in the v1 manifest's style that says what it is and how to run it.
- [ ] **Step 2: Preflight.** Load each manifest with `experiments.ablation.run.load_manifest` and `runs` in a Python one-liner. Expected: the grid has 81 trainable runs + 9 references, and no name duplicates. Each lr manifest has 2 runs. Every reference `.npz` exists.
- [ ] **Step 3: Dry-run one run per row** via `training.datasets.train --dry-run` with the manifest's overrides (as in Task 1 Step 6). Expected: λ_J equals the source value for R4, R6 and R8.
- [ ] **Step 4: Run the full test suite:** `../.venv/bin/python -m pytest -q`. Expected: everything passes (168 + new).
- [ ] **Step 5: Commit** `git add final/robotnik_arm_task/experiments/ablation/<LAUNCH>__architecture-*.yaml && git commit -m "experiments: architecture ablation and lr-check manifests"`

### Task 6: Learning-rate check (validation only)

- [ ] **Step 1: Train and validate** each lr manifest: `../../../.venv/bin/python -m experiments.ablation.run --manifest experiments/ablation/<LAUNCH>__architecture-lr-check-<lr>-v1.yaml --stage train --parallel 2`, then the same with `--stage validate --workers 22`. Expected: 6 models, each with `validation/<name>__validation-student.csv` (32 rows).
- [ ] **Step 2: Apply the rule.** Count the sustained holds and call `lr_decision`. Write `results/ablation/<LAUNCH>__architecture-ablation-v1/lr_check.md` with the 2×3 table and the decision.
- [ ] **Step 3: If `"screen"`, stop and tell the user before training the grid.** The spec's fallback (a three-lr screen per architecture on R4 seed 0) changes the grid's recipe, and that needs a manifest revision. If `"keep"`, continue.

### Task 7: Run the grid and the tests

- [ ] **Step 1: Train R4 first.** `python -m experiments.ablation.run --manifest experiments/ablation/<LAUNCH>__architecture-ablation-v1.yaml --rows R4-dart-gate --stage train --parallel 2`. Watch `nvidia-smi` for the first two runs. If GPU memory is above 10 GiB, use `--parallel 1` for R6/R8. Expected: 27 models.
- [ ] **Step 2: While the GPU trains R6 and then R8** (same command with `--rows R6-gate-dagger-r3`, then `R8-dart-gate-dagger-r3`), evaluate R4 on CPU: `--rows R4-dart-gate --stage validate --workers 16`, then `--stage hold --workers 16`. Then run `python -m evaluation.test_suite --models <the 27 R4 step_42000.npz> --out results/test/<datetime>__arch-ablation-R4 --workers 16`.
- [ ] **Step 3: Repeat Step 2 for R6 and R8** as each row's training finishes, with output dirs `…__arch-ablation-R6` and `…__arch-ablation-R8`. Use `--resume <dir>` if a test run is interrupted. Expected: each `run.json` has `status: complete`, and its `scene_manifest_sha256` starts with `0a916445`.
- [ ] **Step 4: Validate and hold-test the references** (`--stage validate` and `--stage hold` without `--rows` skip everything already done). Expected: the sustained-hold counts are identical to the recorded ones: R4 from v1 `runs.csv` (`student_hold`, `arm_hold`), R6/R8 from their DAgger `summary.csv` round-3 rows. Report any difference to the user before Task 8.
- [ ] **Step 5: Latency on an idle machine**, after every simulation has finished: `python -m evaluation.latency_bench --models <seed-0 npz of each of the 10 architectures, R4> --out results/ablation/<LAUNCH>__architecture-ablation-v1/latency.csv`.
- [ ] **Step 6: Run** `--stage summary` for the grid manifest. Expected: `runs.csv` with 90 lines.

### Task 8: Report, docs and memory

- [ ] **Step 1: Run the report CLI** (Task 4) on the three new test dirs, the reference test dir and the latency CSV. Expected: `check_comparable` passes for all three runs, and the summary, figures and test-dir notes are written.
- [ ] **Step 2: Write `docs/reports/<date>__report__architecture-ablation.md`**, in English and in the existing reports' style. Cover the question, the setup (grid, fixed λ_J, the lr check outcome, data option A), the results table, figures, the recommendation per row, and limitations (DAgger states from the 256×3 student, 3 seeds, a single lr, terminal checkpoint, the T0 ceiling). Then give the literature context from the spec's references, and a ready-to-paste paragraph for the report's Results section.
- [ ] **Step 3: Add the experiment** to `docs/README.md`'s document list and to `final/robotnik_arm_task/results/ablation/<LAUNCH>__architecture-ablation-v1/README.md` (what each file is).
- [ ] **Step 4: Update memory** `comp0245-final-project.md` with one dated line: where the results live, the headline numbers, the recommendation and the lr decision.
- [ ] **Step 5: Commit** the report, the READMEs and the tracked summaries (`results/ablation/**/summary.*`, `architecture_summary.*`, `lr_check.md`, `README.md`). Stage these files by name only.
