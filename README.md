# Robotnik mobile manipulation: assessor's run guide

This project learns to control a 7-DoF arm and a four-wheel mobile base from expert demonstrations. A student predicts arm torques and wheel speeds from measured robot state and the target position. Submitted methods are **Baseline, OU, DAgger and f-DAgger**, each without/with a reach-gate input (R1–R8), plus **full DART + Gate** (R9).

**To inspect submitted models:** complete [installation](#2-installation), read [what seeds mean](#3-before-choosing-a-model-what-do-the-seeds-mean), then use [live comparison](#5-live-comparison-expert-versus-your-chosen-model). No training or demonstration collection is needed. For numerical results, use [validation and testing](#6-validate-and-test-the-submitted-models). To reproduce training, follow [the full pipeline](#7-reproduce-the-data-and-training-pipeline).

## 1. Project framework and working directories

The **repository root** contains this README and `requirements.txt` (`COMP_0245_2026_Final/` in the surrounding workspace). The **task directory** is `final/robotnik_arm_task/` inside it. Commands use Bash on Linux or WSL2.

```text
COMP_0245_2026_Final/                 # repository root: install here
├── README.md
├── requirements.txt                 # pinned runtime dependencies
├── requirements-dev.txt             # runtime dependencies + pytest
├── tests/                           # unit, simulator and data checks
├── RoboEnv/                         # external simulator; install separately
├── docs/                            # method, protocol and reproduction references
├── Archive/                         # retired experiments and documents
└── final/robotnik_arm_task/          # task directory: run project modules here
    ├── core/                        # environment, interfaces, features, model names
    ├── data_collection/             # expert A/B, clean, OU, DAgger and DART collection
    ├── training/                    # datasets, MLP, normalisation, losses, training
    ├── evaluation/                  # validation, unified tests, diagnostics, live demo
    ├── experiments/                 # recipes, runners, frozen splits, demo manifests
    ├── configs/, models/            # simulator configuration and robot URDF/assets
    ├── tools/                       # data preparation and checkpoint export
    ├── data/                        # demonstrations and built feature datasets
    ├── checkpoints/2026-11-05final/  # submitted weights and model index
    ├── results/                     # evaluation tables, plots and rollout records
    └── report/                      # report evidence, figures and supporting code
```

```text
Frozen scenarios → expert trajectories → QA / Clean_253 split → feature datasets
OU-perturbed expert trajectories ──────────────────────────────┘       │
                                                                     ↓
                                                          BC + Jacobian training
                                                                     ↓
                                                               validation
                                                                     ↓
Student rollouts + expert labels → aggregate DAgger data → retrain / validate
                                                                     ↓
                                                        frozen final checkpoints
                                                                     ↓
                                                   live comparison / unified tests
```

The default network is `MLP01`: three 256-unit hidden layers, each using Linear → LayerNorm → SiLU. It predicts 7 arm torques and 2 wheel speeds (right/left), expanded to 4 wheel commands. Gate variants add the `reach_gate` feature for 22 inputs; other final models have 21. Features and normalisation are shared by training and inference. The control interface is 1 kHz, with limits of ±30 N·m and ±6 rad/s.

## 2. Installation

Use **Python 3.10** and **uv**. Native Windows Python is unsupported because the project uses `fcntl`; use WSL2. A desktop/WSLg is required for graphical live comparison. CPU inference and CPU training are supported.

Start in the **repository root**. If you opened the surrounding `ML_Final/` workspace, first run `cd COMP_0245_2026_Final`.

```bash
# Skip cloning if the supplied RoboEnv checkout is already present.
git clone https://github.com/VModugno/RoboEnv.git
git -C RoboEnv checkout 7bb49439677c2ac111347b40e997025ce87bff50
git -C RoboEnv submodule update --init

uv venv --python 3.10 .venv
source .venv/bin/activate

# CPU installation. For NVIDIA CUDA 12.8, replace /cpu with /cu128.
uv pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install -e RoboEnv -r requirements-dev.txt

cd final/robotnik_arm_task
```

On minimal Ubuntu/WSL2, install `libgomp1` if Pinocchio reports a missing OpenMP runtime (`sudo apt install libgomp1`). In each new terminal, activate the environment from the repository root with `source .venv/bin/activate`, then `cd final/robotnik_arm_task`.

**All commands below run from the task directory unless stated otherwise.** They work without the author's absolute paths and without Git in the submitted copy.

| Operation | Required project files, in addition to code and robot assets |
|---|---|
| Live comparison / validation | Complete `checkpoints/2026-11-05final/` and `experiments/splits/clean253.json`; no training data |
| Unified tests | The above, `experiments/splits/unified-test-v1.json`, and the source-inventory metadata/scenario CSVs described in section 6 |
| Retraining R1–R4 | Corresponding complete feature datasets under `data/datasets/A/` (section 7) |
| Rebuilding data | Raw demonstrations or fresh collection; allow about 25 GiB for the original clean/OU preparation pipeline, plus space for DAgger/DART |

Weights, datasets and most results are Git-ignored: a source-only clone does not include them. Preserve the checkpoint directory structure and each model's adjacent `config.yaml`.

## 3. Before choosing a model: what do the seeds mean?

`seed-0`, `seed-1` and `seed-2` identify **three separately trained repetitions of the same method and recipe**. The training seed determines random streams for network initialisation, training-batch sampling and sampling in the Jacobian loss. Repeated seeds measure sensitivity to training randomness. A larger seed is neither a stronger model nor a harder task.

BC+J loss calibration uses a fixed seed-0 initialisation and fitting batch, shared across seeds within a row. For DAgger, seed 0 drives the collection/validation chain; seeds 1/2 retrain on the selected round's aggregated data. These are not three independent collection chains. DART likewise uses a seed-0 iteration chain with additional training repetitions on the collected datasets.

| Seed/context | Meaning |
|---|---|
| Folder `seed-N`, filename `__sN__`, live `--seed N` | Selects an already-trained weight file; does not retrain or change the scene |
| Scenario-generation / environment seeds | Determine sampled target/start poses and recorded scenario setup; separate from model seeds |
| OU/DART collection noise seeds | Determine random demonstration perturbations; separate from model seeds |

To compare methods, keep the **same scenario** and inspect all three model seeds. A live demo shows one scene; use the test suite for aggregate performance. Fresh training across different hardware/software environments may produce different weights or success counts.

## 4. Submitted model choices and filenames

The release contains **27 final models**: R1–R9 × seeds 0, 1, 2. Every model is at terminal update 42,000. The table gives the exact live `--row` values; each accepts `--seed 0`, `1` or `2`.

| Row | Report name | `--row` value | Inputs | Released round |
|---|---|---|---:|---:|
| R1 | Baseline | `R1-Baseline` | 21 | 0 |
| R2 | Baseline + Gate | `R2-Baseline-Gate` | 22 | 0 |
| R3 | OU | `R3-OU` | 21 | 0 |
| R4 | OU + Gate | `R4-OU-Gate` | 22 | 0 |
| R5 | DAgger | `R5-DAgger` | 21 | 3 |
| R6 | DAgger + Gate | `R6-DAgger-Gate` | 22 | 3 |
| R7 | f-DAgger | `R7-f-DAgger` | 21 | 3 |
| R8 | f-DAgger + Gate | `R8-f-DAgger-Gate` | 22 | 3 |
| R9 | DART + Gate | `R9-DART-Gate` | 22 | 3 (noise scale 6) |

R3/R4 use fixed phase-gated OU perturbations; R7/R8 add DAgger to those data. R9 uses learner-fitted Gaussian DART. **Historical `dart` strings in R3/R4/R7/R8 paths mean OU, not full DART.** Gate means the extra input, independently of the phase gate used when injecting OU noise.

Browse files under `checkpoints/2026-11-05final/models/<row>/seed-<N>/`. For example:

```text
models/R8-f-DAgger-Gate/seed-0/
├── R8-f-DAgger-Gate-r3__MLP01-256x3__s0__v3__2026-11-05.npz
├── R8-f-DAgger-Gate-r3__MLP01-256x3__s0__v3__2026-11-05.pt
├── config.yaml
└── ... normaliser, metrics and provenance
```

`r3` is the aggregation round, `MLP01-256x3` the architecture, `s0` the training seed, `v3` the training version (R1–R4: `v1`; R9: `v2`), and the date the recorded training experiment. `.npz` is the NumPy inference export used by live comparison; `.pt` is the PyTorch checkpoint. These release files are for inference, not training resume states.

The [model index](final/robotnik_arm_task/checkpoints/2026-11-05final/model_index.csv) maps filenames, seeds, rounds, source runs and hashes. The release also preserves 22 additional models in `round_history/`; the supplied final demo manifest lists the 27 final models above. See the [release README](final/robotnik_arm_task/checkpoints/2026-11-05final/README.md) and [naming reference](docs/model-naming.md).

## 5. Live comparison: expert versus your chosen model

### Choose a method and seed

**Live comparison model names:** copy one of these exact names into `ROW`. Each name supports `SEED=0`, `SEED=1` and `SEED=2`, giving 27 choices in total. Read [the seed explanation](#3-before-choosing-a-model-what-do-the-seeds-mean) before selecting a repetition.

| Model shown in the report | Name to copy into `ROW` |
|---|---|
| Baseline | `R1-Baseline` |
| Baseline + Gate | `R2-Baseline-Gate` |
| OU | `R3-OU` |
| OU + Gate | `R4-OU-Gate` |
| DAgger | `R5-DAgger` |
| DAgger + Gate | `R6-DAgger-Gate` |
| f-DAgger | `R7-f-DAgger` |
| f-DAgger + Gate | `R8-f-DAgger-Gate` |
| DART + Gate | `R9-DART-Gate` |

For example, `ROW=R1-Baseline; SEED=0` selects Baseline seed 0; `ROW=R4-OU-Gate; SEED=1` selects OU + Gate seed 1; `ROW=R8-f-DAgger-Gate; SEED=2` selects f-DAgger + Gate seed 2. Set the variables, then run the command below again for each model. R8 seed 0 below is an example; any listed combination works.

```bash
RELEASE=checkpoints/2026-11-05final
ROW=R8-f-DAgger-Gate
SEED=0

python -m evaluation.live_compare \
    --protocol "$RELEASE/demo_models.yaml" \
    --teacher A --objective bcj --row "$ROW" --seed "$SEED" \
    --scenario train-0186 \
    --output "results/live_compare/${ROW}-s${SEED}-train-0186.json"
```

Two independent PyBullet windows open: **EXPERT A** and the **chosen student**. Both start with the same scenario and initial arm/base state, and synchronise every 20 physics steps. The student controls both base and arm. Overlays show model/seed, simulation time, end-effector error (mm), torque and wheel speeds.

Explicit `--protocol` selects the submitted release; omitting it selects the code's older demo manifest. This release contains only teacher A / objective `bcj`; teacher B or `bc` will not match its artifacts. The program checks the chosen weight's hash before starting. To switch models, quit and rerun with a different `ROW` or `SEED`.

| Key (focus either simulation window) | Action |
|---|---|
| Space | Pause/resume both simulations |
| R | Reset/replay both from the same initial state |
| F | Toggle follow camera in the focused window |
| Q | Close both and save the summary |

There is **no time limit** by default. Add `--seconds 40` to freeze both windows at 40 simulated seconds; R replays, Q quits. `--speed 2` requests twice normal playback speed while retaining 1 kHz physics. JSON is saved on exit; the same `--output` filename is overwritten on rerun, so use distinct names to retain comparisons.

### Mouse-drag disturbances: inspect robustness interactively

The graphical PyBullet windows support dragging the robot to apply a physical disturbance while its controller is running. This uses PyBullet's built-in mouse picking; no extra option is required. The [official PyBullet mouse-handling implementation](https://github.com/bulletphysics/bullet3/blob/master/examples/SharedMemory/PhysicsServerExample.cpp) uses an unmodified left-button press to pick and a release to stop dragging.

1. Start the graphical comparison above at normal speed (`--speed 1`, the default). Leave `--seconds` unset so the simulation continues while you interact. First observe the robot without intervention.
2. Focus the **student model's window**. Press **F** to turn off its follow camera if you want a fixed view while dragging. Ensure the overlay says **RUNNING**; Space resumes a paused run, and R restarts a run that reached its time limit.
3. Place the pointer on a visible solid part of the robot, such as the mobile base or an arm link. **Hold the left mouse button, drag a short distance, then release.** Do not hold Ctrl or Alt: those modifiers are used for camera controls. Pick the robot itself, rather than the floor or red target marker.
4. After release, observe whether the controller recovers: does the end effector return toward the target, does the displayed **EE error** fall below 10 mm again, and can the robot hold its position without persistent oscillation or drift? Try disturbances during motion and after reaching the target.
5. Press **R** to reset both simulations before another trial. Quit with Q, choose a different `ROW`/`SEED`, and rerun the same scenario to inspect another model. You can also drag the robot in the **EXPERT A** window to inspect the expert's response.

Only the robot in the window you drag receives the disturbance; synchronised simulation time does not copy mouse forces into the other window. For a useful qualitative comparison, keep the scene, model seed, camera view, selected body part and disturbance phase similar, and use approximately the same drag distance and duration. A large or sustained pull can prevent recovery; observe the response after releasing the mouse.

Mouse dragging is an exploratory robustness check. Its force and duration are not calibrated or replayed by this program. The saved live-comparison JSON contains final states/errors, not a mouse-force history or an automatic recovery score. Record your observations separately and use the frozen tests in section 6 for the reported task metrics. Mouse interaction is available only in GUI mode, not with `--headless`.

### Choose a scene or custom target

`--scenario` accepts one of the 32 frozen validation scenes. Their `train-...` IDs are historical names; they are held out from fitting. List valid IDs:

```bash
python - <<'PY'
from evaluation.test_sets import get_split
from pathlib import Path
for scene in get_split("validation", Path("experiments/splits/clean253.json"), "A"):
    print(scene.scenario_id)
PY
```

Custom target/base coordinates are in world-frame metres; bearing is yaw in radians (0 faces +X). This example is an interactive custom scene, not a frozen evaluation result:

```bash
python -m evaluation.live_compare \
    --protocol "$RELEASE/demo_models.yaml" --row "$ROW" --seed "$SEED" \
    --scenario train-0186 --target 2.0 0.5 0.7 --start 0.0 0.0 --bearing 0.0 \
    --seconds 40 --output "results/live_compare/${ROW}-s${SEED}-custom.json"
```

Normal collection ranges: target horizontal distance 0.5–3.0 m from the initial base, target height 0.1–1.25 m, start x/y within ±0.5 m. The program warns about custom scenes outside these ranges. The live 10 mm overlay is a visual reference; formal success is computed by the evaluators.

### No display: short installation check

```bash
python -m evaluation.live_compare \
    --protocol "$RELEASE/demo_models.yaml" --row "$ROW" --seed "$SEED" \
    --headless --seconds 0.1 \
    --output "results/live_compare/${ROW}-s${SEED}-smoke.json"
```

This runs both simulators for 100 steps and exits. A finite `--seconds` is needed for an automatically finishing headless check; 0 means unlimited. This checks model loading/simulation, not task success. Each JSON `runs` entry should have `failure: null`.

## 6. Validate and test the submitted models

### Resolve the selected model file

After setting `RELEASE`, `ROW` and `SEED` as in section 5, resolve the actual filename from the supplied manifest:

```bash
MODEL=$(python - "$RELEASE/demo_models.yaml" "$ROW" "$SEED" <<'PY'
import sys, yaml
artifacts = yaml.safe_load(open(sys.argv[1]))["decisions"]["final"]["artifacts"]
matches = [a for a in artifacts if a["row"] == sys.argv[2] and a["seed"] == int(sys.argv[3])]
assert len(matches) == 1, "Choose a listed row and seed 0/1/2"
print(matches[0]["inference"])
PY
)
printf '%s\n' "$MODEL"
```

### Validation: 32 held-out scenarios

```bash
python -m evaluation.evaluate --policy mlp --artifact "$MODEL" \
    --split validation --workers 4
```

The evaluator prints success/error statistics and writes CSVs under `results/ablation/A/bcj/<run_id>/`. Historical validation requires reaching below 10 mm after 8 s, running another 3 s, and staying below 10 mm throughout the last second (40 s cap). Existing detail files cannot be overwritten; use a fresh working copy to repeat identical validation. `--record` is omitted to preserve the historical results index.

### Unified testing: 240 frozen scenarios

Use **`--models`** to select submitted files explicitly. The default `--rows` lookup uses historical training directories and covers R1–R8; it is not the release-file selector.

For a new run, preserve the supplied `data/scenarios/scenarios_train.csv`, `data/Markovian/Clean_253/scenarios_test.csv`, all `data/datasets/A/*/metadata.json` files, and the `scenarios_train.csv` at each metadata file's `source.root`. The test runner reads these records to check training/test overlap; it does not read training `.npy` arrays or trajectory files. A copy containing only weights and split JSONs supports live comparison/validation, but cannot run this inventory audit.

**Use the existing frozen-test wrapper below for this submission.** Newer DART data add source inventories after the test manifest was frozen. The plain `evaluation.test_suite` new-run command refuses that changed audit, even though all 240 scenes are identical. The report's wrapper rechecks the current inventories, requires identical scenes and zero overlap, then uses the original frozen manifest. It passes through the same test arguments.

```bash
# Selected model + expert on the 64 original test scenes (T0).
python report/07_training_time_comparison/code/test_suite_frozen.py \
    --models "$MODEL" --suites T0 --workers 4

# Selected model + expert on all 240 scenes (T0–T5).
python report/07_training_time_comparison/code/test_suite_frozen.py \
    --models "$MODEL" --workers 4

# All 27 submitted models + expert on all 240 scenes: 6720 rollouts.
mapfile -t MODELS < <(python - <<'PY'
from pathlib import Path
import yaml
p = Path("checkpoints/2026-11-05final/demo_models.yaml")
for a in yaml.safe_load(p.read_text())["decisions"]["final"]["artifacts"]:
    print(a["inference"])
PY
)
python report/07_training_time_comparison/code/test_suite_frozen.py \
    --models "${MODELS[@]}" --workers 4
```

These are alternative scopes; a full run is not required for a quick inspection. Add `--dry-run` to prepare/check without formal rollouts. New runs create timestamped directories under `results/test/`. Use the printed directory to resume or regenerate plots (replace the placeholder below):

```bash
RUN_DIR=results/test/REPLACE_WITH_PRINTED_DIRECTORY
python -m evaluation.test_suite --resume "$RUN_DIR" --workers 4
python -m evaluation.test_suite --plots-only "$RUN_DIR"
```

| Suite | Scenes | Purpose |
|---|---:|---|
| T0 | 64 | Original in-range test scenes |
| T1 | 72 | Far targets: 3–6, 6–10, 10–15 m |
| T2 | 24 | Near targets: 0.3–0.5 m |
| T3 | 32 | Height edges: 0.05–0.1, 1.25–1.30 m |
| T4 | 24 | Paired translations of T0 with shifted starting positions |
| T5 | 24 | New in-range scenes, 20 s continuous hold |

**Unified-test success:** error strictly below 10 mm continuously for 3 s (20 s in T5), counted from t=0. Completion time includes the hold. Final precision is evaluated separately using maximum error in the last second of the fixed-duration rollout. This differs from historical validation/test diagnosis; compare results within the same protocol.

Read `summary.md`/`summary.csv` for aggregates, `rollouts.csv` for individual outcomes, `plots/` for figures, and `run.json` for model/config hashes and run metadata. The expert runs once per scene. The [test protocol reference](docs/2026-11-01__guide__unified-test-suite.md) explains timings, precision grades and resume checks. Existing report evidence is indexed in [`report/`](final/robotnik_arm_task/report/README.md).

## 7. Reproduce the data and training pipeline

Use a fresh working copy for collection/retraining: existing datasets/run IDs can cause overwrite refusals or cause runners to skip completed work. Keep frozen splits unchanged. These commands reproduce the recorded recipes; fresh training may differ numerically across machines. Sections 5–6 only need inference weights.

### Step 1: collect clean demonstrations and build subsets

```bash
python -m data_collection.scenarios
python -m data_collection.collect --teachers markovian stateful --workers 4
python -m data_collection.qa_clean --teachers markovian stateful --workers 4
python -m data_collection.build_clean_subsets
```

Outputs: scenario CSVs in `data/scenarios/`, trajectories/QA in `data/Markovian/` and `data/Stateful/`, and paired `Clean_253`/`Clean_223` subsets. Teacher A (Markovian) supplies the final students; teacher B is collected because the paired builder checks both teachers. If local B episodes differ from the historical ones and the builder refuses, use `python tools/build_a_clean253.py` to build A's subset alone against the frozen split.

`Clean_253` has 221 fitting and 32 validation scenes. The 64 original test scenes are separate. Keep validation/test trajectories out of fitting and augmentation collection.

### Step 2: collect fixed OU perturbations and build features

```bash
# Historical module/protocol names say dart; this is the fixed OU collection.
python -m data_collection.collect_dart \
    --protocol experiments/configs/dart/d2-s075-x2.yaml --workers 4
```

The collector prints a directory under `data/Markovian/DART/`. Set `OU_DIR` to that exact directory; the block below selects it automatically only if exactly one match exists:

```bash
OU_DIR=$(python - <<'PY'
from pathlib import Path
paths = list(Path("data/Markovian/DART").glob("*__collection__d2-s075-x2-*"))
assert len(paths) == 1, "Set OU_DIR manually to the printed collection directory"
print(paths[0])
PY
)

for FEATURES in features-v1 features-v1-gate; do
    python -m training.datasets.build --source clean253 --features "$FEATURES" --workers 4
    python -m training.datasets.build --source "$OU_DIR" --name dart-d2s075x2 \
        --features "$FEATURES" --workers 4
done
```

Outputs: `data/datasets/A/{clean253,dart-d2s075x2}__{features-v1,features-v1-gate}/` (four datasets). The published OU dataset contains 441 accepted trajectories from 221 fitting scenes × 2 draws. Preserve metadata/checksums with the arrays. To retrain from shared data, restore these directories and skip collection. R4 alone needs the two `features-v1-gate` datasets (about 1.03 GiB). The [handoff reference](docs/2026-10-29__guide__reproduction-details.md#reproduction-and-data-handoff) lists exact files.

### Step 3: train R1–R4 (clean/OU × without/with Gate)

For one R4 seed-0 run:

```bash
python -m training.datasets.train --variant R4-dart-gate \
    --set identity.version=v1 --set identity.date=2026-10-28 --set identity.seed=0 \
    --set identity.objective=bcj --set training.lr=1.0e-3 --set training.updates=42000 \
    --set loss.action=huber --set loss.arm_weight=0.5 --set loss.wheel_weight=0.5
```

Add `--dry-run` to check data/calibration without optimisation (datasets are still required), or `--resume` to continue with training resume states. Outputs: checkpoints, exports, configs and metrics in `checkpoints/ablation/2026-10-28__training-ablation-v1/<run_id>/`.

To train all four rows × seeds 0/1/2, use the frozen manifest instead of the single-model command:

```bash
python -m experiments.ablation.run \
    --manifest experiments/ablation/2026-10-28__training-ablation-v1.yaml \
    --stage train --parallel 1
```

Historical aliases are deliberate: the following round runners expect these original base-model run IDs. The single-model trainer also supports canonical names, but those create different run IDs and do not populate the runners' historical default paths.

### Step 4: DAgger rounds for R5–R8

Each row starts from the corresponding v1 seed-0 base: R5 ← R1, R6 ← R2, R7 ← R3, R8 ← R4. Ensure step 3's base models exist. The runner collects student-visited states with expert A labels, builds features, aggregates all rounds, trains from scratch and validates.

```bash
for ROW in R5-baseline-dagger R6-gate-dagger R7-dart-dagger R8-dart-gate-dagger; do
    python -m experiments.dagger_rounds --row "$ROW" --rounds 3 \
        --date 2026-11-05 --final-seeds 1 2 --workers 4 \
        --version v3 --variant-suffix=-nodev --dev-scenes 0
done
# Optional: seeds 1/2 of rounds 1-2 for R6/R8 (the per-round curves of the training-time comparison)
python -m experiments.dagger_extra_seeds --date 2026-11-05 --rows R8-dart-gate-dagger R6-gate-dagger \
    --rounds 1 2 --seeds 1 2 --workers 4 --version v3 --variant-suffix=-nodev --source-date 2026-11-05 --dev-scenes 0
```

Every round rolls the student out on the 221 fitting scenarios only (`--dev-scenes 0`; write `--variant-suffix=-nodev` with `=`). This is ablation v3, which the submission uses. It replaces v2, whose rounds also collected 64 low/near-biased development scenes; see [`report/08_dev_scene_removal/`](final/robotnik_arm_task/report/08_dev_scene_removal/README.md). Without the three v3 options the runner reproduces v2. Round selection uses seed-0 validation hold, with ties choosing the later round; seeds 1/2 train on the selected data. Submitted R5–R8 all use round 3. Models: `checkpoints/ablation/2026-11-05__training-ablation-v3/`. Logs, per-round validation and `summary.csv`: `results/ablation/2026-11-05__training-ablation-v3/dagger/<row>/`.

### Step 5: full DART comparison (R9)

This separate pipeline starts from step 3's R2 seed-0 model. It estimates learner/expert action-error covariance on held-out demonstrations, collects Gaussian-perturbed expert data, aggregates and retrains. Released R9 uses noise scale 6, iteration 3, seeds 0/1/2:

```bash
python -m experiments.dart_full --date 2026-11-04 --scale 6 \
    --iterations 3 --extra-seeds 1 2 --workers 4
```

Models: `checkpoints/ablation/2026-11-04__training-ablation-v2/R9-dart-full-a6-r*/`. Covariance, validation, timing and logs: `results/ablation/2026-11-04__dart-full/`. Scale-1 exploratory weights are excluded from the release. See the [DART experiment record](final/robotnik_arm_task/results/ablation/2026-11-04__dart-full/README.md).

### Step 6: evaluate newly trained models

Set `MODEL` to the new run's actual `.npz`, then use section 6's validation/test commands. Choose models using validation before testing. For live comparison of retrained weights, generate a matching demo manifest: the submitted manifest pins original hashes. The [detailed reproduction reference](docs/2026-10-29__guide__reproduction-details.md) covers the v1 runner's `validate`, `hold`, `summary` and `demo` stages.

## 8. Troubleshooting and further reading

| Symptom | Action |
|---|---|
| `No module named ...` | Activate the environment, run from the task directory, install RoboEnv/pinned requirements |
| Missing checkpoint / no matching artifact | Restore the release and select its manifest, a listed row, seed 0/1/2, teacher A, objective `bcj` |
| Artifact hash differs | Restore the original submitted file; retrained weights require a matching new manifest |
| No display | Use WSLg/a desktop, or `--headless --seconds 0.1` for an installation check |
| Output already exists | Use a fresh result directory/working copy; resume unified runs with `--resume` |
| Test suite looks for historical weights | Use `--models` with final release paths as shown in section 6 |
| `frozen manifest differs; refusing overwrite` | Use the supplied frozen-test wrapper from section 6; retain the original split and source-inventory records |

From the **repository root**, optional installation checks are `python -m pytest tests -m "not sim and not realdata"` and `python -m pytest tests -m sim`. `realdata` tests additionally require their original datasets.

For methods and evidence, start with the [report index](final/robotnik_arm_task/report/README.md). See also the [pipeline](docs/2026-10-30__guide__project-pipeline.md), [documentation index](docs/README.md), [model naming guide](docs/model-naming.md) and [detailed reproduction reference](docs/2026-10-29__guide__reproduction-details.md). Dated documents retain historical context; use this README and the final manifest for submitted-model commands.

## External code

| Code | Source | Status |
|---|---|---|
| `RoboEnv/`, `simulation_and_control` | COMP0245 course | Used as is |
| `robotnik_arm_task.py`, `robotnik_arm_control.py`, `configs/`, `models/` | COMP0245 final task template | Used as is |
| `data_collection/expert.py` | Based on `robotnik_arm_control.py` | Modified: made Markovian |
| `data_collection/template_teacher.py` | Calls template `arm_controller`/`base_controller` unchanged | Project wrapper reproducing the template setup, verified bit-identical |
| Other code in `core/`, `data_collection/`, `training/`, `evaluation/`, `experiments/`, `tools/` | Written for this project | From scratch |
