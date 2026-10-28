# COMP0245 Final Project: Behaviour Cloning for the Robotnik Mobile Manipulator

文件导航：[按时间和类型查看工作区结构](docs/indexes/2026-10-27__index__workspace-structure.md) · [文档目录](docs/README.md)。

A learned policy maps base pose (`base_pos`, `base_bearing`), arm state (`q_mes`, `qd_mes`) and `TARGET_EE_POS` to arm torques (`torque_arm`, 7) and wheel speeds (`wheel_cmds`, 4). The current training target is clean-data MLP behaviour cloning, followed by a separately reported damping-Jacobian loss experiment. DART and DAgger are later, separate workflows. Training and generic evaluation entry points remain placeholders; no learned policy is claimed here.

## Setup

Supported: Linux and Windows through WSL2 (Ubuntu). The code uses `fcntl` file locks, so native Windows Python is not supported. Python 3.10 and [uv](https://docs.astral.sh/uv/getting-started/installation/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`). Run everything below from the repository root.

```bash
# 1. RoboEnv (course simulator: simulation_and_control), pinned to the commit the data was collected with.
#    RoboEnv/ is git-ignored; clone it inside the repository root.
git clone https://github.com/VModugno/RoboEnv.git
git -C RoboEnv checkout 7bb49439677c2ac111347b40e997025ce87bff50
git -C RoboEnv submodule update --init        # simulation_and_control is a submodule (pinned at df58505)

# 2. Environment (.venv/ is git-ignored)
uv venv --python 3.10 .venv && source .venv/bin/activate

# 3. PyTorch 2.11.0 before the requirements: pick ONE line
uv pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128   # NVIDIA GPU (driver with CUDA >= 12.8)
uv pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu     # no NVIDIA GPU

# 4. RoboEnv and the pinned requirements (requirements.txt + pytest), resolved together
uv pip install -e RoboEnv -r requirements-dev.txt
```

Every new terminal needs `source .venv/bin/activate` from the repository root.

Check the install from the repository root:

```bash
python -m pytest tests -m "not sim and not realdata"   # fast suite, no data or simulator needed
python -m pytest tests -m sim                          # starts PyBullet headless
cd final/robotnik_arm_task && ROBOENV_HEADLESS=1 timeout 20 python robotnik_arm_task.py; cd ../..   # original task runs (stopped after 20 s)
cd final/robotnik_arm_task && python -c "from training.common.config import configure_runtime; print(configure_runtime('auto'))"; cd ../..
```

The last command reports the training device: `device: cuda` means a CUDA kernel actually ran; otherwise `fallback_reason` explains why training uses the CPU. The cu128 build includes the `sm_120` kernels that RTX 50-series GPUs need; small MLPs also train on the CPU. Installing torch 2.11.0 downgrades `setuptools` to 78.1.0 (a torch constraint); the simulator packages are unchanged. `-m realdata` tests need the datasets from [Data](#data).

**WSL2 notes.** Clone the repository into the Linux file system (`~/...`), not under `/mnt/c/`: the datasets are thousands of files and Windows-mounted drives are much slower. GPU training needs only the normal Windows NVIDIA driver (do not install a Linux driver inside WSL); `nvidia-smi` inside WSL should list the GPU. PyBullet GUI windows (`robotnik_arm_task.py`, `markovian_expert_demo.py`, `evaluation.live_compare`) need WSLg (Windows 11, or Windows 10 build 19044+ with `wsl --update`); without it, use `ROBOENV_HEADLESS=1` or `--headless`.

## Data

Datasets, feature caches and checkpoints are git-ignored (several GB). Rebuild them locally; simulation is deterministic, and the frozen split in `experiments/splits/clean253.json` is versioned. From `final/robotnik_arm_task/`:

```bash
python -m data_collection.scenarios                         # frozen Sobol scenarios   -> data/scenarios/
python -m data_collection.collect --workers N               # clean expert data (both teachers) -> data/{Markovian,Stateful}/Clean/
python -m data_collection.qa_clean --workers N              # QA + training manifests  -> data/*/Clean/{qa/,training_manifest.csv}
python -m data_collection.build_clean_subsets               # paired subsets           -> data/*/Clean_253/, data/*/Clean_223/
python -m pytest ../../tests -m realdata                    # optional: loads both subsets and builds the feature caches
```

Set `N` to your CPU core count (default 22). Collection took about 4 minutes on 22 workers. Expect about 15 GB of disk in total: the subset folders are byte-identical copies of episodes in `Clean/`, and the feature caches add about 3 GB. `build_clean_subsets` asserts the expected inventory (253 paired successes, 223 without hold chatter, excluded `train-0064`, `train-0131`, `train-0183`). If it fails on another machine, the simulation there did not reproduce the recorded episodes; report it rather than editing the assertions.

## Running

Run every command from `final/robotnik_arm_task/`. Project scripts are run as modules (`python -m ...`) so they can import `core/`.

```bash
cd final/robotnik_arm_task

python robotnik_arm_task.py                                  # original task with GUI
ROBOENV_HEADLESS=1 python robotnik_arm_task.py               # original task, headless

python -m data_collection.scenario_coverage                 # coverage audit          -> data/scenarios/coverage/
python -m data_collection.verify_template_teacher --out data/Stateful/Clean/qa   # wrapper vs original script
python markovian_expert_demo.py                              # Markovian expert with GUI (--target X Y Z, --start X Y, --bearing B)
python -m evaluation.compare_controllers --n 30              # template vs Markovian time to reach -> results/controller_timing.csv
```

Simulator and clean-collection settings are in `final/robotnik_arm_task/settings.yaml`. Its `mlp`, `train_stride` and `dagger` keys are legacy placeholders: the training code never reads them, and the file is left byte-identical because the clean-data QA recomputes a code digest over it. Training settings live in `experiments/configs/` (`common.yaml` → `baseline.yaml`/`jacobian.yaml` → `objectives/` → `teachers/` → `--set key=value`), as defined in the [training structure design](docs/specs/2026-10-26__design__mlp-training-structure.md). The numerical recipe is in [training research](docs/research/2026-10-26__research__mlp-training-recipe.md), and the exact clean split and evaluation predicate are in the [baseline plan](docs/plans/2026-10-26__plan__mlp-baseline-training.md). The clean datasets, outcome definitions and QA are described in `docs/reports/2026-10-26__report__clean-expert-data-collection.md`.

### MLP training and evaluation

New training configurations from 2026-10-27 use `loss.arm_weight=0.8` and `loss.wheel_weight=0.2` for the two
standardised action-group means. Both are explicit CLI/configuration fields and enter the run identity.
Historical checkpoints and frozen protocols without these fields retain their original 0.5/0.5 meaning;
use `--set loss.arm_weight=0.5 --set loss.wheel_weight=0.5` for a new equal-weight run.
BC+J calibration records must match the action weights; changing them requires a matching new calibration.

```bash
cd final/robotnik_arm_task
python -m training.common.dataset --write-split              # experiments/splits/clean253.json (already versioned)

# Clean plain BC (action loss only). Budget, LR and loss come from the pilot protocol; --dry-run validates
# the sources, builds the feature caches and prints the run identity without training.
python -m training.baseline.train --teacher A --set training.lr=3.0e-4 --set training.updates=N --set loss.action=huber
python -m training.baseline.train --teacher A ... --resume                 # continue the identical run from latest.pt
python -m training.baseline.train --teacher A ... --sanity-episode train-0001   # one-episode overfit diagnostic

# Closed-loop validation (8 screening or 32 validation scenarios; test scenarios are sealed)
python -m evaluation.evaluate --policy mlp --artifact checkpoints/clean/A/bc/<run_id>/step_<k>.npz --split screening --record
python -m evaluation.evaluate --policy expert --teacher A --split validation --record    # teacher reference
```

For a live physical comparison, open two independent PyBullet windows on the same validation scenario. The models come
from [`experiments/demo_models/2026-10-28__training-ablation-v1.yaml`](final/robotnik_arm_task/experiments/demo_models/2026-10-28__training-ablation-v1.yaml),
which pins every terminal model of the training-method ablation by hash. It is written by
`experiments.ablation.run --stage demo`. The earlier clean and reweighted demo models are archived (see [Archive/README.md](Archive/README.md)).

```bash
python -m evaluation.live_compare                          # teacher A vs R4-dart-gate seed 0 on train-0186
python -m evaluation.live_compare --row R1-baseline        # the same scenario with the baseline row
python -m evaluation.live_compare --seed 2 --scenario train-0245          # another seed and validation scenario
python -m evaluation.live_compare --target 0.2 1.1 0.5      # custom world-frame target (meters)
```

The default scenario `train-0186` puts the target 2.7 m away, behind the robot (it must turn, drive and reach). All three
R4 seeds hold it on validation.

Both simulations keep the original 1 kHz control/physics interface and synchronize every 20 steps.
Each window shows its controller, simulation time, true end-effector error, torque magnitude and wheel commands;
the red sphere is the target and the red line joins it to the end effector. SPACE pauses/resumes both, R resets
both to the identical start, Q closes both, and F toggles camera following in the focused window (mouse orbit/zoom
remains available). After 40 simulated seconds the scene freezes until R or Q. Use `--speed 0.5` for slow motion,
`--scenario <validation ID>` to select another frozen validation scene, or `--headless --seconds 0.1` for a two-simulator smoke
check. `--target X Y Z` replaces the target in both simulators while retaining the selected scenario's initial
base pose and arm configuration. The run summary is saved to `results/live_compare/comparison.json`;
demo runs do not enter the evaluation index.

Experiment stages run through `experiments.run`, which calls the same launchers and evaluator and records decisions in immutable `experiments/protocols/p-<hash>.yaml` files:

```bash
python -m experiments.run --stage pilot --dry-run                     # 2 constant-LR M2-A Huber pilots, 50k updates
python -m experiments.run --stage pilot                               # train them (offline curves every 1k)
python -m experiments.run --freeze budget.updates=N --freeze budget.pilot_lr=3.0e-4 --evidence <pilot metrics.csv> ...
python -m experiments.run --stage loss --protocol experiments/protocols/<id>.yaml --select   # Huber vs MSE
python -m experiments.run --stage screen --protocol ... [--select]   # 36 fits; LR chosen offline per architecture
python -m experiments.run --stage seeds|jacobian|compare223 --protocol ... --select
python -m training.jacobian.train --teacher A --set identity.architecture=<selected> --set loss.action=<frozen> --calibrate
```

A stage whose decisions are missing stops with `BlockedError` instead of choosing a default. The test stage needs the frozen final protocol.

Final stages: `--stage refit` trains declared finalists afresh on all 253 training episodes (new fitting-only scales, frozen λ_J and update count) and saves `final.pt` + `inference.npz` with no validation selection; selection runs keep `best.pt`. `--stage test` evaluates only the artifacts listed in a frozen `decisions.final` protocol (architecture, LR, budget, loss, seed and checkpoint hashes are checked against the earlier decisions) on the 64 sealed test scenarios, once each. `python -m evaluation.plots` writes summaries and figures from `results/results.csv` to the local `results/figures/`.

Checks: `python -m pytest tests -m "not sim and not realdata"` (fast), `-m realdata` (reads the Clean_253/223 inventories and builds caches), `-m sim` (expert preflight through the student action path in PyBullet).

Runs are written to `checkpoints/<regime>/<teacher>/<objective>/<run_id>/` (`config.yaml`, `provenance.json`, `normaliser.npz`, `metrics.csv`, `step_<k>.pt` + `step_<k>.npz` exports, resumable `latest.pt`; `best.pt` + `inference.npz` once selected). The clean-pipeline checkpoints themselves are archived under `Archive/final/robotnik_arm_task/checkpoints/`, as are the deprecated `training.train_mlp` wrapper and the DAgger A code (see [Archive/README.md](Archive/README.md)).

### Clean MLP results (final protocol `p-e26487f2a548`)

Protocol chain: pilot/budget `p-0dedccad51dc` (LR 3e-4, 42,000 updates), loss `p-2a37f68e30af` (Huber), screen `p-9e89dc42395f`, seeds `p-a67f676c83f0` (selected A M6, B M1), calibration `p-13adaf7f1178` (λ_J 0.0703 A, 0.0671 B), final `p-e26487f2a548`. Refit was skipped; the declared artifacts are the validation-selected `best.pt` of each selected architecture, BC and BC+J, seeds 0–2.

| Teacher | Objective | Test sustained hold (64, seeds 0/1/2) | Test reach | Validation hold (32) |
|---|---|---|---|---|
| A (M6) | BC | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| A (M6) | BC+J | 0 / 1 / 1 | 2 / 3 / 8 | 0 / 0 / 0 |
| B (M1) | BC | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| B (M1) | BC+J | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |

Both experts hold 32/32 validation scenarios through the same evaluation path. No clean-data MLP controls the robot: all 12 screened architecture/teacher selections, the finalist seeds and the Clean_223 comparison hold 0/32 on validation. Offline errors are small (validation arm RMSE about 0.4–0.7 N·m) and expert rollouts with residual-matched i.i.d. noise still hold (A 32/32, B 29/32), so the failure comes from state-dependent errors, not their size. BC+J does its job offline: the learned damping median diagonal moves from about −8 to −29.9 (A) and −27.3 (B) N·m·s/rad, against the teachers' −30. The position stiffness ∂τ/∂q at the hold pose stays indefinite (largest symmetric eigenvalue +16.8 for A BC+J and +69.5 for A BC; teacher about −100 to −116), so the hold remains statically unstable. Remedies that act on this (state-perturbation data such as DART/DAgger, or ∂τ/∂q supervision) lie outside this stage.

### Gate and DART datasets

Two additions for teacher A, both independent of the clean pipeline above:

- **Feature definitions** (`core/feature_sets.py`): `features-v1` (21 inputs, unchanged) and `features-v1-gate` (22 inputs: v1 plus `reach_gate`, teacher A's reach-active rule recomputed from deployable inputs). Every exported model records its definition, so `MLPPolicy` builds the same inputs at deployment.
- **DART collection** (`data_collection/collect_dart.py`): the clean episodes with gated OU noise on the executed action, while the labels stay the clean expert action. Each noise protocol is one file in `experiments/configs/dart/`.

Raw episodes are turned into training-ready datasets in `data/datasets/A/<source>__<definition>/` (features, labels, row offsets, episode list, QA and README; git-ignored):

```bash
cd final/robotnik_arm_task
python -m data_collection.collect_dart --protocol experiments/configs/dart/d2-s075-x2.yaml   # -> data/Markovian/DART/<collection>/
python -m training.datasets.build --source clean253 --features features-v1                  # build v1 before gate:
python -m training.datasets.build --source clean253 --features features-v1-gate             # QA compares their first 21 columns
python -m training.datasets.build --source data/Markovian/DART/<collection> --name dart-d2s075x2 --features features-v1
python -m training.datasets.build --source data/Markovian/DART/<collection> --name dart-d2s075x2 --features features-v1-gate

# One ablation row per file in experiments/configs/datasets/variants/: R1-baseline, R2-gate, R3-dart, R4-dart-gate
python -m training.datasets.train --variant R4-dart-gate --set identity.version=v1 --set identity.date=2026-10-28 \
    --set training.lr=1.0e-3 --set training.updates=42000 --set loss.action=huber \
    --set loss.arm_weight=0.5 --set loss.wheel_weight=0.5 [--set identity.seed=1] [--dry-run]
```

Training uses the frozen split: clean datasets keep only fitting scenarios, DART datasets hold fitting scenarios only (checked), and the 32 validation scenarios come from the clean dataset. Architectures use a new registry, `MLP<NN>` numbered in order of first use (`MLP01` = 256×256×256), in place of the clean pipeline's m1–m6. Runs are named `<row>__<architecture>-<shape>__s<seed>__<version>__<date>` and stored in `checkpoints/ablation/<date>__training-ablation-<version>/`, for example `checkpoints/ablation/2026-10-28__training-ablation-v1/R4-dart-gate__MLP01-256x3__s0__v1__2026-10-28/`. The terminal model is also saved there as `<run name>.npz` and `<run name>.pt`, and it loads in `evaluation.evaluate` like any other MLP. Raise the version whenever the recipe, data or features change; reusing a name with a different configuration is refused.

A whole ablation runs from one manifest, which fixes rows, seeds, recipe and evaluation rules before any result is seen:

```bash
python -m experiments.ablation.run --manifest experiments/ablation/2026-10-28__training-ablation-v1.yaml \
    [--stage train|validate|hold|summary|all] [--parallel 3] [--workers 22]
python -m evaluation.hold_test --artifact <model.npz>     # the hold test for one model
```

Each stage skips finished work, so an interrupted run continues where it stopped. `train` runs `training.datasets.train` for every (row, seed). `validate` evaluates each terminal model on the 32 validation scenarios twice: the full student (the primary metric), and the student arm with teacher A driving the base; the rows are appended to `results/results.csv` with evaluation IDs `rollout-v1` and `rollout-v1-arm`. `hold` runs `evaluation.hold_test`: teacher A reaches and holds a target, the student takes over, and the arm closed loop is linearised. `summary` writes everything to `results/ablation/<date>__training-ablation-<version>/`: `runs.csv` (one row per model), `summary.csv` and `summary.md` (per ablation row: per-seed values and a pooled Wilson 95% interval), `validation/`, `hold_test/` and `logs/`.

## Layout

```
COMP_0245_2026_Final/
├── final/robotnik_arm_task/
│   ├── robotnik_arm_task.py, robotnik_arm_control.py   course template (unmodified)
│   ├── configs/, models/                               robot config and URDF the task loads
│   ├── settings.yaml                                   simulator and clean-collection settings
│   ├── core/              features: interfaces.py, env.py, features.py (v1), feature_sets.py (definitions, reach gate)
│   ├── data_collection/   collection: expert.py (teacher A), template_teacher.py (B), scenarios.py,
│   │                      collect.py (clean), collect_dart.py (DART), QA and subset builders
│   ├── training/          common/ (model, engine, policy), baseline/ + jacobian/ (losses, clean launchers),
│   │                      datasets/ (build.py: raw -> training-ready datasets; train.py: ablation training)
│   ├── evaluation/        evaluate.py (closed loop), offline.py, test_sets.py, plots.py, live_compare.py, expert diagnostics
│   ├── experiments/       configs/ (common, objectives, teachers, dart/, datasets/), splits/, protocols/, run.py
│   ├── data/              git-ignored: scenarios/, Markovian/{Clean,Clean_253,DART,…}/ (raw), datasets/A/ (training-ready)
│   ├── checkpoints/       git-ignored: ablation/<date>__training-ablation-<version>/<run>/
│   └── results/           results.csv (versioned index); per-run and per-experiment details stay local
├── tests/                 pytest suite (pytest.ini: testpaths = tests)
├── docs/                  plans, specs, reports, research, indexes
└── Archive/               retired code (tracked) and pre-ablation models (git-ignored); see Archive/README.md
```

Where the splits live: training = the 221 `fitting_ids` and validation = the 32 `validation_ids` of `experiments/splits/clean253.json` (their rows are in `data/datasets/A/clean253__*`; DART datasets hold training scenarios only); test = the 64 sealed scenarios in `data/Markovian/Clean_253/scenarios_test.csv`, evaluated in closed loop only.

## Outputs

- `results/results.csv`: the compact shared evaluation index, permitted in Git. The target schema includes global run identity, config/split hashes, checkpoint step/path and evaluation split/ID; a training run may have multiple checkpoint evaluations. `evaluation.evaluate` and `experiments.run` append to it under a file lock and refuse duplicate (run, step, split, evaluation ID) keys.
- Per-run details and figures stay local under `results/<regime>/<teacher>/<objective>/<run_id>/`; current expert diagnostics may retain their existing result paths. Ablation experiments keep all their results together in `results/ablation/<date>__training-ablation-<version>/`.
- Raw data, checkpoints, detailed results and figures are not committed. Model artifacts are recreated from the committed protocols with the `experiments.run` stages above; every run records its config, data, split and code hashes in `provenance.json`.

## External code

| Code | Source | Status |
| --- | --- | --- |
| `RoboEnv/`, `simulation_and_control` | COMP0245 course | Used as is |
| `robotnik_arm_task.py`, `robotnik_arm_control.py`, `configs/`, `models/` | COMP0245 final task template | Used as is |
| `data_collection/expert.py` | Based on `robotnik_arm_control.py` | Modified: made Markovian |
| `data_collection/template_teacher.py` | Calls `arm_controller`/`base_controller` from `robotnik_arm_control.py` unchanged | Written for this project: wrapper reproducing `robotnik_arm_task.py`'s setup, verified bit-identical |
| Everything else in `core/`, `data_collection/`, `training/`, `evaluation/` | Written for this project | From scratch |
