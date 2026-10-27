# COMP0245 Final Project: Behaviour Cloning for the Robotnik Mobile Manipulator

A learned policy maps base pose (`base_pos`, `base_bearing`), arm state (`q_mes`, `qd_mes`) and `TARGET_EE_POS` to arm torques (`torque_arm`, 7) and wheel speeds (`wheel_cmds`, 4). The current training target is clean-data MLP behaviour cloning, followed by a separately reported damping-Jacobian loss experiment. DART and DAgger are later, separate workflows. Training and generic evaluation entry points remain placeholders; no learned policy is claimed here.

## Setup

Python 3.10. RoboEnv provides `simulation_and_control`, `pinocchio` and `pybullet`.

```bash
uv venv --python 3.10 ../.venv && source ../.venv/bin/activate
uv pip install -e RoboEnv
uv pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128   # torch 2.11.0+cu128, before the requirements
uv pip install -r requirements-dev.txt                                          # requirements.txt + pytest
```

Training uses PyTorch 2.11.0 built for CUDA 12.8, which includes the `sm_120` kernels the RTX 50-series laptop GPU needs. Check it with `python -c "from training.common.config import configure_runtime; print(configure_runtime('auto'))"` from `final/robotnik_arm_task/`: `device: cuda` means a kernel actually ran; otherwise `fallback_reason` explains why training will use the CPU. Without a usable GPU, install the CPU wheel from `https://download.pytorch.org/whl/cpu` instead. Installing torch 2.11.0 downgrades `setuptools` to 78.1.0 (a torch constraint); the simulator packages are unchanged.

Tests run from the repository root: `python -m pytest tests -m 'not sim and not realdata'` for the fast suite, and `-m realdata` / `-m sim` for checks that read the recorded datasets or start PyBullet.

## Running

Run every command from `final/robotnik_arm_task/`. Project scripts are run as modules (`python -m ...`) so they can import `core/`.

```bash
cd final/robotnik_arm_task

python robotnik_arm_task.py                                  # original task with GUI
ROBOENV_HEADLESS=1 python robotnik_arm_task.py               # original task, headless

python -m data_collection.scenarios                         # frozen Sobol scenarios  -> data/scenarios/
python -m data_collection.scenario_coverage                 # coverage audit          -> data/scenarios/coverage/
python -m data_collection.collect                           # clean expert data       -> data/{Markovian,Stateful}/Clean/
python -m data_collection.qa_clean                          # QA + training manifests -> data/*/Clean/{qa/,training_manifest.csv}
python -m data_collection.verify_template_teacher --out data/Stateful/Clean/qa   # wrapper vs original script
python markovian_expert_demo.py                              # Markovian expert with GUI (--target X Y Z, --start X Y, --bearing B)
python -m evaluation.compare_controllers --n 30              # template vs Markovian time to reach -> results/controller_timing.csv
```

Simulator and clean-collection settings are in `final/robotnik_arm_task/settings.yaml`. Its `mlp`, `train_stride` and `dagger` keys are legacy placeholders: the training code never reads them, and the file is left byte-identical because the clean-data QA recomputes a code digest over it. Training settings live in `experiments/configs/` (`common.yaml` → `baseline.yaml`/`jacobian.yaml` → `objectives/` → `teachers/` → `--set key=value`), as defined in the [training structure design](docs/superpowers/specs/2026-10-26-mlp-training-structure-design.md). The numerical recipe is in [training research](docs/mlp_training_research.md), and the exact clean split and evaluation predicate are in the [baseline plan](docs/mlp_baseline_training_plan.md). The clean datasets, outcome definitions and QA are described in `docs/clean_collection_report.md`.

### MLP training and evaluation

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

Runs are written to `checkpoints/<regime>/<teacher>/<objective>/<run_id>/` (`config.yaml`, `provenance.json`, `normaliser.npz`, `metrics.csv`, `step_<k>.pt` + `step_<k>.npz` exports, resumable `latest.pt`; `best.pt` + `inference.npz` once selected). `training.train_mlp` is a deprecated wrapper that forwards to `training.baseline.train` and rejects its old `--data/--out` arguments. `data_collection.dagger` remains a placeholder for a later stage.

### Clean MLP results (final protocol `p-e26487f2a548`)

Protocol chain: pilot/budget `p-0dedccad51dc` (LR 3e-4, 42,000 updates), loss `p-2a37f68e30af` (Huber), screen `p-9e89dc42395f`, seeds `p-a67f676c83f0` (selected A M6, B M1), calibration `p-13adaf7f1178` (λ_J 0.0703 A, 0.0671 B), final `p-e26487f2a548`. Refit was skipped; the declared artifacts are the validation-selected `best.pt` of each selected architecture, BC and BC+J, seeds 0–2.

| Teacher | Objective | Test sustained hold (64, seeds 0/1/2) | Test reach | Validation hold (32) |
|---|---|---|---|---|
| A (M6) | BC | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| A (M6) | BC+J | 0 / 1 / 1 | 2 / 3 / 8 | 0 / 0 / 0 |
| B (M1) | BC | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| B (M1) | BC+J | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |

Both experts hold 32/32 validation scenarios through the same evaluation path. No clean-data MLP controls the robot: all 12 screened architecture/teacher selections, the finalist seeds and the Clean_223 comparison hold 0/32 on validation. Offline errors are small (validation arm RMSE about 0.4–0.7 N·m) and expert rollouts with residual-matched i.i.d. noise still hold (A 32/32, B 29/32), so the failure comes from state-dependent errors, not their size. BC+J does its job offline: the learned damping median diagonal moves from about −8 to −29.9 (A) and −27.3 (B) N·m·s/rad, against the teachers' −30. The position stiffness ∂τ/∂q at the hold pose stays indefinite (largest symmetric eigenvalue +16.8 for A BC+J and +69.5 for A BC; teacher about −100 to −116), so the hold remains statically unstable. Remedies that act on this (state-perturbation data such as DART/DAgger, or ∂τ/∂q supervision) lie outside this stage.

## Layout

```
final/robotnik_arm_task/
├── robotnik_arm_task.py, robotnik_arm_control.py   course template (unmodified)
├── configs/, models/                               robot config and URDF the task loads
├── settings.yaml                                   simulator and clean-collection settings
├── core/             interfaces.py (Policy, Observation, Action), env.py, features.py
├── data_collection/  expert.py, scenarios.py, collect.py, dagger.py
├── training/         current flat placeholders; target common/, baseline/, jacobian/
├── evaluation/       shared evaluation placeholders and implemented expert diagnostics
├── results/          versioned summary CSV; detailed results and figures stay local
├── data/             datasets (git-ignored)
└── checkpoints/      model weights (git-ignored)
```

## Outputs

- `results/results.csv`: the compact shared evaluation index, permitted in Git. The target schema includes global run identity, config/split hashes, checkpoint step/path and evaluation split/ID; a training run may have multiple checkpoint evaluations. `evaluation.evaluate` and `experiments.run` append to it under a file lock and refuse duplicate (run, step, split, evaluation ID) keys.
- Per-run details and figures stay local under `results/<regime>/<teacher>/<objective>/<run_id>/`; current expert diagnostics may retain their existing result paths.
- Raw data, checkpoints, detailed results and figures are not committed. Model artifacts are recreated from the committed protocols with the `experiments.run` stages above; every run records its config, data, split and code hashes in `provenance.json`.

## External code

| Code | Source | Status |
| --- | --- | --- |
| `RoboEnv/`, `simulation_and_control` | COMP0245 course | Used as is |
| `robotnik_arm_task.py`, `robotnik_arm_control.py`, `configs/`, `models/` | COMP0245 final task template | Used as is |
| `data_collection/expert.py` | Based on `robotnik_arm_control.py` | Modified: made Markovian |
| `data_collection/template_teacher.py` | Calls `arm_controller`/`base_controller` from `robotnik_arm_control.py` unchanged | Written for this project: wrapper reproducing `robotnik_arm_task.py`'s setup, verified bit-identical |
| Everything else in `core/`, `data_collection/`, `training/`, `evaluation/` | Written for this project | From scratch |
