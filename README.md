# COMP0245 Final Project: Behaviour Cloning for the Robotnik Mobile Manipulator

A learned policy maps base pose (`base_pos`, `base_bearing`), arm state (`q_mes`, `qd_mes`) and `TARGET_EE_POS` to arm torques (`torque_arm`, 7) and wheel speeds (`wheel_cmds`, 4). The current training target is clean-data MLP behaviour cloning, followed by a separately reported damping-Jacobian loss experiment. DART and DAgger are later, separate workflows. Training and generic evaluation entry points remain placeholders; no learned policy is claimed here.

## Setup

Python 3.10. RoboEnv provides `simulation_and_control`, `pinocchio` and `pybullet`.

```bash
uv venv --python 3.10 ../.venv && source ../.venv/bin/activate
uv pip install -e RoboEnv
uv pip install -r requirements.txt
```

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

Simulator and clean-collection settings are in `final/robotnik_arm_task/settings.yaml`. Its legacy MLP/DAgger defaults are not the current training recipe. The target template/override rules and future commands are in the [training structure design](docs/superpowers/specs/2026-10-26-mlp-training-structure-design.md); the numerical recipe is in [training research](docs/mlp_training_research.md), and exact clean splits/evaluation are in the [baseline plan](docs/mlp_baseline_training_plan.md). New training templates and entry points are documentation targets, not runnable files yet. The existing clean datasets, outcome definitions and QA are described in `docs/clean_collection_report.md`.

The existing `training.train_mlp`, `data_collection.dagger`, `evaluation.evaluate` and `evaluation.plots` entry points raise `NotImplementedError`; their old command examples are not completed workflows.

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

- `results/results.csv`: the compact shared evaluation index, permitted in Git. The target schema includes global run identity, config/split hashes, checkpoint step/path and evaluation split/ID; a training run may have multiple checkpoint evaluations. The future shared evaluator will write this schema.
- Per-run details and figures stay local under `results/<regime>/<teacher>/<objective>/<run_id>/`; current expert diagnostics may retain their existing result paths.
- Raw data, checkpoints, detailed results and figures are not committed. Their future training/collection commands and provenance rules are defined in the structure design; the training placeholders cannot recreate model artifacts yet.

## External code

| Code | Source | Status |
| --- | --- | --- |
| `RoboEnv/`, `simulation_and_control` | COMP0245 course | Used as is |
| `robotnik_arm_task.py`, `robotnik_arm_control.py`, `configs/`, `models/` | COMP0245 final task template | Used as is |
| `data_collection/expert.py` | Based on `robotnik_arm_control.py` | Modified: made Markovian |
| `data_collection/template_teacher.py` | Calls `arm_controller`/`base_controller` from `robotnik_arm_control.py` unchanged | Written for this project: wrapper reproducing `robotnik_arm_task.py`'s setup, verified bit-identical |
| Everything else in `core/`, `data_collection/`, `training/`, `evaluation/` | Written for this project | From scratch |
