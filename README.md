# Robotnik: collect, train and test

This tutorial reproduces **R4 (DART + reach gate + BC+J), seed 0**: an MLP that controls the robot's arm and mobile base.
Use Linux or Windows WSL2, Python 3.10 and [uv](https://docs.astral.sh/uv/getting-started/installation/). Allow about 25 GiB of disk space for the full collection pipeline.

## 1. Install

Start from the repository root:

```bash
git clone https://github.com/VModugno/RoboEnv.git
git -C RoboEnv checkout 7bb49439677c2ac111347b40e997025ce87bff50
git -C RoboEnv submodule update --init

uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install -e RoboEnv -r requirements-dev.txt

cd final/robotnik_arm_task
```

For an NVIDIA GPU with CUDA 12.8 support, use the `/whl/cu128` PyTorch index instead of `/whl/cpu`.
**Run all remaining commands from `final/robotnik_arm_task/`.** In a new terminal, activate `.venv` again before continuing.

## 2. Collect and prepare data

Start with a fresh data directory. Collect clean expert trajectories, build the paired subsets, then collect DART trajectories:

```bash
python -m data_collection.scenarios
python -m data_collection.collect --teachers markovian stateful --workers 4
python -m data_collection.qa_clean --teachers markovian stateful --workers 4
python -m data_collection.build_clean_subsets
python -m data_collection.collect_dart --protocol experiments/configs/dart/d2-s075-x2.yaml --workers 4
```

Both teachers are needed to build the clean subsets. Training uses 221 fitting scenarios; validation uses 32 and testing uses 64.
Keep the versioned split in `experiments/splits/clean253.json` unchanged.

Locate the DART collection and build the training features:

```bash
DART_DIR=$(python - <<'PY'
from pathlib import Path
paths = list(Path("data/Markovian/DART").glob("*__collection__d2-s075-x2-*"))
assert len(paths) == 1, "Select the DART directory printed by the collection command"
print(paths[0])
PY
)

for FEATURES in features-v1 features-v1-gate; do
    python -m training.datasets.build --source clean253 --features "$FEATURES" --workers 4
    python -m training.datasets.build --source "$DART_DIR" --name dart-d2s075x2 --features "$FEATURES" --workers 4
done
```

**Shared datasets:** To reuse a teammate's data, restore these complete directories and file under the task directory, then continue at step 3:

- `data/datasets/A/clean253__features-v1-gate/`
- `data/datasets/A/dart-d2s075x2__features-v1-gate/`
- `data/Markovian/Clean_253/scenarios_test.csv`

The two feature datasets total about **1.03 GiB**. Keep all their metadata and checksum files.
Also include `evaluation/test_diagnosis.py` in the source handoff; it may still be a locally untracked file.

## 3. Train R4

This trains one model for 42,000 updates. Use a working copy without the original model checkpoints:

```bash
python -m training.datasets.train --variant R4-dart-gate \
    --set identity.version=v1 --set identity.date=2026-10-28 --set identity.seed=0 \
    --set identity.objective=bcj --set training.lr=1.0e-3 --set training.updates=42000 \
    --set loss.action=huber --set loss.arm_weight=0.5 --set loss.wheel_weight=0.5
```

Add `--dry-run` to check the data without training, or `--resume` to continue an interrupted run.
The same command supports CPU and GPU training.

## 4. Validate the model

Set the model path, then run all 32 validation scenarios:

```bash
cd "$(git rev-parse --show-toplevel)/final/robotnik_arm_task"
RUN=R4-dart-gate__MLP01-256x3__s0__v1__2026-10-28
MODEL="checkpoints/ablation/2026-10-28__training-ablation-v1/$RUN/$RUN.npz"

python -m evaluation.evaluate --policy mlp --artifact "$MODEL" --split validation --workers 4
```

The terminal prints success rates and errors. The original seed-0 model holds **29/32** validation targets.
When sharing an existing model, keep its `.npz` and adjacent `config.yaml` together; training data is not needed for evaluation.

## 5. Test on all 64 test scenarios

After fixing your model using validation results, run the complete block below from anywhere inside this repository.
It selects the task directory and sets the model path:

```bash
cd "$(git rev-parse --show-toplevel)/final/robotnik_arm_task"
RUN=R4-dart-gate__MLP01-256x3__s0__v1__2026-10-28
MODEL="checkpoints/ablation/2026-10-28__training-ablation-v1/$RUN/$RUN.npz"

python -m evaluation.test_diagnosis --models "$MODEL" --modes student \
    --workers 4 --out results/test/r4-seed0
```

This tests the student's arm and base together on all 64 scenarios, then runs expert takeover diagnostics for any failures.
Read the test success count with:

```bash
python - <<'PY'
import csv
from pathlib import Path
with Path("results/test/r4-seed0/rollouts.csv").open() as f:
    rows = [r for r in csv.DictReader(f) if r["mode"] == "student"]
assert len(rows) == 64 and len({r["scenario_id"] for r in rows}) == 64
success = sum(r["sustained_hold_pass"] == "True" for r in rows)
print(f"Test sustained hold: {success}/64 ({success / 64:.2%})")
PY
```

The original seed-0 model holds **60/64** test targets; fresh training may differ across machines.
Success means reaching within 10 mm at or after 8 s, continuing for 3 s and staying within 10 mm throughout the final second (40 s episode limit).
Results are saved in `results/test/r4-seed0/`; use a new output directory when repeating a test.
This entry point reproduces the existing R4 test diagnosis; report it as such and disclose any later model changes informed by these test results.

For all three seeds, other ablation rows, packaging and historical results, see the [detailed reproduction reference](docs/2026-10-29__guide__reproduction-details.md).
