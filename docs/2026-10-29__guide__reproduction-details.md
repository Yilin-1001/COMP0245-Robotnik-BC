# Detailed reproduction reference

Date: 29 October 2026, updated 1 November 2026. Status: command reference for ablation v1 (R1–R4) and v2 (R5–R8).

Navigation: [Documentation index](README.md) · [Project README](../README.md) · [Project pipeline](2026-10-30__guide__project-pipeline.md).

A learned policy maps base pose (`base_pos`, `base_bearing`), arm state (`q_mes`, `qd_mes`) and `TARGET_EE_POS` to arm torques (`torque_arm`, 7) and wheel speeds (`wheel_cmds`, 4). The reproducible experiments are the teacher-A training-method ablation v1 (R1–R4: clean data, a reach-gate input, DART data and their combination, with a damping-Jacobian loss (BC+J) in every row) and v2 (R5–R8: the same four rows plus DAgger). Training, closed-loop evaluation and live comparison are implemented. The earlier clean-only experiments are archived.

For teammate reproduction, start with [Reproduction and data handoff](#reproduction-and-data-handoff): required files, packaging, evaluation, retraining and collection from scratch.

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

The last command reports the training device: `device: cuda` means a CUDA kernel actually ran; otherwise `fallback_reason` explains why training uses the CPU. The cu128 build includes the `sm_120` kernels that RTX 50-series GPUs need; small MLPs also train on the CPU. Installing torch 2.11.0 downgrades `setuptools` to 78.1.0 (a torch constraint); the simulator packages are unchanged. `-m realdata` tests need the datasets from [Data](#data-clean-collection-details).

**WSL2 notes.** Clone the repository into the Linux file system (`~/...`), not under `/mnt/c/`: the datasets are thousands of files and Windows-mounted drives are much slower. GPU training needs only the normal Windows NVIDIA driver (do not install a Linux driver inside WSL); `nvidia-smi` inside WSL should list the GPU. PyBullet GUI windows (`robotnik_arm_task.py`, `markovian_expert_demo.py`, `evaluation.live_compare`) need WSLg (Windows 11, or Windows 10 build 19044+ with `wsl --update`); without it, use `ROBOENV_HEADLESS=1` or `--headless`.

## Reproduction and data handoff

This guide follows `experiments/ablation/2026-10-28__training-ablation-v1.yaml`.
Complete Setup above, then run `cd final/robotnik_arm_task` once from the repository root.
**Run all commands in this section from that task directory.** Use Linux or WSL2, Python 3.10 and the pinned RoboEnv/dependency versions.
Start with `--workers 4` and training `--parallel 1`; increase concurrency as memory and CPU capacity allow.

### 1. What to share with teammates

Share the code through Git and record `git rev-parse HEAD`. The datasets and models below are excluded by `.gitignore`;
transfer them separately and preserve their relative directory structure. Sizes are current uncompressed totals in GiB (2³⁰ bytes).

| Reproduction goal | Required files, relative to the task directory | Size and use |
|---|---|---|
| Evaluate the existing R1–R4 models | `checkpoints/ablation/2026-10-28__training-ablation-v1/` | About 0.45 GiB for the complete directory. NumPy inference needs only each run's same-name `.npz` and adjacent `config.yaml`: about 13.2 MiB for all 12 models |
| Retrain R1/R3 with 21 inputs | `data/datasets/A/clean253__features-v1/` and `data/datasets/A/dart-d2s075x2__features-v1/` | About 0.37 / 0.64 GiB respectively. R1 uses clean only; R3 uses both |
| Retrain R2/R4 with 22 inputs | `data/datasets/A/clean253__features-v1-gate/` and `data/datasets/A/dart-d2s075x2__features-v1-gate/` | About 0.38 / 0.66 GiB respectively. R2 uses clean only; R4 uses both |
| Run hold diagnostics and test scenarios | `data/Markovian/Clean_253/scenarios_train.csv` and `scenarios_test.csv` | Hold diagnostics read the first file; test evaluation reads the second. Screening/validation scenarios are embedded in the versioned split JSON and do not require raw trajectories |
| Rebuild all four feature datasets from raw trajectories (optional) | `data/Markovian/Clean_253/` and `data/Markovian/DART/2026-10-28__collection__d2-s075-x2-58af5db87875/` | About 1.79 / 2.90 GiB respectively. Transfer complete directories, including trajectories, manifests, SHA-256 records and QA |

**Recommended handoff: code + all four feature dataset directories (about 2.03 GiB total) + the model directory + both scenario CSVs.**
To retrain only R4, share the two `features-v1-gate` directories (about 1.03 GiB total). Evaluating existing models does not require training data.
Keep every dataset directory complete: `x.npy`, `y.npy`, `offsets.npy`, `diagnostics.npz`, `episodes.csv`, `metadata.json`, `qa.json`, `README.md` and `COMPLETE.json`.
Transfer the metadata and integrity records together with the arrays.

The clean feature datasets contain 253 candidate training trajectories and 3,045,616 rows:
**221 scenarios are used for fitting and 32 for validation**. Raw DART collection contains 221 fitting scenarios × 2 noise draws = **442 trajectories**;
441 pass the sustained-hold filter and enter the feature datasets, yielding 5,289,586 rows.
Both feature versions contain the same trajectories. Inputs have 21/22 dimensions; labels have 9 dimensions
(7 joint torques plus right/left wheel speeds), expanded into four wheel commands at deployment.
Use the frozen split in `experiments/splits/clean253.json`. Keep validation/test scenarios out of DART and preserve the existing split.

The current R1–R4 workflow uses the teacher-A artifacts listed above.
To reproduce the historical A/B clean experiments or run the full `-m realdata` test suite, also share
`Clean_253/` and `Clean_223/` for both `Markovian` and `Stateful`.
The current ablation does not depend on the old `data/cache/`, archived models or outer workspace `outputs/` directory.

### 2. Package the handoff and restore the paths

On the sender's computer, run the following from the task directory. This creates two archives without raw trajectories.

```bash
mkdir -p results/handoff
git rev-parse HEAD > results/handoff/CODE_COMMIT.txt
uv pip freeze > results/handoff/ENVIRONMENT.txt
tar -czf results/handoff/robotnik-models-v1.tar.gz \
    checkpoints/ablation/2026-10-28__training-ablation-v1 \
    data/Markovian/Clean_253/scenarios_train.csv \
    data/Markovian/Clean_253/scenarios_test.csv
tar -czf results/handoff/robotnik-datasets-v1.tar.gz \
    data/datasets/A/clean253__features-v1 \
    data/datasets/A/clean253__features-v1-gate \
    data/datasets/A/dart-d2s075x2__features-v1 \
    data/datasets/A/dart-d2s075x2__features-v1-gate
(cd results/handoff && sha256sum robotnik-models-v1.tar.gz robotnik-datasets-v1.tar.gz > SHA256SUMS)
```

Send both archives, `SHA256SUMS`, `CODE_COMMIT.txt` and `ENVIRONMENT.txt` from `results/handoff/`.
Include any required uncommitted source files in the handoff version: a commit ID alone does not include local changes.
Use `ENVIRONMENT.txt` to compare environments. Install using Setup above, because a frozen environment listing may contain editable paths specific to the sender's computer.

On the recipient's computer, obtain the matching code version and complete Setup. Copy the handoff files into
`final/robotnik_arm_task/`, then run:

```bash
sha256sum -c SHA256SUMS
tar -xzf robotnik-models-v1.tar.gz
tar -xzf robotnik-datasets-v1.tar.gz
```

Extract into a new working copy to avoid overwriting an existing run with the same name. Both archives can be used together for model evaluation.
**For training from scratch, use a separate clean working copy, extract the datasets archive and extract only the scenario CSVs from the models archive:**

```bash
# For a fresh training checkout only; do not extract the existing checkpoints
tar -xzf robotnik-models-v1.tar.gz \
    data/Markovian/Clean_253/scenarios_train.csv \
    data/Markovian/Clean_253/scenarios_test.csv
```

### 3. Evaluate existing models

First run a headless installation check, then evaluate R4 seed 0 on all 32 validation scenarios:

```bash
python -m evaluation.live_compare --headless --seconds 0.1
MODEL_DIR=checkpoints/ablation/2026-10-28__training-ablation-v1
RUN=R4-dart-gate__MLP01-256x3__s0__v1__2026-10-28
python -m evaluation.evaluate --policy mlp \
    --artifact "$MODEL_DIR/$RUN/$RUN.npz" --split validation --workers 4
python -m evaluation.evaluate --policy expert --teacher A --split validation --workers 4
```

The NumPy `.npz` contains weights and normalisation scales, so inference runs on the CPU. Keep `config.yaml` alongside the model.
The evaluator prints `sustained_hold_success`, `reach_success` and error statistics. Per-scenario results are written to
`results/ablation/A/bcj/<run>/rollouts_validation_step42000_rollout-v1.csv`; expert results go to `results/reference/A/`.
These commands omit `--record` to avoid duplicate entries in the historical `results/results.csv` included in Git.
Per-scenario result files also refuse overwriting: use a new working copy or move the corresponding old result file before repeating an evaluation.

To evaluate all 12 models and generate the ablation summary, first back up the versioned result index in a
**fresh working copy with no local evaluation results**, then run the stages in order:

```bash
# Run once before the first evaluation; retain the new results.csv when resuming
mv results/results.csv results/results.reference.csv
python -m experiments.ablation.run --manifest experiments/ablation/2026-10-28__training-ablation-v1.yaml --stage validate --workers 4
python -m experiments.ablation.run --manifest experiments/ablation/2026-10-28__training-ablation-v1.yaml --stage hold --workers 4
python -m experiments.ablation.run --manifest experiments/ablation/2026-10-28__training-ablation-v1.yaml --stage summary
```

`validate` evaluates each model with the full student and with the student arm plus expert base.
`hold` runs takeover diagnostics on 8 fitting scenarios. Main outputs are
`results/ablation/2026-10-28__training-ablation-v1/summary.md`, `summary.csv` and `runs.csv`;
details are saved in its `validation/` and `hold_test/` directories.

Reference **full-student validation sustained hold** for the original terminal models, out of 32 scenarios per seed:

| Experiment | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| R1-baseline | 0/32 | 1/32 | 0/32 |
| R2-gate | 2/32 | 3/32 | 5/32 |
| R3-dart | 0/32 | 1/32 | 0/32 |
| R4-dart-gate | 29/32 | 27/32 | 27/32 |
| Teacher A | 32/32 | — | — |

Success requires reaching an end-effector error below 10 mm at t ≥ 8 s within a 40 s cap, continuing for another 3 s,
and remaining below 10 mm throughout the final 1 s.
Report sustained hold as the primary metric; report `reach_success` and `hold_test` takeover results separately.
The standard evaluation CLI accepts `screening`/`validation`. The table above covers validation.

**See section 6, “Evaluate on the test set”, for the 64-scenario test commands, summary code and reference results.**

### 4. Retrain from the shared datasets

In a separate working copy without the original checkpoints, verify all four feature datasets and their checksums:

```bash
python - <<'PY'
from training.datasets.load import load_dataset
for name in (
    "clean253__features-v1", "clean253__features-v1-gate",
    "dart-d2s075x2__features-v1", "dart-d2s075x2__features-v1-gate",
):
    cache = load_dataset(name)
    print(name, len(cache.episode_ids), cache.x.shape, cache.y.shape)
    del cache
PY

# Prepare data and calibrate BC+J without optimisation; remove --dry-run to train R4 seed 0
python -m training.datasets.train --variant R4-dart-gate \
    --set identity.version=v1 --set identity.date=2026-10-28 --set identity.seed=0 \
    --set identity.objective=bcj --set training.lr=1.0e-3 --set training.updates=42000 \
    --set loss.action=huber --set loss.arm_weight=0.5 --set loss.wheel_weight=0.5 --dry-run

# Train 4 rows x 3 seeds from the frozen manifest; lambda_J is calibrated per row
python -m experiments.ablation.run --manifest experiments/ablation/2026-10-28__training-ablation-v1.yaml \
    --stage train --parallel 1
```

After training, back up the result index once as described in section 3, then run `validate`, `hold` and `summary` in order.
Training writes `checkpoints/ablation/2026-10-28__training-ablation-v1/<run>/<run>.npz` and `.pt`.
Evaluate every model at its terminal update, 42,000.
The `train` stage skips existing same-name terminal models and automatically resumes runs that have only `latest.pt`.
Use a checkout without the sender's checkpoints to perform a fresh training run.
To resume a single model, add `--resume` to the identical training command.
CPU training is supported; use serial runs on machines with limited memory.
Retraining across GPU/CPU or different numerical libraries does not guarantee identical weight hashes or success counts.
Report evaluation of the original models and fresh training separately, retaining the new run's `config.yaml`, `provenance.json` and metrics.
For GUI comparison after retraining, complete validation and run the `--stage demo` stage of `experiments.ablation.run`
to generate a local demo manifest matching the new weight hashes. The versioned demo manifest pins the original model hashes.

### 5. Collect trajectories and build the feature datasets

Use a **fresh working copy with neither handoff archive extracted**. If you already have the four feature datasets, continue directly to training.
Full clean collection and paired subset construction require both teachers A and B: the current `build_clean_subsets` needs both inventories.
Allow at least about 25 GiB of disk space for raw trajectories, both teachers' derived copies, feature datasets and checkpoints.
Runtime depends on CPU and memory capacity.

```bash
# Generate 256 train + 64 test Sobol scenarios, then collect both teachers A and B
python -m data_collection.scenarios
python -m data_collection.collect --teachers markovian stateful --workers 4
python -m data_collection.qa_clean --teachers markovian stateful --workers 4
python -m data_collection.build_clean_subsets

# Collect DART on the 221 frozen fitting scenarios, with 2 noise draws per scenario
python -m data_collection.collect_dart --protocol experiments/configs/dart/d2-s075-x2.yaml --workers 4

# Collection names include the date/code digest; resolve the local collection path
DART_DIR=$(python - <<'PY'
from pathlib import Path
paths = sorted(Path("data/Markovian/DART").glob("*__collection__d2-s075-x2-*"))
if len(paths) != 1:
    raise SystemExit("Select the matching DART collection directory from the collection output")
print(paths[0])
PY
)

# Build v1 before gate; the gate build checks that its first 21 columns match v1
python -m training.datasets.build --source clean253 --features features-v1 --workers 4
python -m training.datasets.build --source clean253 --features features-v1-gate --workers 4
python -m training.datasets.build --source "$DART_DIR" --name dart-d2s075x2 --features features-v1 --workers 4
python -m training.datasets.build --source "$DART_DIR" --name dart-d2s075x2 --features features-v1-gate --workers 4
```

The teacher CLI names are `markovian` (A) and `stateful` (B); each should have 320 raw clean trajectories.
`build_clean_subsets` creates both teachers' `Clean_253/` and `Clean_223/`, verifies paired selection, manifests and trajectory checksums,
and refuses to overwrite existing directories. Retain the versioned split JSON; no `--write-split` step is needed.
`build_clean_subsets` asserts the expected inventory (253 paired successes, 223 without hold chatter, excluded `train-0064`, `train-0131`, `train-0183`); collection took about 4 minutes on 22 workers.
If success counts, the 253/223 subset assertions or QA differ on your machine, retain the logs and check simulator/dependency versions.
If only teacher B differs (for example B fails `train-0220`), `python tools/build_a_clean253.py` builds teacher A's `Clean_253` alone and checks it against the frozen split; it is enough for every teacher-A row.
Preserve these checks when reporting reproduction of the same dataset.
Recollection on another machine does not guarantee identical floating-point trajectories or compressed-file SHA-256 values;
use the sender's data archives when you need the original data.
After building, return to section 4 to train. To rebuild features from shared raw trajectories, start with the `DART_DIR` assignment and the four build commands.

### 6. Evaluate on the test set

The test set contains **64 separate `test-*` scenarios**. Validation uses 32 scenarios; `pytest -m realdata` checks data loading.
Test evaluation runs PyBullet closed-loop simulations: the model controls the robot from each scenario's initial state,
and the evaluator measures reach and sustained hold. The R4 test commands below require:

- `data/Markovian/Clean_253/scenarios_test.csv`: initial poses and targets for all 64 scenarios.
- `experiments/splits/clean253.json`: the versioned split and scenario CSV SHA-256. The loader checks this hash; preserve the CSV bytes.
- The model `.npz` files and each model's adjacent `config.yaml`, plus the installed RoboEnv.

Raw expert trajectories under `test/*.npz` are not required for this closed-loop evaluation.
Complete Setup, extract the model archive and run the following from `final/robotnik_arm_task/` to test all three R4 seeds.
`--modes student` lets the student control both arm and base. Omit `--scenarios` to run the complete 64-scenario test set.

```bash
MODEL_DIR=checkpoints/ablation/2026-10-28__training-ablation-v1
R4_S0=R4-dart-gate__MLP01-256x3__s0__v1__2026-10-28
R4_S1=R4-dart-gate__MLP01-256x3__s1__v1__2026-10-28
R4_S2=R4-dart-gate__MLP01-256x3__s2__v1__2026-10-28
python -m evaluation.test_diagnosis --models \
    "$MODEL_DIR/$R4_S0/$R4_S0.npz" \
    "$MODEL_DIR/$R4_S1/$R4_S1.npz" \
    "$MODEL_DIR/$R4_S2/$R4_S2.npz" \
    --modes student --workers 4 --out results/test/r4-student-reproduction
```

This runs **192 primary rollouts (64 × 3)**, followed by expert takeover diagnostics for failed student trajectories.
To test one seed, keep only that model path after `--models`. Add `--scenarios test-0000` for a single-scenario installation check;
report such a run as a smoke check. For newly trained models, replace `--models` with their terminal export paths and retain the matching `config.yaml` files.

Outputs are saved under `--out`. **Use a new output directory for each evaluation**: the diagnostic script permits overwriting existing results.

| File | Interpretation |
|---|---|
| `rollouts.csv` | One row per mode, model and scenario. The primary metric is `sustained_hold_pass` for `mode=student`; `ever_reached` records reach and `final_err_mm` records final error |
| `run.json` | Model paths, modes and scenario count. A complete test run has `n_scenarios: 64` |
| `traces/` | Expert/full-student trajectory records for failure analysis |
| `takeover.csv` | Created only when full-student failures exist. Expert recovery is reported separately from the student's success rate |

Run the following to calculate counts, success rates and median final error per model/mode, and write `summary.csv`:

```bash
python - <<'PY'
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

out = Path("results/test/r4-student-reproduction")  # Match the --out directory above
assert json.loads((out / "run.json").read_text())["n_scenarios"] == 64, "not the complete test set"
groups = defaultdict(list)
with (out / "rollouts.csv").open(newline="") as f:
    for row in csv.DictReader(f):
        groups[row["mode"], row["policy"]].append(row)
summary = []
for (mode, policy), rows in sorted(groups.items()):
    assert len(rows) == 64 and len({r["scenario_id"] for r in rows}) == 64
    hold = sum(r["sustained_hold_pass"] == "True" for r in rows)
    reach = sum(r["ever_reached"] == "True" for r in rows)
    summary.append(dict(mode=mode, policy=policy, n=64, hold=hold, reach=reach,
                        hold_rate=hold / 64, reach_rate=reach / 64,
                        median_final_err_mm=statistics.median(float(r["final_err_mm"]) for r in rows)))
    print(f"{mode} {policy}: hold={hold}/64 ({hold/64:.2%}), reach={reach}/64")
with (out / "summary.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary[0]))
    writer.writeheader()
    writer.writerows(summary)
PY
```

Reference full-student sustained hold for the original R4 terminal models is **60/64 for seed 0, 60/64 for seed 1 and 59/64 for seed 2**.
Pooled across seeds, this is 179/192, or about **93.23%**. The success predicate is the same as in section 3:
reach below 10 mm at t ≥ 8 s, continue for 3 s, and remain below 10 mm throughout the final 1 s.
Compare original-model reproductions with these values; report your measured counts for newly trained models.

To reproduce the **full bottleneck diagnosis from 2026-10-29**, replace `--modes student` with
`--modes teacher student arm base` and choose another `--out` directory.
This runs 640 rollouts: 64 expert rollouts plus three seeds × three student/mixed modes × 64 scenarios, followed by failed-trajectory takeovers.
The expert reference is 64/64. Mixed modes pair the student arm with the expert base, or the student base with the expert arm.
Use them to locate bottlenecks and retain `mode=student` as the primary model result.

**Reporting:** this entry point reproduces the R4 diagnosis after the test set was viewed. It does not write to the formal `results/results.csv`
and should be reported with that test-set exposure disclosed. Freeze models, hyperparameters and checkpoint choices using training/validation data
before testing, and disclose any use of these test outcomes in subsequent design.
The `evaluation.evaluate` CLI has no `--split test` option.
The clean-data MLP pipeline that preceded the ablation (M1–M6, `experiments.run`, frozen protocols; no clean model held a validation target) is archived with its commands and results in [Archive/docs/2026-10-29__guide__clean-pipeline-legacy.md](../Archive/docs/2026-10-29__guide__clean-pipeline-legacy.md).

### 7. DAgger rows (ablation v2)

Each DAgger row starts from its base row's v1 seed-0 model in `checkpoints/ablation/2026-10-28__training-ablation-v1/`, so restore or train v1 first (sections 2–4).
One command runs the whole row: collect, build, train and validate for each round, then seeds 1–2 of the round with the best seed-0 validation hold.

```bash
for ROW in R5-baseline-dagger R6-gate-dagger R7-dart-dagger R8-dart-gate-dagger; do
    python -m experiments.dagger_rounds --row "$ROW" --rounds 3 --date 2026-10-30 --final-seeds 1 2 --workers 4
done
```

| Output | Location |
|---|---|
| Raw DAgger episodes (one folder per round and student) | `data/Markovian/DAgger/<date>__collection__r<k>-<digest>/` |
| Feature datasets | `data/datasets/A/dagger-R<n>-v2-r<k>__<features>/` |
| Variant configs (written by the runner) | `experiments/configs/datasets/variants/<row>-r<k>.yaml` |
| Models | `checkpoints/ablation/2026-10-30__training-ablation-v2/<row>-r<k>__MLP01-256x3__s<seed>__v2__2026-10-30/` |
| Per-row summary, validation JSON and logs | `results/ablation/2026-10-30__training-ablation-v2/dagger/<row>/` |

To hand over v2 models, package `checkpoints/ablation/2026-10-30__training-ablation-v2/` as in section 2. The DAgger feature datasets are only needed for retraining.
The v2 results recorded so far come from a teammate's machine and use legacy names; the mapping is in `results/ablation/2026-10-30__training-ablation-v2/README.md`.

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

Simulator and clean-collection settings are in `final/robotnik_arm_task/settings.yaml`. Its `mlp`, `train_stride` and `dagger` keys are legacy placeholders: the training code never reads them, and the file is left byte-identical because the clean-data QA recomputes a code digest over it. Training settings for the ablation resolve as `experiments/configs/common.yaml` → `objectives/<objective>.yaml` → `datasets/base.yaml` → `datasets/variants/<row>.yaml` → `--set key=value`. The recipe, split and success predicate are in the [project pipeline](2026-10-30__guide__project-pipeline.md) (Sections 2, 3 and 6); the clean datasets, outcome definitions and QA are in the [clean collection report](reports/2026-10-26__report__clean-expert-data-collection.md). The design notes these settings came from are archived in `Archive/docs/`.

The live comparison demo (two synchronised PyBullet windows, expert A against a student) is described in the [project pipeline, Section 9](2026-10-30__guide__project-pipeline.md#9-comparison-demo).

Checks: `python -m pytest tests -m "not sim and not realdata"` (fast), `-m realdata` (reads the Clean_253/223 inventories and builds caches), `-m sim` (expert preflight through the student action path in PyBullet).

## Experiment naming

- **Feature definitions** (`core/feature_sets.py`): `features-v1` (21 inputs) and `features-v1-gate` (22 inputs: v1 plus `reach_gate`, teacher A's reach-active rule recomputed from deployable inputs). Every exported model records its definition, so `MLPPolicy` builds the same inputs at deployment.
- **DART collection** (`data_collection/collect_dart.py`): the clean episodes with gated OU noise on the executed action, while the labels stay the clean expert action. Each noise protocol is one file in `experiments/configs/dart/`.
- **DAgger collection** (`data_collection/collect_dagger.py`): the student drives, teacher A labels every visited state (section 7).
- **Training-ready datasets**: `data/datasets/A/<source>__<definition>/` (features, labels, row offsets, episode list, QA and README; git-ignored), built by `training.datasets.build` (section 5).
- **Rows**: one file per row in `experiments/configs/datasets/variants/`. v1: `R1-baseline`, `R2-gate`, `R3-dart`, `R4-dart-gate`. v2: `R5-baseline-dagger`, `R6-gate-dagger`, `R7-dart-dagger`, `R8-dart-gate-dagger` (one variant per round, `<row>-r<k>`), plus the R4 controls `R4-dart-gate-s0375`, `-s150`, `-x4` and `-split`.

Training uses the frozen split: clean datasets keep only fitting scenarios, DART datasets hold fitting scenarios only (checked), and the 32 validation scenarios come from the clean dataset. Architectures use a new registry, `MLP<NN>` numbered in order of first use (`MLP01` = 256×256×256; the v2 network-size runs added `MLP02` = 64×64, `MLP03` = 128×128 and `MLP04` = 128×128×128 through `--set`), in place of the clean pipeline's m1–m6. Runs are named `<row>__<architecture>-<shape>__s<seed>__<version>__<date>` and stored in `checkpoints/ablation/<date>__training-ablation-<version>/`, for example `checkpoints/ablation/2026-10-28__training-ablation-v1/R4-dart-gate__MLP01-256x3__s0__v1__2026-10-28/`. The terminal model is also saved there as `<run name>.npz` and `<run name>.pt`, and it loads in `evaluation.evaluate` like any other MLP. Raise the version whenever the recipe, data or features change (v2 added DAgger data, with experiment date 2026-10-30); reusing a name with a different configuration is refused.

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
│   ├── training/          common/ (model, engine, policy), baseline/ + jacobian/ (BC and BC+J losses),
│   │                      datasets/ (build.py: raw -> training-ready datasets; train.py: ablation training)
│   ├── evaluation/        evaluate.py (closed loop), offline.py, test_sets.py, test_diagnosis.py, probes, hold_test.py, live_compare.py, expert diagnostics
│   ├── experiments/       configs/ (common, objectives, teachers, dart/, datasets/), splits/, ablation/ (v1 runner + manifest),
│   │                      dagger_rounds.py (v2 DAgger runner), demo_models/
│   ├── tools/             build_a_clean253.py (teacher-A-only Clean_253 when teacher B differs locally)
│   ├── data/              git-ignored: scenarios/, Markovian/{Clean,Clean_253,DART,…}/ (raw), datasets/A/ (training-ready)
│   ├── checkpoints/       git-ignored: ablation/<date>__training-ablation-<version>/<run>/
│   └── results/           results.csv (v1 evaluation index); ablation/<date>__training-ablation-<version>/ (summaries tracked)
├── tests/                 pytest suite (pytest.ini: testpaths = tests)
├── docs/                  guides, specs, reports, references (index: docs/README.md)
└── Archive/               retired code (tracked) and pre-ablation models (git-ignored); see Archive/README.md
```

Where the splits live: training = the 221 `fitting_ids` and validation = the 32 `validation_ids` of `experiments/splits/clean253.json` (their rows are in `data/datasets/A/clean253__*`; DART datasets hold training scenarios only); test = the 64 sealed scenarios in `data/Markovian/Clean_253/scenarios_test.csv`, evaluated in closed loop only.

## Outputs

Every number in the report comes from an experiment folder, `results/ablation/<date>__training-ablation-<version>/`:

| Experiment | Summary (tracked in Git) | Details (local only) |
|---|---|---|
| v1 (R1–R4) | `summary.md`, `summary.csv`, `generalisation/summary.json` | `runs.csv`, `validation/`, `hold_test/`, `logs/` |
| v2 (R5–R8 and controls) | `README.md`, `dagger/<row>/summary.csv`, `probes/{far,robust}/<models>/summary.json` | `dagger/<row>/validation/`, `logs/` |
| Test runs | Not tracked yet | `results/test/<name>/` |

- `.gitignore` keeps `results/` local except `results.csv`, `figures/` and the `summary.*` and `README.md` files under `results/ablation/`.
- `results/results.csv` is the append-only index of evaluations recorded with `--record` (`evaluation.evaluate`, `experiments.ablation.run`): the v1 validation runs and the archived clean pipeline. It refuses duplicate (run, step, split, evaluation ID) keys. The v2 runner does not write to it, so its folder summaries are the reference. Use the experiment folders, not `results.csv`, as the source of reported numbers.
- Raw data, checkpoints and per-rollout details are not committed. Every run records its config, data, split and code hashes in `provenance.json`.

## External code

The table of external, modified and from-scratch code is in the [project README](../README.md#external-code).
