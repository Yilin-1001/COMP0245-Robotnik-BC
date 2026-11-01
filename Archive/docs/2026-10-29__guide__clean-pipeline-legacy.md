# Clean-pipeline reproduction (archived)

Archived on 2026-10-30 from `docs/2026-10-29__guide__reproduction-details.md`. These sections describe the clean-data
MLP pipeline (M1–M6 architectures, `experiments/run.py`, frozen `experiments/protocols/p-*.yaml`). That pipeline was
superseded by the R1–R4 training-method ablation. Its code, protocols and models are under `Archive/final/robotnik_arm_task/`;
move them back to their original paths (see `Archive/README.md`) before running any command below.

## Formal test of the historical clean models

Historical clean A/B models use `python -m experiments.run --stage test --protocol experiments/protocols/p-e26487f2a548.yaml` for formal testing.
That protocol accepts only its declared older models, paths and hashes; those archived models are outside the current R1–R4 handoff archive.

## MLP training and evaluation (clean pipeline)


Clean-pipeline defaults from 2026-10-27 use `loss.arm_weight=0.8` and `loss.wheel_weight=0.2` for the two
standardised action-group means. The R1–R4 ablation explicitly overrides both to 0.5 in its manifest.
Both are explicit CLI/configuration fields and enter the run identity.
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

Runs are written to `checkpoints/<regime>/<teacher>/<objective>/<run_id>/` (`config.yaml`, `provenance.json`, `normaliser.npz`, `metrics.csv`, `step_<k>.pt` + `step_<k>.npz` exports, resumable `latest.pt`; `best.pt` + `inference.npz` once selected). The clean-pipeline checkpoints themselves are archived under `Archive/final/robotnik_arm_task/checkpoints/`, as are the deprecated `training.train_mlp` wrapper and the DAgger A code (see [Archive/README.md](../README.md)).

### Clean MLP results (final protocol `p-e26487f2a548`)

Protocol chain: pilot/budget `p-0dedccad51dc` (LR 3e-4, 42,000 updates), loss `p-2a37f68e30af` (Huber), screen `p-9e89dc42395f`, seeds `p-a67f676c83f0` (selected A M6, B M1), calibration `p-13adaf7f1178` (λ_J 0.0703 A, 0.0671 B), final `p-e26487f2a548`. Refit was skipped; the declared artifacts are the validation-selected `best.pt` of each selected architecture, BC and BC+J, seeds 0–2.

| Teacher | Objective | Test sustained hold (64, seeds 0/1/2) | Test reach | Validation hold (32) |
|---|---|---|---|---|
| A (M6) | BC | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| A (M6) | BC+J | 0 / 1 / 1 | 2 / 3 / 8 | 0 / 0 / 0 |
| B (M1) | BC | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| B (M1) | BC+J | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |

Both experts hold 32/32 validation scenarios through the same evaluation path. No clean-data MLP controls the robot: all 12 screened architecture/teacher selections, the finalist seeds and the Clean_223 comparison hold 0/32 on validation. Offline errors are small (validation arm RMSE about 0.4–0.7 N·m) and expert rollouts with residual-matched i.i.d. noise still hold (A 32/32, B 29/32), so the failure comes from state-dependent errors, not their size. BC+J does its job offline: the learned damping median diagonal moves from about −8 to −29.9 (A) and −27.3 (B) N·m·s/rad, against the teachers' −30. The position stiffness ∂τ/∂q at the hold pose stays indefinite (largest symmetric eigenvalue +16.8 for A BC+J and +69.5 for A BC; teacher about −100 to −116), so the hold remains statically unstable. Remedies that act on this (state-perturbation data such as DART/DAgger, or ∂τ/∂q supervision) lie outside this stage.
