# Jacobian validation source snapshot

Copied on 2026-11-04. The files preserve their repository paths and contents. Original paths and SHA-256 hashes are in [`../../data/jacobian_validation_sources.json`](../../data/jacobian_validation_sources.json). These are inspection copies of the current implementation, not a standalone training package or the historical code revisions of the October and November runs.

| Files | Purpose |
|---|---|
| `training/jacobian/loss.py` | Masked derivative loss, sub-batch selection and λ_J calibration |
| `evaluation/offline.py` | Standardised derivative targets, autodiff Jacobian and physical damping diagnostics |
| `training/baseline/loss.py`, `training/common/normalisation.py` | Action loss and units |
| `training/common/models.py`, `training/common/engine.py` | Network, independent initialisation/sampling streams and optimisation |
| `tests/training/test_jacobian.py`, `tests/conftest.py` | Nine correctness checks and their original fixtures |
| `experiments/ablation/run.py` | Training, closed-loop evaluation, hold diagnostics and result aggregation |
| `experiments/ablation/*v1.yaml` | BC objective ablation and the original R1–R4 BC+J experiment |
| `experiments/configs/datasets/variants/R4-dart-gate*.yaml` | The two R4 variants' identical data and input definitions |

Run from `final/robotnik_arm_task/` in the complete repository:

```bash
python report/05_training_details/code/verify_jacobian_sources.py
python -m pytest ../../tests/training/test_jacobian.py -q
python report/05_training_details/code/build_jacobian_validation.py
```

The verification command checks all 33 code/data copies against their recorded hashes and the original files. Tests run against the repository modules, whose inspected source is verified equal to the copies. The figure/table builder uses only copied CSVs and configs; it checks per-scene outcomes against summary counts and checks configuration equality on the experimental controls. It does not need model weights or run the simulator. See the [training-details explanation](../../README.md) for the interpretation and full experiment command.
