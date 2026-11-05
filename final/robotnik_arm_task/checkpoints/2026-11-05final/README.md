# Final report checkpoints

Date: 2026-11-05. 27 final models + 22 round-history models.

| Row | Report name | Directory |
|---|---|---|
| R1 | Baseline | `models/R1-Baseline/` |
| R2 | Baseline + Gate | `models/R2-Baseline-Gate/` |
| R3 | OU | `models/R3-OU/` |
| R4 | OU + Gate | `models/R4-OU-Gate/` |
| R5 | DAgger | `models/R5-DAgger/` |
| R6 | DAgger + Gate | `models/R6-DAgger-Gate/` |
| R7 | f-DAgger | `models/R7-f-DAgger/` |
| R8 | f-DAgger + Gate | `models/R8-f-DAgger-Gate/` |
| R9 | DART + Gate | `models/R9-DART-Gate/` |

Each final row has seeds 0, 1, 2. R1–R4 use round 0; R5–R9 use round 3.
R5–R8 come from ablation v3 (2026-11-05): every DAgger round collected the 221 fitting scenarios only,
without development scenes (report/08_dev_scene_removal). v3 run ids carry "-nodev"; release names drop it.
R9 is full DART with alpha=6. The alpha=1 exploratory run is excluded.
`round_history/` contains the other report training runs, with their original available seeds.
Gate means the 22nd reach_gate input; rows without Gate have 21 inputs.
OU and f-DAgger use fixed OU noise. Only R9 is named DART.

Every model directory contains a .pt checkpoint, .npz NumPy export, normaliser, metrics,
released config, original source_config.yaml, and source provenance. Use .npz for NumPy inference
or .pt for PyTorch inference. These are terminal step-42000 inference releases, not resume states.

`manifest.json` and `model_index.csv` map canonical names to source run IDs and SHA-256 hashes.
NPZ source_checkpoint is relative to its own directory. Original paths inside source records
are provenance and are not needed for inference. No symlinks or external weights are required.
All network tensors, exported arrays and normalisers were checked for exact equality;
128 input probes per backend/model produced exactly equal predictions before/after naming.

From final/robotnik_arm_task:
```bash
python -m evaluation.live_compare --protocol checkpoints/2026-11-05final/demo_models.yaml --row R8-f-DAgger-Gate --seed 0
```
The demo protocol paths are relative to the task directory. If relocating this folder,
load the chosen model directly with training.common.policies.MLPPolicy.load(path).
