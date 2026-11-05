# Model names and final checkpoint submission

Names follow the Report. `core/model_names.py` is the shared naming registry used by training aliases,
demo labels, evaluation reports, plots and the final checkpoint exporter.

| Row | Report name | Canonical model ID | Historical alias |
|---|---|---|---|
| R1 | Baseline | `R1-Baseline` | `R1-baseline` |
| R2 | Baseline + Gate | `R2-Baseline-Gate` | `R2-gate` |
| R3 | OU | `R3-OU` | `R3-dart` |
| R4 | OU + Gate | `R4-OU-Gate` | `R4-dart-gate` |
| R5 | DAgger | `R5-DAgger` | `R5-baseline-dagger` |
| R6 | DAgger + Gate | `R6-DAgger-Gate` | `R6-gate-dagger` |
| R7 | f-DAgger | `R7-f-DAgger` | `R7-dart-dagger` |
| R8 | f-DAgger + Gate | `R8-f-DAgger-Gate` | `R8-dart-gate-dagger` |
| R9 | DART + Gate | `R9-DART-Gate` | `R9-dart-full` |

`Gate` always means the 22nd input, `reach_gate`. Rows without it have 21 inputs.
OU collection uses phase-gated noise even for R3/R7; that noise gate does not mean the model has a Gate input.
R3/R4 and R7/R8 use fixed OU noise; DART is reserved for full learner-fitted Gaussian DART, R9.
“f-DAgger” alone identifies R7 when naming a model. The algorithm can also be discussed generically as f-DAgger.

Round, seed, architecture, training version and training date remain explicit in artifact filenames, e.g.
`R8-f-DAgger-Gate-r3__MLP01-256x3__s0__v3__2026-11-05.npz`.
For R9 the `-a6` qualifier records alpha=6; the exploratory alpha=1 run is excluded from this submission.

## Upload folder

Upload [`final/robotnik_arm_task/checkpoints/2026-11-05final/`](../final/robotnik_arm_task/checkpoints/2026-11-05final/README.md).

- `models/`: the 24 Report-tested R1–R8 final models plus 3 DART comparison models, seeds 0/1/2.
- `round_history/`: 22 additional report round-history models, with the original available seeds.
- `model_index.csv`: names, seeds, rounds, artifact paths, source run IDs and hashes.
- `manifest.json`: the complete provenance and numerical verification records.
- `SHA256SUMS`: file integrity checks, including configs and provenance.
- `demo_models.yaml`: the final 27-model demo manifest.

R1–R4 use their frozen v1 terminal models. R5–R8 use the validation-selected round 3 of ablation v3 (DAgger without
development scenes, 2026-11-05; it replaces v2, see `final/robotnik_arm_task/report/08_dev_scene_removal/`);
R9 uses alpha=6, round 3. Every checkpoint is at step 42,000.
The package contains real files and needs no links to historical weights.
PT metadata, NPZ metadata and the released config all use the canonical run name.
`source_config.yaml` and `legacy_run_id` retain the original training record. Renaming does not retrain the network.
The original experiment folders, raw result IDs and source snapshots remain provenance aliases, so their
recorded hashes and reproduction commands still refer to the actual historical experiment.

From `final/robotnik_arm_task/`:

```bash
# Use the submission weights in the live demo.
python -m evaluation.live_compare --protocol checkpoints/2026-11-05final/demo_models.yaml --row R8-f-DAgger-Gate --seed 0

# Canonical names also resolve the original training templates for new runs.
python -m training.datasets.train --variant R4-OU-Gate --dry-run --set identity.date=2026-11-04 --set identity.version=v1 --set training.lr=1e-3 --set training.updates=42000 --set loss.action=huber

# Reproduce the release from the original local artifacts; refuses an existing destination.
python -m tools.export_final_checkpoints --date 2026-11-05
```

Both old and canonical names work in the demo. Old training commands retain their original run IDs;
canonical `--variant` commands produce canonical run IDs from the same data and recipe.
The final release has exact tensor/array/normaliser comparisons and 128 prediction probes for each backend
(NumPy and PyTorch) per model; every pre/post naming prediction difference is zero.
