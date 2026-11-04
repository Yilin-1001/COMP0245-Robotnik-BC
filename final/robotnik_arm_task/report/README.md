# Report evidence

The upload weights use these Report names in filenames, checkpoint metadata and configs:
[`../checkpoints/2026-11-04final/`](../checkpoints/2026-11-04final/README.md).
See the [canonical model IDs and historical aliases](../../../docs/model-naming.md).

Each folder backs one part of the written report with its figures, tables, code and data.

| Section | Folder | Contents |
|---|---|---|
| 01. Memoryless Markovian expert | [`01_markovian_expert/`](01_markovian_expert/README.md) | Why the project needs a memoryless (Markovian) expert, how it differs from the course template controller, and the measured capability trade-off between the two |
| 02. Data collection | [`02_data_collection/`](02_data_collection/README.md) | How demonstrations are collected, how scenario and recovery-state coverage support generalisation, why seeds are fixed, and why scrambled Sobol sampling is used, with reproducible coverage figures and tables |
| 03. Feature engineering | [`03_feature_engineering/`](03_feature_engineering/README.md) | The 21/22-input deployable representation, base-yaw geometry, FK task error, reach gate, fitting-only scaling and shared training/inference code, with original source snapshots, QA, accuracy evidence and LaTeX figures |
| 04. f-DAgger Algorithm | [`04_f_dagger/`](04_f_dagger/README.md) | The f-DAgger algorithm (phase-gated OU noise injection, then DAgger): LaTeX pseudocode, design evidence for each setting, and the gate × OU noise × DAgger ablation (R1–R8, validation and unified test) |
| 05. Training details | [`05_training_details/`](05_training_details/README.md) | The shared training recipe (network, Huber + damping-Jacobian loss, AdamW, warm-up + cosine lr, 42k updates), its provenance, training curves, M6 offline and R4 closed-loop BC vs BC+J comparisons, numerical correctness checks and source snapshots |
| 06. Ablation Experiment | [`06_ablation/`](06_ablation/README.md) | The R1–R8 ablation under the report names Baseline, OU, DAgger and f-DAgger, each without and with the Gate: validation, unified test per suite and band, DAgger rounds, gate effect, failure modes, hold test, cost and efficiency |
| 07. Training time comparison | [`07_training_time_comparison/`](07_training_time_comparison/README.md) | Training time vs test success for f-DAgger + Gate (ours), DAgger + Gate and full DART (DART + Gate), with the supporting noise and far-target analyses |

The [OU noise parameter-selection experiments](04_f_dagger/ou_parameter_selection.md) are indexed under `04_f_dagger/`, with their original pilot traces, stress-test records, protocol files and source hashes in [`04_f_dagger/data/ou_noise/`](04_f_dagger/data/ou_noise/README.md).

**Naming (all folders).** Four methods, each without and with the Gate (the 22nd input `reach_gate`): R1 Baseline, R2 Baseline + Gate, R3 OU, R4 OU + Gate, R5 DAgger, R6 DAgger + Gate, R7 f-DAgger, R8 f-DAgger + Gate. The main pipeline (R4 → R8) is therefore **f-DAgger + Gate**, its DAgger comparison (R2 → R6) **DAgger + Gate**, and full DART (R9, also 22 inputs) **DART + Gate**. "f-DAgger" without "+ Gate" means R7, or the algorithm itself in `04_f_dagger/`. Run ids are unchanged (`R1-baseline` … `R8-dart-gate-dagger`, `R9-dart-full-a6`; "dart" in R1–R8 ids = OU noise); the mapping is in [`06_ablation/tables/naming.csv`](06_ablation/tables/naming.csv).
