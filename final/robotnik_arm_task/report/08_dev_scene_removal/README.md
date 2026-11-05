# 08. DAgger without development scenes (v3 replaces v2)

The report's DAgger rows R5–R8 (DAgger, DAgger + Gate, f-DAgger, f-DAgger + Gate) come from **ablation v3** (2026-11-05). This folder records why v3 replaced the earlier v2 run (2026-11-01) and how much the change moved the results.

## Why v3 exists

In v2, every DAgger round collected the 221 fitting scenarios **plus 64 fresh development scenes**, so each round had 285 episodes. The development-scene sampler was biased towards low and near targets: 70% of targets at heights 0.1–0.5 m and 50% at distances 0.5–1.0 m (manifest field `development_scenes`). That bias was chosen after a failure diagnosis that had looked at the T0 test scenes. No test scene was copied, and every development scene lies inside the training ranges. Even so, test information could have shaped the training data.

v3 removes the development scenes (`--dev-scenes 0`), so each round collects **221 episodes**, all fitting scenarios. Everything else is the same as v2:

- the same round-0 students (the v1 seed-0 models of R1–R4)
- the same recipe (BC+J, MLP01, 42k updates)
- 3 rounds, with round 3 selected for every row
- seeds 0–2
- the same 32 validation scenes
- the same 240 unified test scenes (`scene_manifest.json` is byte-identical in both test runs)

R9 (DART + Gate) never used development scenes and is unchanged.

| Version | Episodes per round | Fitting | Development | Checkpoints | Validation | Unified test |
|---|---:|---:|---:|---|---|---|
| v2 (superseded) | 285 | 221 | 64 | `checkpoints/ablation/2026-11-01__training-ablation-v2/` | `results/ablation/2026-11-01__training-ablation-v2/dagger/` | `results/test/2026-11-01_215340__R1-R8/` |
| **v3 (report)** | **221** | 221 | 0 | `checkpoints/ablation/2026-11-05__training-ablation-v3/` | `results/ablation/2026-11-05__training-ablation-v3/dagger/` | `results/test/2026-11-05_130144__nodev-dagger-R5-R8/` |

## Effect of removing the development scenes

Success rates are pooled over 3 seeds. Differences are v3 − v2 in percentage points, paired by scene, with 95% paired-bootstrap intervals. Holm correction is over all 36 tests in [`tables/dev_scene_effect.csv`](tables/dev_scene_effect.csv). The complete tables are in [`tables/summary.md`](tables/summary.md).

| Row | Validation | T0 | All 240 | Holm-significant changes |
|---|---|---|---|---|
| R5 DAgger | 89.6 → 88.5 | 96.4 → 96.4 | 75.0 → **80.0** | T3 +20.8 [13.5, 28.1]; T1–T5 +6.8; all 240 +5.0 |
| R6 DAgger + Gate | 93.8 → 85.4 | 97.9 → 91.7 | 80.8 → 76.7 | none |
| R7 f-DAgger | 94.8 → 88.5 | 97.4 → 96.4 | 75.7 → 67.6 | T1 −22.2 [−30.1, −14.4]; T1–T5 −10.6; all 240 −8.1 |
| R8 f-DAgger + Gate | 96.9 → 92.7 | 99.0 → 97.9 | 75.3 → 70.0 | T2 −25.0 [−34.7, −15.3]; T1–T5 −6.8; all 240 −5.3 |

What this means for the report:

- No row changes significantly on validation or T0.
- R8's v2 advantage on near targets (T2) came largely from the near-biased development scenes.
- For the comparisons between rows under v3, see [`../06_ablation/`](../06_ablation/README.md).

**Limitation.** Each version is one seed-0 collection chain, and seeds 1–2 only retrain on that chain's data. So the effect of the development scenes cannot be separated from the randomness of the chain. Read the table as "v2 versus v3 as run", not as a causal estimate.

## Reproduce

All commands run from `final/robotnik_arm_task/`. The v3 run itself is reproduced, for each row, with:

```bash
python -m experiments.dagger_rounds --row R8-dart-gate-dagger --rounds 3 --date 2026-11-05 --final-seeds 1 2 \
    --version v3 --variant-suffix=-nodev --dev-scenes 0 --workers 20
```

It was run on 2026-11-05 for the rows R8, R6, R7 and R5, in that order. `--variant-suffix=-nodev` needs the `=` because the value starts with `-`.

The run used a wrapper script that set these three values on `experiments/dagger_rounds.py` at run time. The options above now do the same thing, and `tests/datasets/test_dagger_rounds.py` checks that they produce the recorded names:

- variants `experiments/configs/datasets/variants/<row>-nodev-r<k>.yaml` (byte-identical files, whose SHA-256 is recorded in each checkpoint)
- datasets `dagger-R<n>-v3-r<k>`
- runs `<row>-nodev-r<k>__MLP01-256x3__s<seed>__v3__2026-11-05`

Rebuild the tables (no simulation):

```bash
python report/08_dev_scene_removal/code/make_tables.py
```

## Contents

```text
08_dev_scene_removal/
  code/make_tables.py        data/ -> tables/
  tables/  collections.csv   per version, row and round: collector, episodes (fitting / development), student holds
           rates.csv         success % with 95% scene-cluster bootstrap, R1-R4 (v1) and R5-R8 under v2 and v3
           dev_scene_effect.csv  v3 - v2 per row and metric, paired bootstrap CI, permutation p, Holm p
           summary.md        all of the above as Markdown
  data/    test/             rollouts, run and scene manifest of both unified test runs (v2 2026-11-01, v3 2026-11-05)
           validation/per_scene/  32-scene validation records: v1 R1-R4, v2 and v3 R5-R8 round 3, seeds 0-2
           dagger/           per-row DAgger summaries (validation per round and seed), v2 and v3
           collections/      the 24 DAgger collection manifests (v2, v3; 4 rows x 3 rounds)
           logs/             v3 run log and v3 extra-seed run log (saved as .txt because *.log is git-ignored)
```

All files in `data/` are unchanged copies:

- test files from `results/test/<run>/`
- v1 and v2 validation records and v2 manifests from [`../06_ablation/data/`](../06_ablation/data/)
- v3 validation records from `results/ablation/A/bcj/<run>/rollouts_validation_step42000_rollout-v1.csv`
- v3 manifests from `data/Markovian/DAgger/2026-11-05__collection__*/manifest.json`

Statistics follow [`../06_ablation/`](../06_ablation/README.md). The first version of this comparison (a scratch script, 2026-11-05) gave identical numbers.
