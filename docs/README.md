# Documentation

**For assessment and current commands, start with the [assessor's run guide](../README.md).** It covers installation, seed meanings, all 27 final model choices, live comparison, validation/unified testing and reproduction of R1–R9.
The [project pipeline](2026-10-30__guide__project-pipeline.md) explains the method; the [detailed reproduction reference](2026-10-29__guide__reproduction-details.md) preserves the earlier v1/v2 command and handoff details.

## Current status

| Experiment | Rows | Validation | Test evidence | Results |
|---|---|---|---|---|
| Training-method ablation v1 | R1–R4 | Done | Historical R4 diagnosis; all rows also tested under unified T0–T5 | `results/ablation/2026-10-28__training-ablation-v1/` |
| Training-method ablation v3 (DAgger, no development scenes) | R5–R8 | Done (2026-11-05); **used by the report and release** | Unified T0–T5 test completed; see report evidence | `results/ablation/2026-11-05__training-ablation-v3/` |
| Training-method ablation v2 (DAgger + 64 development scenes) | R5–R8 | Done (2026-11-01); superseded by v3 | Unified T0–T5 test completed; compared with v3 in [`08_dev_scene_removal`](../final/robotnik_arm_task/report/08_dev_scene_removal/README.md) | `results/ablation/2026-11-01__training-ablation-v2/` |
| v2 development run (teammate) | R5–R8, plus R4 controls | Done on a teammate's machine (record only) | Not used | `results/ablation/2026-10-30__training-ablation-v2/` |

The [final release](../final/robotnik_arm_task/checkpoints/2026-11-05final/README.md) contains R1–R9, each with seeds 0/1/2; R5–R9 use round 3. Report evidence for the completed unified tests is in [the ablation section](../final/robotnik_arm_task/report/06_ablation/README.md), and the full DART comparison is in [the training-time comparison](../final/robotnik_arm_task/report/07_training_time_comparison/README.md). Current row names are in the [model naming guide](model-naming.md).

## By stage

| Stage | Documents |
|---|---|
| Expert and data collection | [Expert model accuracy](reports/2026-10-24__report__markovian-expert-model-accuracy.md) · [Expert dataset spec](specs/2026-10-26__spec__expert-datasets.md) · [Clean data collection](reports/2026-10-26__report__clean-expert-data-collection.md) · [Expert operating limits](reports/2026-10-26__report__expert-operating-limits.md) · [Teacher A operating envelope](reports/2026-10-29__report__teacher-a-operating-envelope.md) |
| Datasets and splits | [Paired subsets Clean_253 / Clean_223](reports/2026-10-26__report__paired-clean-dataset-subsets.md) · split and success predicate: pipeline Sections 2–3 |
| Training | Recipe and loss: pipeline Section 6 · [FK input accuracy](reports/2026-10-27__report__fk-input-accuracy.md) |
| Ablation v1 (R1–R4) | Pipeline Section 7 · [R4 test-set diagnosis](reports/2026-10-29__report__r4-test-diagnosis.md) |
| Ablation v2 (R5–R8, DAgger) | Pipeline Section 7.2 · [reference-run summaries](../final/robotnik_arm_task/results/ablation/2026-11-01__training-ablation-v2/README.md) · [DAgger ablation report (development run)](reports/2026-10-31__report__dagger-ablation-and-final-model.md) · [development-run summaries and legacy names](../final/robotnik_arm_task/results/ablation/2026-10-30__training-ablation-v2/README.md) |
| References | [Bibliography](references/2026-10-26__reference__research-bibliography.md) |

Figures used by the pipeline document are in `figures/pipeline/`; regenerate them with `python -m evaluation.plot_pipeline_figures` from `final/robotnik_arm_task/`.
The course brief is `references/2026-10-24__assignment__comp0245-final-project.pdf` (local only, not in Git).

## Conventions

- File names: `YYYY-MM-DD__<type>__<topic>.md`. Types: `guide` (docs root), `spec`, `report`, `reference`. Archived types also include `plan`, `roadmap`, `research` and `design`.
- Each document opens with a `#` title and a line `Date: … Status: …`.
- Paths are relative to `final/robotnik_arm_task/` unless stated otherwise. Success counts are written `held/total`; pooled seeds add a Wilson 95% interval.
- Numbers come from the reference machine (the one that produced the v1 models) unless a document says otherwise.

## Archived

Earlier plans, research notes and designs (24–26 October: project plan, milestones, ablation strategy, clean MLP baseline plan, training recipe research, training code structure) and the reports of earlier training versions (clean M1–M6, arm/wheel 0.8/0.2 retraining, D8-W256, the 27 October DAgger round 1) are kept in a local `Archive/` folder together with the retired code and models. `Archive/` is not part of the submission. Mentions of `Archive/...` paths in documents, configs and report READMEs record provenance only. Everything needed to run or check the submitted work is outside it.

## Old document names

Source comments and the manifests of recorded data still use these old names. Those files feed code digests and dataset hashes, so they were not rewritten.

| Old name (`docs/…`) | Current location |
|---|---|
| `project_plan.md`, `milestones.md`, `ablation_strategy.md`, `mlp_baseline_training_plan.md`, `mlp_training_research.md` | Archived locally (not submitted) |
| `expert_datasets.md` | [specs/2026-10-26__spec__expert-datasets.md](specs/2026-10-26__spec__expert-datasets.md) |
| `expert_model_accuracy.md` | [reports/2026-10-24__report__markovian-expert-model-accuracy.md](reports/2026-10-24__report__markovian-expert-model-accuracy.md) |
| `expert_operating_limits.md` | [reports/2026-10-26__report__expert-operating-limits.md](reports/2026-10-26__report__expert-operating-limits.md) |
| `clean_collection_report.md` | [reports/2026-10-26__report__clean-expert-data-collection.md](reports/2026-10-26__report__clean-expert-data-collection.md) |
| `clean_dataset_subsets.md` | [reports/2026-10-26__report__paired-clean-dataset-subsets.md](reports/2026-10-26__report__paired-clean-dataset-subsets.md) |
| `references.md` | [references/2026-10-26__reference__research-bibliography.md](references/2026-10-26__reference__research-bibliography.md) |
