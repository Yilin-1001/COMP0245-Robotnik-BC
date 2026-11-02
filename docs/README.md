# Documentation

**Start with the [project pipeline](2026-10-30__guide__project-pipeline.md).** It covers the expert, data collection, splits, features, method, training, ablation (R1–R8), test results and the live demo for teacher A.
For commands, see the [project README](../README.md) (step-by-step reproduction of R4) and the [detailed reproduction reference](2026-10-29__guide__reproduction-details.md).

## Current status

| Experiment | Rows | Validation | Test (64 scenarios) | Results |
|---|---|---|---|---|
| Training-method ablation v1 | R1–R4 | Done | R4 only (diagnostic, before DAgger) | `results/ablation/2026-10-28__training-ablation-v1/` |
| Training-method ablation v2 (DAgger) | R5–R8 | Done (reference run, 2026-11-01) | **Not yet run** | `results/ablation/2026-11-01__training-ablation-v2/` |
| v2 development run (teammate) | R5–R8, plus R4 controls | Done on a teammate's machine (record only) | Not used | `results/ablation/2026-10-30__training-ablation-v2/` |

The final model has not been chosen yet. Row names are defined in the pipeline document, [Section 7](2026-10-30__guide__project-pipeline.md#7-ablation-experiment-design).

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

Earlier plans, research notes and designs (24–26 October: project plan, milestones, ablation strategy, clean MLP baseline plan, training recipe research, training code structure) and the reports of earlier training versions (clean M1–M6, arm/wheel 0.8/0.2 retraining, D8-W256, the 27 October DAgger round 1) are in [`Archive/docs/`](../Archive/docs/). The full list is in [Archive/README.md](../Archive/README.md).

## Old document names

Source comments and the manifests of recorded data still use these old names. Those files feed code digests and dataset hashes, so they were not rewritten.

| Old name (`docs/…`) | Current location |
|---|---|
| `project_plan.md` | [Archive/docs/plans/2026-10-26__plan__project-overview.md](../Archive/docs/plans/2026-10-26__plan__project-overview.md) |
| `milestones.md` | [Archive/docs/plans/2026-10-24__roadmap__project-milestones.md](../Archive/docs/plans/2026-10-24__roadmap__project-milestones.md) |
| `ablation_strategy.md` | [Archive/docs/plans/2026-10-26__plan__ablation-strategy.md](../Archive/docs/plans/2026-10-26__plan__ablation-strategy.md) |
| `mlp_baseline_training_plan.md` | [Archive/docs/plans/2026-10-26__plan__mlp-baseline-training.md](../Archive/docs/plans/2026-10-26__plan__mlp-baseline-training.md) |
| `mlp_training_research.md` | [Archive/docs/research/2026-10-26__research__mlp-training-recipe.md](../Archive/docs/research/2026-10-26__research__mlp-training-recipe.md) |
| `expert_datasets.md` | [specs/2026-10-26__spec__expert-datasets.md](specs/2026-10-26__spec__expert-datasets.md) |
| `expert_model_accuracy.md` | [reports/2026-10-24__report__markovian-expert-model-accuracy.md](reports/2026-10-24__report__markovian-expert-model-accuracy.md) |
| `expert_operating_limits.md` | [reports/2026-10-26__report__expert-operating-limits.md](reports/2026-10-26__report__expert-operating-limits.md) |
| `clean_collection_report.md` | [reports/2026-10-26__report__clean-expert-data-collection.md](reports/2026-10-26__report__clean-expert-data-collection.md) |
| `clean_dataset_subsets.md` | [reports/2026-10-26__report__paired-clean-dataset-subsets.md](reports/2026-10-26__report__paired-clean-dataset-subsets.md) |
| `references.md` | [references/2026-10-26__reference__research-bibliography.md](references/2026-10-26__reference__research-bibliography.md) |
