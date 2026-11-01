# Documentation

**Start with the [project pipeline](2026-10-30__guide__project-pipeline.md).** It covers the expert, data collection, splits, features, method, training, ablation, test results and the live demo for teacher A.
For commands, see the [project README](../README.md) (step-by-step reproduction of R4) and the [detailed reproduction reference](2026-10-29__guide__reproduction-details.md).

## By stage

| Stage | Documents |
|---|---|
| Planning | [Project overview](plans/2026-10-26__plan__project-overview.md) · [Milestones](plans/2026-10-24__roadmap__project-milestones.md) · [Ablation strategy](plans/2026-10-26__plan__ablation-strategy.md) · [Bibliography](references/2026-10-26__reference__research-bibliography.md) |
| Expert and data collection | [Expert model accuracy](reports/2026-10-24__report__markovian-expert-model-accuracy.md) · [Expert dataset spec](specs/2026-10-26__spec__expert-datasets.md) · [Clean data collection](reports/2026-10-26__report__clean-expert-data-collection.md) · [Expert operating limits](reports/2026-10-26__report__expert-operating-limits.md) · [Teacher A operating envelope](reports/2026-10-29__report__teacher-a-operating-envelope.md) |
| Datasets and splits | [Paired subsets Clean_253 / Clean_223](reports/2026-10-26__report__paired-clean-dataset-subsets.md) · [Split and evaluation predicate](plans/2026-10-26__plan__mlp-baseline-training.md) |
| Training | [Training code structure](specs/2026-10-26__design__mlp-training-structure.md) · [Training recipe research](research/2026-10-26__research__mlp-training-recipe.md) · [FK input accuracy](reports/2026-10-27__report__fk-input-accuracy.md) |
| Testing | [R4 test-set diagnosis](reports/2026-10-29__report__r4-test-diagnosis.md) |

Figures used by the pipeline document are in `figures/pipeline/`; regenerate them with `python -m evaluation.plot_pipeline_figures` from `final/robotnik_arm_task/`.
The course brief is `references/2026-10-24__assignment__comp0245-final-project.pdf` (local only, not in Git).

## Archived

Plans, reports and designs of earlier training versions (clean M1–M6, arm/wheel 0.8/0.2 retraining, D8-W256, DAgger R1) are in [`Archive/docs/`](../Archive/docs/). The full list is in [Archive/README.md](../Archive/README.md).

## Old document names

Source comments and the manifests of recorded data still use these old names. Those files feed code digests and dataset hashes, so they were not rewritten.

| Old name (`docs/…`) | Current location |
|---|---|
| `project_plan.md` | [plans/2026-10-26__plan__project-overview.md](plans/2026-10-26__plan__project-overview.md) |
| `milestones.md` | [plans/2026-10-24__roadmap__project-milestones.md](plans/2026-10-24__roadmap__project-milestones.md) |
| `ablation_strategy.md` | [plans/2026-10-26__plan__ablation-strategy.md](plans/2026-10-26__plan__ablation-strategy.md) |
| `mlp_baseline_training_plan.md` | [plans/2026-10-26__plan__mlp-baseline-training.md](plans/2026-10-26__plan__mlp-baseline-training.md) |
| `mlp_training_research.md` | [research/2026-10-26__research__mlp-training-recipe.md](research/2026-10-26__research__mlp-training-recipe.md) |
| `expert_datasets.md` | [specs/2026-10-26__spec__expert-datasets.md](specs/2026-10-26__spec__expert-datasets.md) |
| `expert_model_accuracy.md` | [reports/2026-10-24__report__markovian-expert-model-accuracy.md](reports/2026-10-24__report__markovian-expert-model-accuracy.md) |
| `expert_operating_limits.md` | [reports/2026-10-26__report__expert-operating-limits.md](reports/2026-10-26__report__expert-operating-limits.md) |
| `clean_collection_report.md` | [reports/2026-10-26__report__clean-expert-data-collection.md](reports/2026-10-26__report__clean-expert-data-collection.md) |
| `clean_dataset_subsets.md` | [reports/2026-10-26__report__paired-clean-dataset-subsets.md](reports/2026-10-26__report__paired-clean-dataset-subsets.md) |
| `references.md` | [references/2026-10-26__reference__research-bibliography.md](references/2026-10-26__reference__research-bibliography.md) |
